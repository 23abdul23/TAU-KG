"""
src/paper_entity_extractor.py
=============================
GPT-4 based entity extraction from scientific papers.

Extracts:
- Genes
- Proteins  
- Diseases
- Pathways
- Relationships between entities

Features:
- Confidence scores for each extraction
- Validation against existing deb_data entities
- Relationship extraction with evidence
"""

import os
import json
import re
import importlib.util
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional, Iterable
import logging
from pathlib import Path
from threading import Lock
from src.llm_provider import LLMClient
from src.paper_schema import (
    canonicalize_edge_type,
    normalize_entities_for_paper,
    normalize_entity_type,
    normalize_relationships_for_paper,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)

from src.pmc_service import normalize_section_type

if load_dotenv:
    load_dotenv()


DEFAULT_EXTRACTION_MAX_CHARS = 9000
EXTRACTION_SECTION_PRIORITY = [
    "abstract",
    "results",
    "discussion",
    "conclusion",
    "introduction",
    "methods",
]
EXTRACTION_SECTION_BUDGETS = {
    "abstract": 0.20,
    "results": 0.30,
    "discussion": 0.25,
    "conclusion": 0.10,
    "introduction": 0.10,
    "methods": 0.05,
}
EXTRACTION_FALLBACK_BUCKETS = ["supplementary", "other"]
SCISPACY_DEFAULT_MODEL = "en_ner_bionlp13cg_md"
SCISPACY_DEFAULT_CHUNK_SIZE = 50000
SCISPACY_ENTITY_LABEL_MAP = {
    "GENE_OR_GENE_PRODUCT": ("gene_or_protein", 0.83),
    "DISEASE": ("disease", 0.88),
}
RELATIONSHIP_CONTEXT_CHARS = 220
NER_PIPELINE_DISEASE_CONFIDENCE = 0.88
NER_PIPELINE_GENE_PROTEIN_CONFIDENCE = 0.83
NER_PIPELINE_PATHWAY_CONFIDENCE = 0.76
_MODEL_WARMUP_LOCK = Lock()
_MODEL_WARMUP_DONE = False


def get_llm_client() -> LLMClient:
    """Initialize and return the selected LLM client."""
    client = LLMClient()
    if not client.is_available():
        raise ValueError(client.unavailable_reason or "LLM client unavailable")
    return client


def _get_max_extraction_chars() -> int:
    """Read the configurable section-aware context size budget from the environment."""
    raw_value = os.getenv("PAPER_ENTITY_EXTRACTION_MAX_CHARS", "").strip()
    if not raw_value:
        return DEFAULT_EXTRACTION_MAX_CHARS

    try:
        parsed_value = int(raw_value)
        return max(1000, parsed_value)
    except ValueError:
        logger.warning(
            "Invalid PAPER_ENTITY_EXTRACTION_MAX_CHARS=%r. Using default %s.",
            raw_value,
            DEFAULT_EXTRACTION_MAX_CHARS,
        )
        return DEFAULT_EXTRACTION_MAX_CHARS


def _truncate_text(text: str, max_chars: int) -> str:
    """Trim text to a stable boundary without cutting far past the requested budget."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_boundary = max(
        truncated.rfind("\n"),
        truncated.rfind(". "),
        truncated.rfind("; "),
        truncated.rfind(", "),
        truncated.rfind(" "),
    )
    if last_boundary >= int(max_chars * 0.7):
        truncated = truncated[:last_boundary]
    return truncated.strip()


def _join_bucket_texts(parts: List[str]) -> str:
    """Combine unique section fragments while keeping their original order."""
    seen = set()
    ordered_parts = []
    for part in parts:
        clean_part = (part or "").strip()
        if not clean_part or clean_part in seen:
            continue
        seen.add(clean_part)
        ordered_parts.append(clean_part)
    return "\n\n".join(ordered_parts)


def _get_extraction_backend() -> str:
    raw_backend = os.getenv("PAPER_ENTITY_EXTRACTION_BACKEND", "hybrid").strip().lower()
    if raw_backend in {"llm", "scispacy", "hybrid"}:
        return raw_backend
    logger.warning(
        "Invalid PAPER_ENTITY_EXTRACTION_BACKEND=%r. Falling back to 'hybrid'.",
        raw_backend,
    )
    return "hybrid"


def _iter_text_chunks(text: str, chunk_size: int) -> Iterable[Tuple[int, str]]:
    clean_text = str(text or "")
    for index in range(0, len(clean_text), max(1000, chunk_size)):
        yield index, clean_text[index:index + chunk_size]


@lru_cache(maxsize=1)
def _get_scispacy_nlp():
    try:
        import spacy
    except Exception as exc:
        logger.warning("spaCy import failed; SciSpaCy backend disabled: %s", exc)
        return None

    model_name = os.getenv("PAPER_SCISPACY_MODEL", SCISPACY_DEFAULT_MODEL).strip() or SCISPACY_DEFAULT_MODEL
    try:
        return spacy.load(model_name)
    except Exception as exc:
        logger.warning(
            "Unable to load SciSpaCy model '%s'; backend disabled: %s",
            model_name,
            exc,
        )
        return None


def _is_probable_sequence(name: str) -> bool:
    token = re.sub(r"[^A-Za-z]", "", name or "")
    if len(token) < 10:
        return False
    base_count = sum(token.upper().count(ch) for ch in "ACGTN")
    return (base_count / max(1, len(token))) >= 0.85


def _is_valid_entity_mention(name: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip()
    if len(cleaned) < 2 or len(cleaned) > 80:
        return False
    if cleaned.lower() in {"co2", "co 2", "dna", "rna"}:
        return False
    if _is_probable_sequence(cleaned):
        return False
    if re.search(r"\b(?:table|figure|fig\.?|supplementary|copyright)\b", cleaned, re.IGNORECASE):
        return False
    if re.fullmatch(r"[A-Z0-9 _\-:/]{20,}", cleaned) and " " not in cleaned.strip():
        return False
    return True


def _infer_gene_or_protein_bucket(name: str, context: str) -> str:
    candidate = f"{name} {context}".lower()
    if re.search(r"\b(protein|peptide|kinase|phosphatase|receptor|enzyme|antibody)\b", candidate):
        return "proteins"
    if re.fullmatch(r"[A-Z0-9\-]{2,12}", name or ""):
        return "genes"
    return "genes"


def _extract_context_window(text: str, start: int, end: int, radius: int = RELATIONSHIP_CONTEXT_CHARS) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    window = text[left:right].strip()
    return re.sub(r"\s+", " ", window)


def _is_valid_pathway_name(name: str) -> bool:
    clean_name = re.sub(r"\s+", " ", str(name or "")).strip(" -;,.")
    if len(clean_name) < 5 or len(clean_name) > 120:
        return False
    if clean_name.lower() in {"pathway", "signaling pathway", "signalling pathway"}:
        return False
    tokens = clean_name.split()
    if len(tokens) > 8:
        return False
    if tokens[0].lower() in {"the", "a", "an", "disease", "alzheimer", "parkinson"}:
        return False
    if re.search(r"\b(involves?|involved|regulates?|activates?|inhibits?|causes?|associated|showed|shows)\b", clean_name, re.IGNORECASE):
        return False
    return True


def _normalize_pathway_candidate(raw_name: str) -> str:
    tokens = re.sub(r"\s+", " ", str(raw_name or "")).strip(" -;,.").split(" ")
    if not tokens:
        return ""

    lowering = [tok.lower() for tok in tokens]
    relation_words = {"involves", "involve", "involved", "regulates", "regulate", "activates", "activate", "inhibits", "inhibit", "associated"}
    for idx, tok in enumerate(lowering):
        if tok in relation_words and idx < len(tokens) - 1:
            tokens = tokens[idx + 1:]
            lowering = lowering[idx + 1:]
            break

    if "signaling" in lowering:
        idx = lowering.index("signaling")
        start = max(0, idx - 2)
        tokens = tokens[start:]
    elif "signalling" in lowering:
        idx = lowering.index("signalling")
        start = max(0, idx - 2)
        tokens = tokens[start:]
    elif "pathway" in lowering and len(tokens) > 4:
        idx = lowering.index("pathway")
        start = max(0, idx - 3)
        tokens = tokens[start:]

    return " ".join(tokens).strip(" -;,.")


def _extract_pathway_entities(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    patterns = [
        re.compile(
            r"\b([A-Za-z0-9\-/]+(?:\s+[A-Za-z0-9\-/]+){0,3}\s+(?:signaling|signalling)\s+pathway)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Za-z0-9\-/]+(?:\s+[A-Za-z0-9\-/]+){0,2}\s+(?:pathway|signaling cascade|signalling cascade|signaling axis|signalling axis))\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(autophagy|neuroinflammation|mitochondrial dysfunction|oxidative stress)\b", re.IGNORECASE),
    ]

    seen = set()
    extracted = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            name = _normalize_pathway_candidate(match.group(1))
            normalized_key = name.lower()
            if not _is_valid_pathway_name(name) or normalized_key in seen:
                continue
            seen.add(normalized_key)
            extracted.append(
                {
                    "name": name,
                    "context": _extract_context_window(text, match.start(), match.end()),
                    "confidence": 0.72,
                    "entity_type": "pathway",
                    "chromosome": "",
                }
            )
    return extracted


def _dedupe_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for entity in entities:
        name = str(entity.get("name", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def _scispacy_extract_entities(
    paper_text: str,
    sections: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    nlp = _get_scispacy_nlp()
    if nlp is None:
        return {
            "genes": [],
            "proteins": [],
            "diseases": [],
            "pathways": _extract_pathway_entities(paper_text),
        }

    chunk_size = int(os.getenv("PAPER_SCISPACY_CHUNK_SIZE", str(SCISPACY_DEFAULT_CHUNK_SIZE)) or SCISPACY_DEFAULT_CHUNK_SIZE)
    texts_with_section = []
    if sections:
        for section in sections:
            section_text = str(section.get("text", "")).strip()
            if section_text:
                texts_with_section.append((normalize_section_type(section.get("type", "")) or "other", section_text))
    if not texts_with_section:
        texts_with_section = [("full_text", paper_text)]

    extracted = {"genes": [], "proteins": [], "diseases": [], "pathways": []}
    for _, text in texts_with_section:
        for _, chunk in _iter_text_chunks(text, chunk_size):
            if not chunk.strip():
                continue
            doc = nlp(chunk)
            for ent in doc.ents:
                mapping = SCISPACY_ENTITY_LABEL_MAP.get(ent.label_)
                if not mapping:
                    continue
                normalized_kind, base_confidence = mapping
                name = re.sub(r"\s+", " ", ent.text).strip()
                if not _is_valid_entity_mention(name):
                    continue
                entity_record = {
                    "name": name,
                    "context": _extract_context_window(chunk, ent.start_char, ent.end_char),
                    "confidence": float(base_confidence),
                    "chromosome": "",
                }
                if normalized_kind == "gene_or_protein":
                    bucket = _infer_gene_or_protein_bucket(name, entity_record["context"])
                    entity_record["entity_type"] = "protein" if bucket == "proteins" else "gene"
                    extracted[bucket].append(entity_record)
                elif normalized_kind == "disease":
                    entity_record["entity_type"] = "disease"
                    extracted["diseases"].append(entity_record)

    extracted["pathways"].extend(_extract_pathway_entities(paper_text))
    for bucket in ["genes", "proteins", "diseases", "pathways"]:
        extracted[bucket] = _dedupe_entities(extracted[bucket])

    logger.info(
        "SciSpaCy extracted entities: genes=%s proteins=%s diseases=%s pathways=%s",
        len(extracted["genes"]),
        len(extracted["proteins"]),
        len(extracted["diseases"]),
        len(extracted["pathways"]),
    )
    return extracted


def _infer_sentence_relationships(
    paper_text: str,
    entities_by_type: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    text = str(paper_text or "")
    if not text:
        return []

    entity_lookup: Dict[str, Dict[str, str]] = {}
    for bucket, values in entities_by_type.items():
        type_name = {
            "genes": "Gene",
            "proteins": "Protein",
            "diseases": "Disease",
            "pathways": "Pathway",
        }.get(bucket, "Entity")
        for entity in values:
            name = str(entity.get("name", "")).strip()
            if len(name) < 2:
                continue
            entity_lookup[name.lower()] = {"name": name, "type": type_name}

    if len(entity_lookup) < 2:
        return []

    sentence_pattern = re.compile(r"(?<=[\.\?\!])\s+")
    relation_scores: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for sentence in sentence_pattern.split(text):
        clean_sentence = re.sub(r"\s+", " ", sentence).strip()
        if len(clean_sentence) < 25:
            continue

        found: List[Dict[str, str]] = []
        lowered_sentence = clean_sentence.lower()
        for key, record in entity_lookup.items():
            if len(key) <= 4:
                if not re.search(rf"\b{re.escape(key)}\b", lowered_sentence):
                    continue
            elif key not in lowered_sentence:
                continue
            found.append(record)

        if len(found) < 2:
            continue

        unique_found = []
        seen_names = set()
        for record in found:
            name_key = record["name"].lower()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            unique_found.append(record)

        for i in range(len(unique_found)):
            for j in range(i + 1, len(unique_found)):
                source = unique_found[i]
                target = unique_found[j]
                edge_type = canonicalize_edge_type(clean_sentence, source["type"], target["type"])
                pair_key = (source["name"].lower(), target["name"].lower(), edge_type)
                score_bonus = 0.12 if edge_type != "ASSOCIATES" else 0.0
                score = min(0.92, 0.56 + score_bonus + (0.02 * min(len(unique_found), 4)))

                current = relation_scores.get(pair_key)
                if not current or score > float(current["edge_weight"]):
                    relation_scores[pair_key] = {
                        "source_name": source["name"],
                        "source_type": source["type"],
                        "target_name": target["name"],
                        "target_type": target["type"],
                        "edge_type": edge_type,
                        "edge_weight": round(score, 3),
                        "evidence": clean_sentence[:700],
                        "extraction_method": "scispacy_sentence_cooccurrence",
                    }

    ordered = sorted(
        relation_scores.values(),
        key=lambda item: (float(item.get("edge_weight", 0.0)), len(str(item.get("evidence", "")))),
        reverse=True,
    )
    return ordered[:80]


def _merge_extraction_results(
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    merged = {
        "genes": _dedupe_entities((primary.get("genes", []) or []) + (fallback.get("genes", []) or [])),
        "proteins": _dedupe_entities((primary.get("proteins", []) or []) + (fallback.get("proteins", []) or [])),
        "diseases": _dedupe_entities((primary.get("diseases", []) or []) + (fallback.get("diseases", []) or [])),
        "pathways": _dedupe_entities((primary.get("pathways", []) or []) + (fallback.get("pathways", []) or [])),
        "relationships": (primary.get("relationships", []) or []) + (fallback.get("relationships", []) or []),
    }
    return merged


def _parse_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        logger.warning("Invalid %s=%r. Using default %s.", name, raw_value, default)
        return default


def _parse_float_env(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = float(raw_value)
        return max(minimum, min(maximum, parsed))
    except ValueError:
        logger.warning("Invalid %s=%r. Using default %s.", name, raw_value, default)
        return default


def _load_module_from_path(module_name: str, module_path: Path) -> Optional[Any]:
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_full_ner_pipeline_modules() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load the full NER_extracion pipeline stages from local modules.

    We first look for src-local modules, then fall back to test helpers to stay
    compatible with the existing repository layout.
    """
    repo_root = Path(__file__).resolve().parent.parent

    scispacy_module = (
        _load_module_from_path("src_scispacy_pipeline", repo_root / "src" / "scispacy_pipeline.py")
        or _load_module_from_path("test_scispacy_pipeline", repo_root / "test" / "scispacy_pipeline.py")
    )
    biobert_module = (
        _load_module_from_path("src_biobert_refinement", repo_root / "src" / "biobert_refinement.py")
        or _load_module_from_path("test_biobert_refinement", repo_root / "test" / "biobert_refinement.py")
    )

    if scispacy_module is None:
        logger.warning(
            "Full NER pipeline SciSpaCy stage not found under src/ or test/. Falling back to local extractor."
        )
    if biobert_module is None:
        logger.warning(
            "Full NER pipeline BioBERT refinement stage not found under src/ or test/. "
            "Using SciSpaCy stage output only."
        )
    return scispacy_module, biobert_module


def warmup_entity_extraction_models() -> Dict[str, Any]:
    """
    Best-effort one-time model warmup for concurrent extraction workloads.

    This avoids repeated cold starts when many PMC extraction jobs arrive at once.
    """
    global _MODEL_WARMUP_DONE

    with _MODEL_WARMUP_LOCK:
        if _MODEL_WARMUP_DONE:
            return {"status": "ready", "cached": True}

        backend = _get_extraction_backend()
        details: Dict[str, Any] = {
            "status": "ready",
            "cached": False,
            "backend": backend,
            "scispacy": "not_attempted",
            "biobert": "not_attempted",
        }

        if backend == "llm":
            _MODEL_WARMUP_DONE = True
            details["note"] = "LLM backend selected; local NER models were not warmed."
            return details

        scispacy_module, biobert_module = _load_full_ner_pipeline_modules()

        try:
            if scispacy_module is not None and hasattr(scispacy_module, "load_models"):
                scispacy_module.load_models()
                details["scispacy"] = "ready"
            elif _get_scispacy_nlp() is not None:
                details["scispacy"] = "ready"
            else:
                details["scispacy"] = "unavailable"
        except Exception as exc:
            details["scispacy"] = f"error:{exc}"
            logger.warning("SciSpaCy warmup failed: %s", exc)

        try:
            if biobert_module is not None and hasattr(biobert_module, "load_model"):
                model_name_default = str(getattr(biobert_module, "DEFAULT_MODEL_NAME", "d4data/biomedical-ner-all"))
                biobert_module.load_model(
                    model_name=os.getenv("PAPER_BIOBERT_MODEL_NAME", "").strip() or model_name_default,
                    device=None,
                )
                details["biobert"] = "ready"
            elif biobert_module is None:
                details["biobert"] = "unavailable"
            else:
                details["biobert"] = "unsupported_module"
        except Exception as exc:
            details["biobert"] = f"error:{exc}"
            logger.warning("BioBERT warmup failed: %s", exc)

        _MODEL_WARMUP_DONE = True
        return details


def _convert_full_pipeline_categories(
    categories: Dict[str, Any],
    paper_text: str,
) -> Dict[str, List[Dict[str, Any]]]:
    converted = {
        "genes": [],
        "proteins": [],
        "diseases": [],
        "pathways": [],
    }

    for entity in categories.get("diseases", []) or []:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("text", "")).strip()
        context = str(entity.get("evidence_text", "")).strip() or name
        if not name:
            continue
        converted["diseases"].append(
            {
                "name": name,
                "context": context,
                "confidence": NER_PIPELINE_DISEASE_CONFIDENCE,
                "entity_type": "disease",
                "chromosome": "",
            }
        )

    for entity in categories.get("genes_proteins", []) or []:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("text", "")).strip()
        context = str(entity.get("evidence_text", "")).strip() or name
        if not name:
            continue
        bucket = _infer_gene_or_protein_bucket(name, context)
        converted[bucket].append(
            {
                "name": name,
                "context": context,
                "confidence": NER_PIPELINE_GENE_PROTEIN_CONFIDENCE,
                "entity_type": "protein" if bucket == "proteins" else "gene",
                "chromosome": "",
            }
        )

    for entity in categories.get("pathways", []) or []:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("text", "")).strip()
        context = str(entity.get("evidence_text", "")).strip() or name
        if not name:
            continue
        converted["pathways"].append(
            {
                "name": name,
                "context": context,
                "confidence": NER_PIPELINE_PATHWAY_CONFIDENCE,
                "entity_type": "pathway",
                "chromosome": "",
            }
        )

    # Keep the lightweight regex-based pathway fallback to maintain prior recall.
    converted["pathways"].extend(_extract_pathway_entities(paper_text))

    for bucket in ["genes", "proteins", "diseases", "pathways"]:
        converted[bucket] = _dedupe_entities(converted[bucket])

    return converted


def _extract_entities_with_full_ner_pipeline(
    paper_text: str,
) -> Dict[str, List[Dict[str, Any]]]:
    scispacy_module, biobert_module = _load_full_ner_pipeline_modules()
    if scispacy_module is None:
        return _scispacy_extract_entities(paper_text)

    try:
        scispacy_entities, scispacy_timings = scispacy_module.main_pipeline(paper_text)
        logger.info(
            "Full pipeline SciSpaCy stage complete: diseases=%s genes_proteins=%s pathways=%s",
            len(scispacy_entities.get("diseases", []) or []),
            len(scispacy_entities.get("genes_proteins", []) or []),
            len(scispacy_entities.get("pathways", []) or []),
        )
        if scispacy_timings:
            logger.debug("SciSpaCy timings: %s", scispacy_timings)
    except Exception as exc:
        logger.warning("Full pipeline SciSpaCy stage failed, falling back to local extractor: %s", exc)
        return _scispacy_extract_entities(paper_text)

    categories = scispacy_entities
    if biobert_module is not None:
        try:
            model_name_default = str(getattr(biobert_module, "DEFAULT_MODEL_NAME", "d4data/biomedical-ner-all"))
            context_window_default = int(getattr(biobert_module, "DEFAULT_CONTEXT_WINDOW", 80))
            batch_size_default = int(getattr(biobert_module, "DEFAULT_BATCH_SIZE", 32))
            confidence_default = float(getattr(biobert_module, "DEFAULT_CONFIDENCE_THRESHOLD", 0.65))
            span_default = float(getattr(biobert_module, "DEFAULT_SPAN_THRESHOLD", 0.55))

            categories, refinement_report, refinement_timings = biobert_module.main_pipeline(
                text=paper_text,
                input_json=scispacy_entities,
                model_name=os.getenv("PAPER_BIOBERT_MODEL_NAME", "").strip() or model_name_default,
                context_window=_parse_int_env("PAPER_BIOBERT_CONTEXT_WINDOW", context_window_default),
                batch_size=_parse_int_env("PAPER_BIOBERT_BATCH_SIZE", batch_size_default),
                confidence_threshold=_parse_float_env("PAPER_BIOBERT_CONFIDENCE_THRESHOLD", confidence_default),
                span_threshold=_parse_float_env("PAPER_BIOBERT_SPAN_THRESHOLD", span_default),
                device=None,
            )
            logger.info(
                "Full pipeline BioBERT stage complete: diseases=%s genes_proteins=%s pathways=%s",
                len(categories.get("diseases", []) or []),
                len(categories.get("genes_proteins", []) or []),
                len(categories.get("pathways", []) or []),
            )
            if refinement_timings:
                logger.debug("BioBERT timings: %s", refinement_timings)
            if refinement_report:
                logger.debug("BioBERT report summary: %s", refinement_report.get("summary", {}))
        except Exception as exc:
            logger.warning("Full pipeline BioBERT refinement failed, keeping SciSpaCy entities: %s", exc)

    return _convert_full_pipeline_categories(categories, paper_text)


def build_entity_extraction_context(
    paper_text: str,
    sections: Optional[List[Dict[str, Any]]] = None,
    abstract: str = "",
    max_chars: Optional[int] = None,
) -> str:
    """Build a section-aware extraction context under a configurable char budget."""
    extraction_limit = max_chars or _get_max_extraction_chars()
    if not sections:
        return _truncate_text(paper_text, extraction_limit)

    bucketed_sections: Dict[str, List[str]] = {bucket: [] for bucket in EXTRACTION_SECTION_PRIORITY}
    bucketed_sections["supplementary"] = []
    bucketed_sections["other"] = []

    if abstract.strip():
        bucketed_sections["abstract"].append(abstract.strip())

    for section in sections:
        section_type = normalize_section_type(section.get("type", ""))
        section_text = (section.get("text") or "").strip()
        if not section_text or section_type == "metadata":
            continue

        target_bucket = section_type if section_type in bucketed_sections else "other"
        bucketed_sections[target_bucket].append(section_text)

    bucket_texts = {
        bucket: _join_bucket_texts(parts)
        for bucket, parts in bucketed_sections.items()
    }

    if not any(bucket_texts.values()):
        return _truncate_text(paper_text, extraction_limit)

    selected_texts = {bucket: "" for bucket in bucket_texts}
    remaining_chars = extraction_limit

    for bucket in EXTRACTION_SECTION_PRIORITY:
        bucket_text = bucket_texts.get(bucket, "")
        if not bucket_text or remaining_chars <= 0:
            continue

        bucket_budget = int(extraction_limit * EXTRACTION_SECTION_BUDGETS[bucket])
        if bucket_budget <= 0:
            continue

        selected_text = _truncate_text(bucket_text, min(bucket_budget, remaining_chars))
        selected_texts[bucket] = selected_text
        remaining_chars -= len(selected_text)

    redistribution_order = EXTRACTION_SECTION_PRIORITY + EXTRACTION_FALLBACK_BUCKETS
    for bucket in redistribution_order:
        if remaining_chars <= 0:
            break

        bucket_text = bucket_texts.get(bucket, "")
        already_selected = selected_texts.get(bucket, "")
        if not bucket_text:
            continue

        remaining_bucket_text = bucket_text[len(already_selected):].strip()
        if not remaining_bucket_text:
            continue

        extra_text = _truncate_text(remaining_bucket_text, remaining_chars)
        if not extra_text:
            continue

        if already_selected:
            selected_texts[bucket] = f"{already_selected}\n\n{extra_text}".strip()
        else:
            selected_texts[bucket] = extra_text
        remaining_chars -= len(extra_text)

    ordered_parts = []
    for bucket in redistribution_order:
        bucket_text = selected_texts.get(bucket, "").strip()
        if bucket_text:
            ordered_parts.append(f"{bucket.upper()}:\n{bucket_text}")

    merged_context = "\n\n".join(ordered_parts).strip()
    return _truncate_text(merged_context or paper_text, extraction_limit)


def build_extraction_prompt(
    paper_text: str,
    title: str = "",
    abstract: str = "",
    sections: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """
    Build GPT-4 prompt for entity extraction from paper.
    
    Args:
        paper_text: Full paper text
        title: Paper title (optional)
        abstract: Paper abstract (optional)
    
    Returns:
        str: Formatted prompt for GPT-4
    """
    system_prompt = """You are an expert biomedical text mining system. Your task is to extract biological entities and relationships from scientific papers.

Extract ONLY these entity types:
1. GENES - gene names and symbols (e.g., MAPT, TP53, BRCA1)
2. PROTEINS - protein names (e.g., Tau Protein, Amyloid-beta, Alpha-synuclein)
3. DISEASES - disease names (e.g., Alzheimer's Disease, Parkinson's Disease)
4. PATHWAYS - biological pathways (e.g., Autophagy, Neuroinflammation, Mitochondrial dysfunction)

For EACH extracted entity, provide:
- name: exact entity name from text
- context: 1-2 sentence snippet showing entity in context
- confidence: 0.0-1.0 confidence score based on clarity and evidence in paper
- chromosome: chromosome if explicitly stated for genes, otherwise ""

Also extract KEY RELATIONSHIPS:
- source entity -> relationship -> target entity
- source_type and target_type must be one of Gene, Protein, Disease, Pathway, Drug, Tissue, Entity
- edge_type must be one of ENCODES, EXPRESSES, INTERACTS, TREATS, PARTICIPATES, ACTIVATES, INHIBITS, REGULATES, ASSOCIATES
- edge_weight: 0.0-1.0 score
- evidence: supporting sentence from paper

OUTPUT FORMAT (JSON):
{
    "genes": [
        {"name": "MAPT", "context": "MAPT mutations cause...", "confidence": 0.95, "chromosome": "17"},
        ...
    ],
    "proteins": [
        {"name": "Tau Protein", "context": "...", "confidence": 0.90},
        ...
    ],
    "diseases": [
        {"name": "Alzheimer's Disease", "context": "...", "confidence": 0.92},
        ...
    ],
    "pathways": [
        {"name": "Autophagy", "context": "...", "confidence": 0.88},
        ...
    ],
    "relationships": [
        {
            "source_name": "MAPT",
            "source_type": "Gene",
            "target_name": "Tau Protein",
            "target_type": "Protein",
            "edge_type": "ENCODES",
            "edge_weight": 0.87,
            "evidence": "The MAPT gene encodes the tau protein..."
        },
        ...
    ]
}

IMPORTANT:
- Be precise: only extract entities clearly mentioned in text
- High confidence (>0.8) only if explicitly discussed
- Lower confidence (0.5-0.8) for inferred or indirect mentions
- Include up to 20 entities per type MAX
- Extract top 10 relationships only
- Return valid JSON only, no other text"""

    extraction_context = build_entity_extraction_context(
        paper_text,
        sections=sections,
        abstract=abstract,
    )

    user_message = f"""Please extract entities and relationships from this paper:

TITLE: {title if title else "Not provided"}

ABSTRACT: {abstract if abstract else "Not provided"}

FULL TEXT (section-aware excerpt):
{extraction_context}

Extract entities and relationships. Respond ONLY with valid JSON."""

    return system_prompt, user_message


def _extract_entities_with_llm(
    paper_text: str,
    title: str = "",
    abstract: str = "",
    sections: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """LLM extraction path retained for compatibility and fallback."""
    try:
        client = get_llm_client()
    except ValueError as e:
        logger.error(f"Cannot initialize LLM client: {e}")
        return {
            "genes": [],
            "proteins": [],
            "diseases": [],
            "pathways": [],
            "relationships": [],
            "error": str(e)
        }
    
    system_prompt, user_message = build_extraction_prompt(
        paper_text,
        title,
        abstract,
        sections=sections,
    )
    
    try:
        logger.info("Calling %s for entity extraction...", client.get_provider_label())
        response = client.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_message,
            model=os.getenv("PAPER_EXTRACTION_MODEL", ""),
            max_tokens=2000,
            temperature=0.1,
            json_mode=True,
        )

        response_text = response.text.strip()
        logger.info("LLM response received")
        
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            extracted_data = json.loads(json_str)
        else:
            logger.warning("Could not find JSON in GPT response")
            return _parse_text_extraction(response_text)

        return _normalize_extracted_payload(extracted_data, title=title)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return {
            "genes": [],
            "proteins": [],
            "diseases": [],
            "pathways": [],
            "relationships": [],
            "error": f"JSON parsing error: {e}"
        }
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return {
            "genes": [],
            "proteins": [],
            "diseases": [],
            "pathways": [],
            "relationships": [],
            "error": str(e)
        }


def _build_fulltext_from_sections_or_text(
    paper_text: str,
    sections: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if sections:
        section_texts = []
        for section in sections:
            section_type = normalize_section_type(section.get("type", ""))
            if section_type == "metadata":
                continue
            value = str(section.get("text", "")).strip()
            if value:
                section_texts.append(value)
        merged = "\n\n".join(section_texts).strip()
        if merged:
            return merged
    return str(paper_text or "").strip()


def extract_entities_from_text(
    paper_text: str,
    title: str = "",
    abstract: str = "",
    sections: Optional[List[Dict[str, Any]]] = None,
    existing_entities: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Extract entities from paper text.

    Backends:
    - llm: original LLM-only extraction over a section-aware excerpt
    - scispacy: full NER_extracion pipeline (SciSpaCy stage + BioBERT refinement) + sentence relationships
    - hybrid (default): full NER_extracion pipeline first; LLM fallback if extraction returns empty
    """
    if not existing_entities:
        existing_entities = {
            "genes": [],
            "proteins": [],
            "diseases": [],
            "pathways": []
        }

    backend = _get_extraction_backend()
    full_text = _build_fulltext_from_sections_or_text(paper_text, sections)
    extracted_data: Dict[str, Any] = {
        "genes": [],
        "proteins": [],
        "diseases": [],
        "pathways": [],
        "relationships": [],
    }

    if backend in {"scispacy", "hybrid"}:
        ner_pipeline_entities = _extract_entities_with_full_ner_pipeline(full_text)
        extracted_data = {
            **ner_pipeline_entities,
            "relationships": _infer_sentence_relationships(full_text, ner_pipeline_entities),
            "extraction_backend": "scispacy_biobert_pipeline",
        }
        if backend == "scispacy":
            extracted_data = _boost_confidence_for_known_entities(extracted_data, existing_entities)
            extracted_data = _normalize_extracted_payload(extracted_data, title=title)
            logger.info("Pipeline-only extraction complete: %s entities", _count_entities(extracted_data))
            return extracted_data

        if _count_entities(extracted_data) == 0:
            logger.info("Hybrid extraction: NER pipeline returned no entities, falling back to LLM.")
            llm_extracted = _extract_entities_with_llm(
                paper_text=paper_text,
                title=title,
                abstract=abstract,
                sections=sections,
            )
            extracted_data = _merge_extraction_results(extracted_data, llm_extracted)
            extracted_data["extraction_backend"] = "hybrid"
        else:
            extracted_data["extraction_backend"] = "hybrid"

    elif backend == "llm":
        extracted_data = _extract_entities_with_llm(
            paper_text=paper_text,
            title=title,
            abstract=abstract,
            sections=sections,
        )
        extracted_data["extraction_backend"] = "llm"

    extracted_data = _boost_confidence_for_known_entities(extracted_data, existing_entities)
    extracted_data = _normalize_extracted_payload(extracted_data, title=title)
    logger.info("Extraction complete via %s: %s entities", backend, _count_entities(extracted_data))
    return extracted_data


def extract_relationships_from_text(
    paper_text: str,
    extracted_entities: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Extract specific relationships between entities using GPT-4.
    
    Args:
        paper_text: Paper text
        extracted_entities: Previously extracted entity dict
    
    Returns:
        list: Relationships with source, target, relation, confidence, evidence
    """
    # Extract from the main extraction already includes relationships
    # This is a supplementary function for more detailed relationship mining
    
    entities_str = ", ".join([
        e["name"] for entities in extracted_entities.values() 
        for e in entities if isinstance(e, dict) and "name" in e
    ])
    
    if not entities_str:
        return []
    
    try:
        client = get_llm_client()
    except ValueError as e:
        logger.error(f"Cannot initialize LLM client: {e}")
        return []
    
    prompt = f"""Given these extracted entities: {entities_str[:500]}

From this paper text, extract ONLY direct biological relationships:
{paper_text[:3000]}

Format as JSON array:
[
    {{
      "source_name": "GENE1",
      "source_type": "Gene",
      "target_name": "PROTEIN1",
      "target_type": "Protein",
      "edge_type": "ENCODES",
      "edge_weight": 0.9,
      "evidence": "..."
    }},
    ...
]

Return ONLY valid JSON, no other text."""

    try:
        response = client.generate_text(
            system_prompt="You are a biomedical relationship extraction assistant.",
            user_prompt=prompt,
            model=os.getenv("PAPER_EXTRACTION_MODEL", ""),
            max_tokens=1000,
            temperature=0.1,
            json_mode=True,
        )

        response_text = response.text.strip()
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            relationships = json.loads(response_text[json_start:json_end])
            return _normalize_extracted_payload({"relationships": relationships}).get("relationships", [])
        return []
    except Exception as e:
        logger.error(f"Error extracting relationships: {e}")
        return []


def _count_entities(extracted_data: Dict[str, Any]) -> int:
    """Count total entities in extracted data."""
    count = 0
    for key in ["genes", "proteins", "diseases", "pathways"]:
        if key in extracted_data and isinstance(extracted_data[key], list):
            count += len(extracted_data[key])
    return count


def _boost_confidence_for_known_entities(
    extracted_data: Dict[str, List[Dict[str, Any]]],
    existing_entities: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Increase confidence scores for entities already in database.
    
    Args:
        extracted_data: Data from GPT-4 extraction
        existing_entities: Dict of known entities
    
    Returns:
        updated extracted_data with adjusted confidence scores
    """
    entity_types = {
        "genes": "genes",
        "proteins": "proteins",
        "diseases": "diseases",
        "pathways": "pathways"
    }
    
    for extracted_type, db_list_key in entity_types.items():
        if extracted_type in extracted_data:
            known_names = set(name.lower() for name in existing_entities.get(db_list_key, []))
            
            for entity in extracted_data[extracted_type]:
                if isinstance(entity, dict):
                    entity_name_lower = entity.get("name", "").lower()
                    if entity_name_lower in known_names:
                        # Boost confidence for known entities (add 0.05)
                        old_conf = entity.get("confidence", 0.5)
                        entity["confidence"] = min(0.99, old_conf + 0.05)
                        entity["is_known_entity"] = True
    
    return extracted_data


def _parse_text_extraction(response_text: str) -> Dict[str, Any]:
    """
    Fallback parser if GPT response is not valid JSON.
    Attempts to extract entities from free-form text response.
    
    Args:
        response_text: GPT-4 response as text
    
    Returns:
        dict: Partially extracted entities
    """
    result = {
        "genes": [],
        "proteins": [],
        "diseases": [],
        "pathways": [],
        "relationships": [],
        "parse_method": "text_fallback"
    }
    
    lines = response_text.split('\n')
    current_type = None
    
    for line in lines:
        line_lower = line.lower()
        if 'gene' in line_lower and ':' in line:
            current_type = "genes"
        elif 'protein' in line_lower and ':' in line:
            current_type = "proteins"
        elif 'disease' in line_lower and ':' in line:
            current_type = "diseases"
        elif 'pathway' in line_lower and ':' in line:
            current_type = "pathways"
        elif current_type and '-' in line and len(line.strip()) > 5:
            # Try to extract entity from bullet point
            entity_name = line.split('-')[1].strip().split('(')[0].strip()
            if entity_name and len(entity_name) > 2:
                result[current_type].append({
                    "name": entity_name,
                    "context": "",
                    "confidence": 0.6
                })
    
    return result


def validate_extracted_entities(
    extracted_data: Dict[str, Any],
    min_confidence: float = 0.60
) -> Dict[str, Any]:
    """
    Filter and validate extracted entities by confidence threshold.
    
    Args:
        extracted_data: Raw extracted data
        min_confidence: Minimum confidence threshold (0.0-1.0)
    
    Returns:
        dict: Filtered and validated entities
    """
    validated = {
        "genes": [],
        "proteins": [],
        "diseases": [],
        "pathways": [],
        "relationships": []
    }
    
    for entity_type in ["genes", "proteins", "diseases", "pathways"]:
        for entity in extracted_data.get(entity_type, []):
            if isinstance(entity, dict):
                confidence = entity.get("confidence", 0)
                if confidence >= min_confidence:
                    # Ensure required fields
                    if "name" in entity and entity["name"].strip():
                        validated[entity_type].append(entity)
    
    # Validate relationships
    for rel in extracted_data.get("relationships", []):
        if isinstance(rel, dict):
            confidence = rel.get("edge_weight", rel.get("confidence", 0))
            if confidence >= min_confidence:
                if all(k in rel for k in ["source_name", "target_name", "edge_type"]):
                    validated["relationships"].append(rel)
    
    return validated


def format_extraction_for_review(
    extracted_data: Dict[str, Any],
    paper_id: str
) -> Dict[str, Any]:
    """
    Format extracted data for manual review UI.
    
    Args:
        extracted_data: Raw extracted data from GPT-4
        paper_id: ID of source paper
    
    Returns:
        dict: Formatted data ready for review interface
    """
    normalized = _normalize_extracted_payload(extracted_data, paper_id=paper_id)
    formatted = normalize_entities_for_paper(paper_id, normalized)
    formatted["relationships"] = normalize_relationships_for_paper(
        paper_id,
        normalized.get("relationships", []),
        formatted,
    )
    return formatted


def _normalize_extracted_payload(
    extracted_data: Optional[Dict[str, Any]],
    paper_id: str = "",
    title: str = "",
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "genes": [],
        "proteins": [],
        "diseases": [],
        "pathways": [],
        "relationships": [],
    }
    payload = extracted_data or {}

    for entity_type in ["genes", "proteins", "diseases", "pathways"]:
        for entity in payload.get(entity_type, []):
            if not isinstance(entity, dict):
                continue
            entity_record = {
                "id": str(entity.get("id", "")).strip(),
                "name": str(entity.get("name", "")).strip(),
                "confidence": float(entity.get("confidence", 0.5) or 0.5),
                "context": str(entity.get("context", "")).strip(),
                "entity_type": normalize_entity_type(entity.get("entity_type", entity_type)),
                "chromosome": str(entity.get("chromosome", "")).strip(),
                "approved": bool(entity.get("approved", False)),
                "mapped_to_existing": str(entity.get("mapped_to_existing", "")),
                "notes": str(entity.get("notes", "")),
            }
            if entity_record["name"]:
                if not entity_record["context"] and title:
                    entity_record["context"] = f"Extracted from paper: {title}"
                normalized[entity_type].append(entity_record)

    temp_entities = normalize_entities_for_paper(paper_id or "paper", normalized)
    lookup_by_name = {}
    for entity_type in ["genes", "proteins", "diseases", "pathways"]:
        for entity in temp_entities.get(entity_type, []):
            lookup_by_name[str(entity.get("name", "")).strip().lower()] = entity

    for index, rel in enumerate(payload.get("relationships", [])):
        if not isinstance(rel, dict):
            continue
        source_name = str(rel.get("source_name", rel.get("source", ""))).strip()
        target_name = str(rel.get("target_name", rel.get("target", ""))).strip()
        if not source_name or not target_name:
            continue

        source_match = lookup_by_name.get(source_name.lower(), {})
        target_match = lookup_by_name.get(target_name.lower(), {})
        source_type = rel.get("source_type") or source_match.get("entity_type", "Entity")
        target_type = rel.get("target_type") or target_match.get("entity_type", "Entity")
        original_relation = str(rel.get("original_relation", rel.get("relation", rel.get("edge_type", "")))).strip()
        edge_type = str(rel.get("edge_type", "")).strip() or canonicalize_edge_type(
            original_relation,
            source_type,
            target_type,
        )

        normalized["relationships"].append({
            "id": str(rel.get("id", "")).strip() or f"{paper_id}_rel_{index}",
            "source_name": source_name,
            "source_type": normalize_entity_type(source_type, title_case=True),
            "source_id": str(rel.get("source_id", source_match.get("id", ""))).strip(),
            "target_name": target_name,
            "target_type": normalize_entity_type(target_type, title_case=True),
            "target_id": str(rel.get("target_id", target_match.get("id", ""))).strip(),
            "edge_type": edge_type,
            "edge_weight": float(rel.get("edge_weight", rel.get("confidence", 0.5)) or 0.5),
            "evidence": str(rel.get("evidence", "")).strip(),
            "source_chromosome": str(rel.get("source_chromosome", source_match.get("chromosome", ""))).strip(),
            "original_relation": original_relation,
            "extraction_method": str(rel.get("extraction_method", "gpt4")),
            "approved": bool(rel.get("approved", False)),
            "notes": str(rel.get("notes", "")),
        })

    return normalized
