import argparse
import json
import logging
import os
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import spacy
from gliner import GLiNER
from lxml import etree, html

# ---------------- CONFIG ---------------- #
USER_AGENT = "Mozilla/5.0 (compatible; Biomedical-NER/2.0)"
CHUNK_SIZE_CHARS = 12000
CHUNK_OVERLAP_CHARS = 200
GLINER_CHUNK_SIZE_CHARS = 9000
GLINER_CHUNK_OVERLAP_CHARS = 80
GLINER_DISEASE_LABELS = [
    "Disease or disorder",
    "Genetic disease",
    "Disease",
]
GLINER_THRESHOLD = 0.5
SCISPACY_BATCH_SIZE = 16

SCISPACY_MODELS = {
    "en_ner_bc5cdr_md": "DISEASE",
    "en_ner_bionlp13cg_md": "GENE_OR_PROTEIN",
    "en_ner_jnlpba_md": "GENE_OR_PROTEIN",
    "en_ner_craft_md": "PATHWAY",
}

# Source priority for dedup replacement decisions.
SOURCE_PRIORITY = {
    "DISEASE": {
        "gliner_biomed_disease": 100,
        "en_ner_bc5cdr_md": 90,
    },
    "GENE_OR_PROTEIN": {
        "en_ner_bionlp13cg_md": 100,
        "en_ner_jnlpba_md": 90,
    },
    "PATHWAY": {
        "en_ner_craft_md": 100,
    },
}

logger = logging.getLogger("biomed_ner_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_models_lock = threading.Lock()
_model_locks: Dict[str, threading.Lock] = {}
MODELS: Dict[str, object] = {}


def _load_scispacy_model(model_name: str):
    # Keep only NER-relevant components to reduce latency.
    return spacy.load(
        model_name,
        exclude=[
            "tagger",
            "parser",
            "lemmatizer",
            "attribute_ruler",
            "morphologizer",
            "senter",
        ],
    )


# ---------------- LOAD ONCE ---------------- #
def load_models() -> Dict[str, object]:
    global MODELS

    if MODELS:
        return MODELS

    with _models_lock:
        if MODELS:
            return MODELS

        logger.info("Loading SciSpaCy + GLiNER models once at startup...")
        load_start = time.perf_counter()

        loaded: Dict[str, object] = {"scispacy": {}, "gliner": None}
        total_to_load = len(SCISPACY_MODELS) + 1
        load_workers = min(total_to_load, max(1, os.cpu_count() or 1))

        with ThreadPoolExecutor(max_workers=load_workers) as executor:
            future_map = {}
            future_map[executor.submit(GLiNER.from_pretrained, "Ihor/gliner-biomed-large-v1.0")] = (
                "gliner",
                "gliner_biomed_disease",
            )
            for model_name in SCISPACY_MODELS:
                future_map[executor.submit(_load_scispacy_model, model_name)] = ("scispacy", model_name)

            for future in as_completed(future_map):
                kind, model_name = future_map[future]
                model_obj = future.result()
                if kind == "gliner":
                    loaded["gliner"] = model_obj
                else:
                    loaded["scispacy"][model_name] = model_obj
                _model_locks[model_name] = threading.Lock()

        _model_locks["gliner_biomed_disease"] = threading.Lock()
        MODELS = loaded
        logger.info("Model load completed in %.2fs", time.perf_counter() - load_start)

    return MODELS


# ---------------- TEXT/PARSING UTIL ---------------- #
def normalize_pmcid(source: str) -> str:
    value = (source or "").strip()

    match = re.search(r"PMC(\d+)", value, flags=re.IGNORECASE)
    if match:
        return f"PMC{match.group(1)}"

    if value.isdigit():
        return f"PMC{value}"

    raise ValueError(f"Invalid PMCID: {source}")


def fetch_article_xml(pmcid: str) -> bytes:
    pmcid_num = pmcid.replace("PMC", "")
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    response = requests.get(
        url,
        params={"db": "pmc", "id": pmcid_num, "retmode": "xml"},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()

    content = response.content
    if b"<article" not in content and b"<pmc-articleset" not in content:
        raise ValueError("Invalid XML response from PMC")

    return content


def parse_xml(content: bytes):
    try:
        return etree.fromstring(content)
    except Exception:
        return html.fromstring(content)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_article_text(content: bytes) -> str:
    root = parse_xml(content)
    paragraphs = root.xpath("//*[local-name()='p']")

    text_parts = []
    for p in paragraphs:
        txt = normalize_whitespace(" ".join(p.itertext()))
        if txt:
            text_parts.append(txt)

    return "\n".join(text_parts)


def iter_text_chunks(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS):
    if not text:
        return

    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)

        # Keep cuts near whitespace for better token boundaries.
        if end < n:
            ws_idx = text.rfind(" ", start, end)
            if ws_idx > start:
                end = ws_idx

        if end <= start:
            end = min(start + chunk_size, n)

        yield start, text[start:end]

        if end >= n:
            break

        start = max(0, end - overlap)


# ---------------- MODEL RUNNERS ---------------- #
def run_single_scispacy_model(model_name: str, text: str) -> Dict:
    started = time.perf_counter()
    nlp = MODELS["scispacy"][model_name]
    entities: List[Dict] = []

    chunks = list(iter_text_chunks(text))
    if not chunks:
        return {"model": model_name, "entities": []}

    # Only one thread should call a given pipeline object at a time.
    with _model_locks[model_name]:
        chunk_texts = [chunk for _, chunk in chunks]
        for (offset, chunk), doc in zip(chunks, nlp.pipe(chunk_texts, batch_size=SCISPACY_BATCH_SIZE), strict=False):
            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": offset + ent.start_char,
                        "end": offset + ent.end_char,
                    }
                )

    return {
        "model": model_name,
        "entities": entities,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def run_scispacy_models(text: str) -> List[Dict]:
    load_models()

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=len(SCISPACY_MODELS)) as executor:
        futures = [executor.submit(run_single_scispacy_model, model_name, text) for model_name in SCISPACY_MODELS]
        for future in as_completed(futures):
            results.append(future.result())

    return results


def run_all_models_parallel(text: str) -> List[Dict]:
    load_models()
    jobs = [(run_single_scispacy_model, model_name, text) for model_name in SCISPACY_MODELS]
    jobs.append((run_gliner, text))

    # Reduce oversubscription: these models are heavy and can slow each other down.
    workers = min(len(jobs), max(2, (os.cpu_count() or 4) // 2))
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for job in jobs:
            fn = job[0]
            args = job[1:]
            futures.append(executor.submit(fn, *args))
        for future in as_completed(futures):
            results.append(future.result())
    return results


def run_gliner(text: str) -> Dict:
    started = time.perf_counter()
    load_models()

    model = MODELS["gliner"]
    entities: List[Dict] = []

    for offset, chunk in iter_text_chunks(
        text,
        chunk_size=GLINER_CHUNK_SIZE_CHARS,
        overlap=GLINER_CHUNK_OVERLAP_CHARS,
    ):
        with _model_locks["gliner_biomed_disease"]:
            predictions = model.predict_entities(chunk, GLINER_DISEASE_LABELS, threshold=GLINER_THRESHOLD)

        for pred in predictions:
            entity = {
                "text": pred.get("text", ""),
                "label": pred.get("label", ""),
                "start": offset + int(pred.get("start", 0)),
                "end": offset + int(pred.get("end", 0)),
            }
            if "score" in pred:
                entity["confidence"] = float(pred["score"])
            entities.append(entity)

    return {
        "model": "gliner_biomed_disease",
        "entities": entities,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


# ---------------- NORMALIZATION + MERGE ---------------- #
def normalize_entity_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


def span_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    if intersection <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return (intersection / union) if union > 0 else 0.0


def map_to_unified_label(model_name: str, raw_label: str) -> Optional[str]:
    label_upper = (raw_label or "").upper()

    if model_name == "gliner_biomed_disease":
        return "DISEASE"

    if model_name == "en_ner_bc5cdr_md":
        if "DISEASE" in label_upper:
            return "DISEASE"
        return None

    if model_name == "en_ner_bionlp13cg_md":
        if "GENE" in label_upper or "PROTEIN" in label_upper:
            return "GENE_OR_PROTEIN"
        return None

    if model_name == "en_ner_jnlpba_md":
        # JNLPBA provides DNA/RNA/PROTEIN/CELL labels; keep gene/protein-related labels.
        if any(k in label_upper for k in ("GENE", "DNA", "RNA", "PROTEIN")):
            return "GENE_OR_PROTEIN"
        return None

    if model_name == "en_ner_craft_md":
        # Keep GO terms as pathway proxy. Be tolerant to naming variants.
        if "GO" in label_upper or "GO_TERM" in label_upper:
            return "PATHWAY"
        return None

    return None


def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    deduped: List[Dict] = []

    def priority(entity: Dict) -> int:
        unified = entity["unified_label"]
        source = entity["source"]
        return SOURCE_PRIORITY.get(unified, {}).get(source, 0)

    for candidate in entities:
        candidate_norm = normalize_entity_text(candidate["text"])
        candidate["_norm_text"] = candidate_norm

        duplicate_idx = None
        for i, existing in enumerate(deduped):
            if existing["unified_label"] != candidate["unified_label"]:
                continue

            same_text = existing["_norm_text"] == candidate_norm
            high_overlap = span_iou(existing["start"], existing["end"], candidate["start"], candidate["end"]) > 0.7

            if same_text or high_overlap:
                duplicate_idx = i
                break

        if duplicate_idx is None:
            deduped.append(candidate)
            continue

        existing = deduped[duplicate_idx]
        existing_priority = priority(existing)
        candidate_priority = priority(candidate)

        if candidate_priority > existing_priority:
            deduped[duplicate_idx] = candidate
        elif candidate_priority == existing_priority:
            existing_conf = float(existing.get("confidence", -1.0))
            candidate_conf = float(candidate.get("confidence", -1.0))
            if candidate_conf > existing_conf:
                deduped[duplicate_idx] = candidate

    for entity in deduped:
        entity.pop("_norm_text", None)

    return deduped


def merge_entities(results: List[Dict]) -> Dict[str, List[Dict]]:
    unified: List[Dict] = []

    for result in results or []:
        model_name = result.get("model", "")
        for ent in result.get("entities", []):
            unified_label = map_to_unified_label(model_name, ent.get("label", ""))
            if not unified_label:
                continue

            merged = {
                "text": ent.get("text", ""),
                "start": int(ent.get("start", 0)),
                "end": int(ent.get("end", 0)),
                "source": model_name,
                "unified_label": unified_label,
            }
            if "confidence" in ent:
                merged["confidence"] = ent["confidence"]

            # Skip empty strings and invalid spans.
            if not merged["text"].strip() or merged["end"] <= merged["start"]:
                continue

            unified.append(merged)

    unified = deduplicate_entities(unified)

    diseases = []
    genes_proteins = []
    pathways = []

    for ent in unified:
        output_item = {
            "text": ent["text"],
            "start": ent["start"],
            "end": ent["end"],
            "source": ent["source"],
        }
        if "confidence" in ent:
            output_item["confidence"] = ent["confidence"]

        if ent["unified_label"] == "DISEASE":
            diseases.append(output_item)
        elif ent["unified_label"] == "GENE_OR_PROTEIN":
            genes_proteins.append(output_item)
        elif ent["unified_label"] == "PATHWAY":
            pathways.append(output_item)

    diseases.sort(key=lambda x: (x["start"], x["end"]))
    genes_proteins.sort(key=lambda x: (x["start"], x["end"]))
    pathways.sort(key=lambda x: (x["start"], x["end"]))

    return {
        "diseases": diseases,
        "genes_proteins": genes_proteins,
        "pathways": pathways,
    }


# ---------------- PIPELINE ENTRY ---------------- #
def _run_pipeline_with_timings(text: str) -> Tuple[Dict[str, List[Dict]], Dict[str, object]]:
    if not text or not text.strip():
        empty = {"diseases": [], "genes_proteins": [], "pathways": []}
        return empty, {"total_pipeline_seconds": 0.0, "parallel_inference_seconds": 0.0, "merge_seconds": 0.0}

    pipeline_start = time.perf_counter()
    inference_start = time.perf_counter()
    all_results = run_all_models_parallel(text)
    inference_seconds = time.perf_counter() - inference_start

    model_runtime_seconds = {}
    for result in all_results:
        model_name = result.get("model", "unknown_model")
        model_runtime_seconds[model_name] = result.get("runtime_seconds", None)

    merge_start = time.perf_counter()
    merged = merge_entities(all_results)
    merge_seconds = time.perf_counter() - merge_start

    timings = {
        "model_runtime_seconds": model_runtime_seconds,
        "parallel_inference_seconds": round(inference_seconds, 3),
        "merge_seconds": round(merge_seconds, 3),
        "total_pipeline_seconds": round(time.perf_counter() - pipeline_start, 3),
    }
    return merged, timings


def main_pipeline(text: str) -> Dict[str, List[Dict]]:
    load_models()

    merged, _ = _run_pipeline_with_timings(text)
    return merged


# ---------------- CLI ---------------- #
def build_text_from_args(args: argparse.Namespace) -> str:
    if args.text:
        return args.text

    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")

    if args.source:
        pmcid = normalize_pmcid(args.source)
        xml_content = fetch_article_xml(pmcid)
        return extract_article_text(xml_content)

    raise ValueError("Provide one of: --text, --text-file, or --source <PMCID>")


def main():
    parser = argparse.ArgumentParser(description="Unified biomedical NER (GLiNER + SciSpaCy)")
    parser.add_argument("--text", help="Raw input text")
    parser.add_argument("--text-file", help="Path to UTF-8 text file")
    parser.add_argument("--source", help="PMCID (e.g., PMC1234567) or digits")
    parser.add_argument("--output", default="output_unified_ner.json", help="Output JSON path")
    args = parser.parse_args()

    start = time.perf_counter()

    input_start = time.perf_counter()
    text = build_text_from_args(args)
    input_seconds = time.perf_counter() - input_start

    logger.info("Running unified biomedical NER pipeline...")
    result, timings = _run_pipeline_with_timings(text)
    logger.info(
        "Timing: input=%.2fs, inference=%.2fs, merge=%.2fs, pipeline_total=%.2fs",
        input_seconds,
        timings.get("parallel_inference_seconds", 0.0),
        timings.get("merge_seconds", 0.0),
        timings.get("total_pipeline_seconds", 0.0),
    )

    payload = {
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "timings": {
            "input_build_seconds": round(input_seconds, 3),
            **timings,
        },
        "counts": {
            "diseases": len(result["diseases"]),
            "genes_proteins": len(result["genes_proteins"]),
            "pathways": len(result["pathways"]),
        },
        **result,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved output to %s", output_path)


if __name__ == "__main__":
    main()
