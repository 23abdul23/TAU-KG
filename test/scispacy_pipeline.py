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
from lxml import etree, html

# ---------------- CONFIG ---------------- #
USER_AGENT = "Mozilla/5.0 (compatible; SciSpaCy-NER/1.0)"
CHUNK_SIZE_CHARS = 12000
CHUNK_OVERLAP_CHARS = 200
SCISPACY_BATCH_SIZE = 16

SCISPACY_MODELS = {
    "en_ner_bc5cdr_md": "DISEASE",
    "en_ner_bionlp13cg_md": "GENE_OR_PROTEIN",
    "en_ner_jnlpba_md": "GENE_OR_PROTEIN",
    "en_ner_craft_md": "PATHWAY",
}

SOURCE_PRIORITY = {
    "DISEASE": {
        "en_ner_bc5cdr_md": 100,
    },
    "GENE_OR_PROTEIN": {
        "en_ner_bionlp13cg_md": 100,
        "en_ner_jnlpba_md": 90,
    },
    "PATHWAY": {
        "en_ner_craft_md": 100,
    },
}

logger = logging.getLogger("scispacy_ner_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_models_lock = threading.Lock()
_model_locks: Dict[str, threading.Lock] = {}
MODELS: Dict[str, object] = {}


# ---------------- MODEL LOAD ---------------- #
def _load_scispacy_model(model_name: str):
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


def load_models() -> Dict[str, object]:
    global MODELS

    if MODELS:
        return MODELS

    with _models_lock:
        if MODELS:
            return MODELS

        logger.info("Loading SciSpaCy models once...")
        load_start = time.perf_counter()

        loaded: Dict[str, object] = {"scispacy": {}}
        workers = min(len(SCISPACY_MODELS), max(1, os.cpu_count() or 1))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_load_scispacy_model, model_name): model_name
                for model_name in SCISPACY_MODELS
            }
            for future in as_completed(future_map):
                model_name = future_map[future]
                loaded["scispacy"][model_name] = future.result()
                _model_locks[model_name] = threading.Lock()

        MODELS = loaded
        logger.info("Model load completed in %.2fs", time.perf_counter() - load_start)

    return MODELS


# ---------------- INPUT UTIL ---------------- #
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


def iter_text_chunks(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS):
    if not text:
        return

    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)

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


# ---------------- NER ---------------- #
def run_single_scispacy_model(model_name: str, text: str) -> Dict:
    started = time.perf_counter()
    nlp = MODELS["scispacy"][model_name]
    entities: List[Dict] = []

    chunks = list(iter_text_chunks(text))
    if not chunks:
        return {"model": model_name, "entities": [], "runtime_seconds": 0.0}

    with _model_locks[model_name]:
        chunk_texts = [chunk for _, chunk in chunks]
        for (offset, _chunk), doc in zip(chunks, nlp.pipe(chunk_texts, batch_size=SCISPACY_BATCH_SIZE), strict=False):
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

    jobs = [(run_single_scispacy_model, model_name, text) for model_name in SCISPACY_MODELS]
    workers = min(len(jobs), max(2, (os.cpu_count() or 4) // 2))

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, *args) for fn, *args in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return results


# ---------------- NORMALIZE + DEDUP ---------------- #
def normalize_entity_text(text: str) -> str:
    value = (text or "").lower().strip()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value)


def span_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    if intersection <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return (intersection / union) if union > 0 else 0.0


def map_to_unified_label(model_name: str, raw_label: str) -> Optional[str]:
    label_upper = (raw_label or "").upper()

    if model_name == "en_ner_bc5cdr_md":
        return "DISEASE" if "DISEASE" in label_upper else None

    if model_name == "en_ner_bionlp13cg_md":
        return "GENE_OR_PROTEIN" if ("GENE" in label_upper or "PROTEIN" in label_upper) else None

    if model_name == "en_ner_jnlpba_md":
        if any(k in label_upper for k in ("GENE", "DNA", "RNA", "PROTEIN")):
            return "GENE_OR_PROTEIN"
        return None

    if model_name == "en_ner_craft_md":
        return "PATHWAY" if ("GO" in label_upper or "GO_TERM" in label_upper) else None

    return None


def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    deduped: List[Dict] = []

    def priority(entity: Dict) -> int:
        return SOURCE_PRIORITY.get(entity["unified_label"], {}).get(entity["source"], 0)

    for candidate in entities:
        candidate["_norm_text"] = normalize_entity_text(candidate["text"])

        duplicate_idx = None
        for i, existing in enumerate(deduped):
            if existing["unified_label"] != candidate["unified_label"]:
                continue

            same_text = existing["_norm_text"] == candidate["_norm_text"]
            high_overlap = span_iou(existing["start"], existing["end"], candidate["start"], candidate["end"]) > 0.7
            if same_text or high_overlap:
                duplicate_idx = i
                break

        if duplicate_idx is None:
            deduped.append(candidate)
            continue

        existing = deduped[duplicate_idx]
        if priority(candidate) > priority(existing):
            deduped[duplicate_idx] = candidate

    for entity in deduped:
        entity.pop("_norm_text", None)

    return deduped


def build_evidence_text(full_text: str, start: int, end: int, window: int = 80) -> str:
    if not full_text:
        return ""
    left = max(0, start - window)
    right = min(len(full_text), end + window)
    snippet = full_text[left:right].replace("\n", " ")
    return normalize_whitespace(snippet)


def merge_entities(results: List[Dict], full_text: str) -> Dict[str, List[Dict]]:
    unified: List[Dict] = []

    for result in results:
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
                "default_label": ent.get("label", ""),
            }
            if not merged["text"].strip() or merged["end"] <= merged["start"]:
                continue
            unified.append(merged)

    deduped = deduplicate_entities(unified)

    diseases, genes_proteins, pathways = [], [], []
    for ent in deduped:
        out = {
            "text": ent["text"],
            "start": ent["start"],
            "end": ent["end"],
            "source": ent["source"],
            "evidence_text": build_evidence_text(full_text, ent["start"], ent["end"]),
            "default_label": ent["default_label"],
            "deafult_label": ent["default_label"],
            "mapped_label": ent["unified_label"],
        }
        if ent["unified_label"] == "DISEASE":
            diseases.append(out)
        elif ent["unified_label"] == "GENE_OR_PROTEIN":
            genes_proteins.append(out)
        elif ent["unified_label"] == "PATHWAY":
            pathways.append(out)

    diseases.sort(key=lambda x: (x["start"], x["end"]))
    genes_proteins.sort(key=lambda x: (x["start"], x["end"]))
    pathways.sort(key=lambda x: (x["start"], x["end"]))

    return {
        "diseases": diseases,
        "genes_proteins": genes_proteins,
        "pathways": pathways,
    }


# ---------------- PIPELINE ---------------- #
def main_pipeline(text: str) -> Tuple[Dict[str, List[Dict]], Dict[str, object]]:
    load_models()

    if not text or not text.strip():
        empty = {"diseases": [], "genes_proteins": [], "pathways": []}
        return empty, {"parallel_inference_seconds": 0.0, "merge_seconds": 0.0, "model_runtime_seconds": {}}

    inference_start = time.perf_counter()
    model_results = run_scispacy_models(text)
    inference_seconds = time.perf_counter() - inference_start

    model_runtime_seconds = {r.get("model", "unknown"): r.get("runtime_seconds", 0.0) for r in model_results}

    merge_start = time.perf_counter()
    merged = merge_entities(model_results, text)
    merge_seconds = time.perf_counter() - merge_start

    timings = {
        "parallel_inference_seconds": round(inference_seconds, 3),
        "merge_seconds": round(merge_seconds, 3),
        "model_runtime_seconds": model_runtime_seconds,
    }
    return merged, timings


def main():
    parser = argparse.ArgumentParser(description="SciSpaCy-only biomedical NER pipeline")
    parser.add_argument("--text", help="Raw input text")
    parser.add_argument("--text-file", help="Path to UTF-8 text file")
    parser.add_argument("--source", help="PMCID (e.g., PMC1234567) or digits")
    parser.add_argument("--output", default="output_scispacy_ner.json", help="Output JSON path")
    args = parser.parse_args()

    total_start = time.perf_counter()
    input_start = time.perf_counter()
    text = build_text_from_args(args)
    input_seconds = time.perf_counter() - input_start

    logger.info("Running SciSpaCy-only NER pipeline...")
    result, timings = main_pipeline(text)

    payload = {
        "runtime_seconds": round(time.perf_counter() - total_start, 3),
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
