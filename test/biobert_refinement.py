import argparse
from collections import Counter
import json
import logging
import re
import string
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import pipeline

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


DEFAULT_MODEL_NAME = "d4data/biomedical-ner-all"
DEFAULT_CONTEXT_WINDOW = 80
DEFAULT_BATCH_SIZE = 32
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_SPAN_THRESHOLD = 0.55

CATEGORY_TO_LABEL = {
    "diseases": "DISEASE",
    "genes_proteins": "GENE_OR_PROTEIN",
    "pathways": "PATHWAY",
}
LABEL_TO_CATEGORY = {value: key for key, value in CATEGORY_TO_LABEL.items()}

NOISE_TERMS = {
    "antibody",
    "antibodies",
    "chromosomal",
    "markers",
}
GENERIC_TERMS = {
    "gene",
    "genes",
    "protein",
    "proteins",
    "disease",
    "diseases",
    "pathway",
    "pathways",
    "marker",
    "markers",
}

logger = logging.getLogger("biobert_refinement")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_model_lock = threading.Lock()
_NER_PIPELINE = None
_CONTEXT_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_entity_text(text: str) -> str:
    value = (text or "").lower().strip()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return normalize_whitespace(value)


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def label_from_category(category: str) -> str:
    return CATEGORY_TO_LABEL.get(category, "DISEASE")


def map_biobert_label(raw_label: str) -> Optional[str]:
    label = (raw_label or "").lower().replace("-", "_").replace(" ", "_")

    gene_tokens = ("gene", "protein", "dna", "rna", "transcript", "gene_or_gene_product")
    pathway_tokens = ("pathway", "signaling", "signal_transduction", "cascade", "biological_process")
    disease_tokens = (
        "disease",
        "disorder",
        "syndrome",
        "pathology",
        "injury",
        "cancer",
        "tumor",
        "neoplasm",
        "sign_symptom",
    )

    if any(token in label for token in gene_tokens):
        return "GENE_OR_PROTEIN"
    if any(token in label for token in pathway_tokens):
        return "PATHWAY"
    if any(token in label for token in disease_tokens):
        return "DISEASE"
    return None


def load_model(model_name: str = DEFAULT_MODEL_NAME, device: Optional[int] = None):
    global _NER_PIPELINE

    if _NER_PIPELINE is not None:
        return _NER_PIPELINE

    with _model_lock:
        if _NER_PIPELINE is not None:
            return _NER_PIPELINE

        if device is None:
            if torch is not None and torch.cuda.is_available():
                device = 0
            else:
                device = -1

        logger.info("Loading BioBERT refinement model: %s (device=%s)", model_name, device)
        _NER_PIPELINE = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=device,
        )
    return _NER_PIPELINE


def extract_context(full_text: str, start: int, end: int, window: int = DEFAULT_CONTEXT_WINDOW) -> Tuple[str, int, int]:
    if not full_text:
        return "", 0, 0

    left = max(0, int(start) - window)
    right = min(len(full_text), int(end) + window)
    return full_text[left:right], left, right


def flatten_input_entities(
    full_text: str,
    input_json: Dict[str, Any],
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []

    for category in ("diseases", "genes_proteins", "pathways"):
        for idx, raw in enumerate(input_json.get(category, []) or []):
            try:
                start = int(raw.get("start", 0))
                end = int(raw.get("end", 0))
            except Exception:
                continue

            if end <= start:
                continue

            text = str(raw.get("text", "")) or (full_text[start:end] if full_text else "")
            if not text.strip():
                continue

            can_use_full_text = bool(full_text and full_text.strip() and end <= len(full_text))
            if can_use_full_text:
                context_text, context_start, _ = extract_context(full_text, start, end, context_window)
                context_mode = "full_text"
            else:
                # Fallback for payloads like output_scispacy_ner.json where only local evidence is available.
                context_text = str(raw.get("evidence_text", "")) or text
                context_start = max(0, start - context_window)
                context_mode = "evidence_text"

            original_label = str(raw.get("mapped_label", "")).strip().upper() or label_from_category(category)
            if original_label not in LABEL_TO_CATEGORY:
                original_label = label_from_category(category)
            default_label = str(raw.get("default_label", raw.get("deafult_label", ""))).strip().upper() or original_label

            flat.append(
                {
                    "entity_id": f"{category}:{idx}",
                    "text": text,
                    "start": start,
                    "end": end,
                    "source": raw.get("source", "original_model"),
                    "original_label": original_label,
                    "original_category": category,
                    "default_label": default_label,
                    "input_score": float(raw.get("confidence", 0.0) or 0.0),
                    "evidence_text": normalize_whitespace(str(raw.get("evidence_text", ""))),
                    "context_text": context_text,
                    "context_start": context_start,
                    "context_mode": context_mode,
                }
            )
    return flat


def _normalize_pipeline_batch_output(raw_output: Any, expected_batch_size: int) -> List[List[Dict[str, Any]]]:
    if expected_batch_size == 0:
        return []

    if expected_batch_size == 1 and isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
        return [raw_output]

    if isinstance(raw_output, list):
        normalized: List[List[Dict[str, Any]]] = []
        for item in raw_output:
            if isinstance(item, list):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append([item])
            else:
                normalized.append([])
        return normalized

    return [[] for _ in range(expected_batch_size)]


def run_biobert_batch(
    contexts: List[str],
    nlp,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[List[Dict[str, Any]]]:
    if not contexts:
        return []

    results: List[Optional[List[Dict[str, Any]]]] = [None] * len(contexts)
    uncached_indices: List[int] = []
    uncached_contexts: List[str] = []

    for idx, ctx in enumerate(contexts):
        cached = _CONTEXT_CACHE.get(ctx)
        if cached is not None:
            results[idx] = cached
        else:
            uncached_indices.append(idx)
            uncached_contexts.append(ctx)

    if uncached_contexts:
        raw_predictions = nlp(uncached_contexts, batch_size=batch_size)
        normalized_predictions = _normalize_pipeline_batch_output(raw_predictions, len(uncached_contexts))

        for idx, pred in zip(uncached_indices, normalized_predictions, strict=False):
            _CONTEXT_CACHE[contexts[idx]] = pred
            results[idx] = pred

    return [item if item is not None else [] for item in results]


def _prediction_global_span(pred: Dict[str, Any], context_start: int, text_len: Optional[int]) -> Tuple[int, int]:
    p_start = context_start + int(pred.get("start", 0))
    p_end = context_start + int(pred.get("end", 0))

    if text_len is not None:
        p_start = max(0, min(p_start, text_len))
        p_end = max(0, min(p_end, text_len))
    return p_start, p_end


def _find_best_prediction_for_entity(
    entity: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    full_text: str,
) -> Optional[Dict[str, Any]]:
    ent_start, ent_end = int(entity["start"]), int(entity["end"])
    ent_norm = normalize_entity_text(entity["text"])
    best: Optional[Tuple[Tuple[float, int, int], Dict[str, Any]]] = None
    text_len: Optional[int] = len(full_text) if (full_text and full_text.strip()) else None
    context_text = str(entity.get("context_text", ""))

    for pred in predictions or []:
        p_start, p_end = _prediction_global_span(pred, int(entity["context_start"]), text_len)
        if p_end <= p_start:
            continue

        rel_start = max(0, int(pred.get("start", 0)))
        rel_end = max(rel_start, int(pred.get("end", 0)))
        context_pred_text = context_text[rel_start:rel_end] if context_text else ""
        global_pred_text = full_text[p_start:p_end] if text_len is not None else ""
        p_text = global_pred_text or context_pred_text
        p_norm = normalize_entity_text(p_text or str(pred.get("word", "")))
        score = float(pred.get("score", 0.0) or 0.0)
        overlap = overlap_len(ent_start, ent_end, p_start, p_end)

        substring_hit = bool(ent_norm and p_norm and (ent_norm in p_norm or p_norm in ent_norm))
        if overlap <= 0 and not substring_hit:
            continue

        mapped = map_biobert_label(str(pred.get("entity_group", pred.get("entity", ""))))
        candidate = {
            "start": p_start,
            "end": p_end,
            "text": p_text if p_text else str(pred.get("word", "")),
            "score": score,
            "raw_label": str(pred.get("entity_group", pred.get("entity", ""))),
            "mapped_label": mapped,
            "overlap": overlap,
        }

        rank = (score, overlap, p_end - p_start)
        if best is None or rank > best[0]:
            best = (rank, candidate)

    return best[1] if best else None


def reclassify_entities(
    full_text: str,
    entities: List[Dict[str, Any]],
    predictions_by_context: List[List[Dict[str, Any]]],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    span_threshold: float = DEFAULT_SPAN_THRESHOLD,
) -> List[Dict[str, Any]]:
    refined: List[Dict[str, Any]] = []
    has_full_text = bool(full_text and full_text.strip())

    for entity, predictions in zip(entities, predictions_by_context, strict=False):
        best_pred = _find_best_prediction_for_entity(entity, predictions, full_text)

        text = entity["text"]
        start = int(entity["start"])
        end = int(entity["end"])
        refined_label = entity["original_label"]
        refined_score = float(entity.get("input_score", 0.0))

        if best_pred:
            biobert_label = best_pred["mapped_label"]
            biobert_score = float(best_pred["score"])

            if biobert_label and biobert_score >= confidence_threshold:
                refined_label = biobert_label
                refined_score = biobert_score
            else:
                refined_score = max(refined_score, biobert_score)

            # Span correction for larger, overlapping BioBERT spans.
            if (
                biobert_score >= span_threshold
                and best_pred["end"] > best_pred["start"]
                and spans_overlap(start, end, best_pred["start"], best_pred["end"])
                and (best_pred["end"] - best_pred["start"]) > (end - start)
                and (biobert_label is None or biobert_label == refined_label)
            ):
                start = int(best_pred["start"])
                end = int(best_pred["end"])
                if has_full_text and full_text[start:end].strip():
                    text = full_text[start:end]
                else:
                    text = best_pred["text"]

        refined.append(
            {
                "entity_id": entity.get("entity_id", ""),
                "text": text,
                "start": start,
                "end": end,
                "refined_label": refined_label,
                "refined_score": round(float(refined_score), 4),
                "source": entity.get("source", "original_model"),
                "evidence_text": normalize_whitespace(entity.get("evidence_text", "")),
                "original_category": entity.get("original_category"),
                "original_label": entity.get("original_label"),
                "original_start": int(entity.get("start", start)),
                "original_end": int(entity.get("end", end)),
                "original_text": entity.get("text", text),
                "default_label": entity.get("default_label", entity.get("original_label")),
            }
        )

    return refined


def filter_noise(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cleaned: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for entity in entities:
        text = (entity.get("text", "") or "").strip()
        norm = normalize_entity_text(text)

        reason = None
        if len(text) < 3:
            reason = "too_short"
        elif not norm:
            reason = "empty_normalized_text"
        elif norm in NOISE_TERMS:
            reason = "blocked_noise_term"
        elif norm in GENERIC_TERMS:
            reason = "generic_term"
        elif not any(ch.isalpha() for ch in text):
            reason = "no_alpha_characters"

        if reason:
            removed.append({"entity": entity, "reason": f"noise_filter:{reason}"})
            continue

        cleaned.append(entity)

    return cleaned, removed


def deduplicate_entities(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scored = sorted(
        entities,
        key=lambda e: (float(e.get("refined_score", 0.0)), (int(e["end"]) - int(e["start"]))),
        reverse=True,
    )
    deduped: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for candidate in scored:
        candidate_norm = normalize_entity_text(candidate.get("text", ""))
        duplicate_idx: Optional[int] = None

        for idx, existing in enumerate(deduped):
            same_text = normalize_entity_text(existing.get("text", "")) == candidate_norm and bool(candidate_norm)
            overlap = spans_overlap(
                int(existing["start"]),
                int(existing["end"]),
                int(candidate["start"]),
                int(candidate["end"]),
            )
            if same_text or overlap:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            deduped.append(candidate)
            continue

        existing = deduped[duplicate_idx]
        existing_score = float(existing.get("refined_score", 0.0))
        candidate_score = float(candidate.get("refined_score", 0.0))
        existing_len = int(existing["end"]) - int(existing["start"])
        candidate_len = int(candidate["end"]) - int(candidate["start"])

        candidate_wins = candidate_score > existing_score or (candidate_score == existing_score and candidate_len > existing_len)
        if candidate_wins:
            deduped[duplicate_idx] = candidate
            removed.append(
                {
                    "entity": existing,
                    "reason": "deduplicated:lower_score_or_shorter_span",
                    "kept_entity_id": candidate.get("entity_id", ""),
                }
            )
        else:
            removed.append(
                {
                    "entity": candidate,
                    "reason": "deduplicated:lower_score_or_shorter_span",
                    "kept_entity_id": existing.get("entity_id", ""),
                }
            )

    deduped.sort(key=lambda e: (int(e["start"]), int(e["end"])))
    return deduped, removed


def _build_evidence_text(full_text: str, start: int, end: int, window: int = DEFAULT_CONTEXT_WINDOW) -> str:
    if full_text and full_text.strip() and 0 <= start < end <= len(full_text):
        snippet, _left, _right = extract_context(full_text, start, end, window)
        return normalize_whitespace(snippet.replace("\n", " "))
    return ""


def restructure_output(
    entities: List[Dict[str, Any]],
    full_text: str,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {
        "diseases": [],
        "genes_proteins": [],
        "pathways": [],
    }

    for entity in entities:
        category = LABEL_TO_CATEGORY.get(entity.get("refined_label", ""))
        if not category:
            continue

        start = int(entity["start"])
        end = int(entity["end"])
        evidence_text = _build_evidence_text(full_text, start, end, window=context_window) or normalize_whitespace(
            str(entity.get("evidence_text", ""))
        )
        default_label = str(entity.get("default_label", entity.get("original_label", entity.get("refined_label", "")))).upper()

        output[category].append(
            {
                "text": entity["text"],
                "start": start,
                "end": end,
                "source": entity.get("source", "original_model"),
                "evidence_text": evidence_text,
                "default_label": default_label,
                "deafult_label": default_label,
                "mapped_label": entity.get("refined_label", ""),
            }
        )

    for key in output:
        output[key].sort(key=lambda e: (e["start"], e["end"]))
    return output


def _entity_snapshot(entity: Dict[str, Any], category: str, label: str) -> Dict[str, Any]:
    return {
        "category": category,
        "text": entity.get("text", ""),
        "start": int(entity.get("start", 0)),
        "end": int(entity.get("end", 0)),
        "mapped_label": label,
        "source": entity.get("source", "original_model"),
    }


def build_report(
    input_json: Dict[str, Any],
    flat_entities: List[Dict[str, Any]],
    final_entities: List[Dict[str, Any]],
    removed_events: List[Dict[str, Any]],
    output_categories: Dict[str, List[Dict[str, Any]]],
    timings: Dict[str, float],
    biobert_available: bool,
) -> Dict[str, Any]:
    original_by_id = {item.get("entity_id", ""): item for item in flat_entities}
    changes: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for entity in final_entities:
        entity_id = entity.get("entity_id", "")
        orig = original_by_id.get(entity_id)
        if not orig:
            continue

        from_category = orig.get("original_category", "")
        to_category = LABEL_TO_CATEGORY.get(entity.get("refined_label", ""), from_category)
        from_label = orig.get("original_label", "")
        to_label = entity.get("refined_label", from_label)

        change_items: List[Dict[str, Any]] = []
        if from_label != to_label:
            change_items.append({"field": "mapped_label", "from": from_label, "to": to_label})
        if from_category != to_category:
            change_items.append({"field": "category", "from": from_category, "to": to_category})
        if int(orig.get("start", 0)) != int(entity.get("start", 0)) or int(orig.get("end", 0)) != int(entity.get("end", 0)):
            change_items.append(
                {
                    "field": "span",
                    "from": [int(orig.get("start", 0)), int(orig.get("end", 0))],
                    "to": [int(entity.get("start", 0)), int(entity.get("end", 0))],
                }
            )
        if str(orig.get("text", "")) != str(entity.get("text", "")):
            change_items.append({"field": "text", "from": str(orig.get("text", "")), "to": str(entity.get("text", ""))})

        if change_items:
            changes.append(
                {
                    "entity_id": entity_id,
                    "from": _entity_snapshot(orig, from_category, from_label),
                    "to": _entity_snapshot(entity, to_category, to_label),
                    "changes": change_items,
                    "refined_score": float(entity.get("refined_score", 0.0)),
                }
            )

    for event in removed_events:
        entity = event.get("entity", {}) or {}
        entity_id = entity.get("entity_id", "")
        orig = original_by_id.get(entity_id, entity)
        category = orig.get("original_category", LABEL_TO_CATEGORY.get(orig.get("refined_label", ""), ""))
        label = orig.get("original_label", orig.get("refined_label", ""))
        removed_entry = {
            "entity_id": entity_id,
            "from": _entity_snapshot(orig, category, label),
            "reason": event.get("reason", ""),
        }
        if event.get("kept_entity_id"):
            removed_entry["kept_entity_id"] = event["kept_entity_id"]
        removed.append(removed_entry)

    input_counts = {
        "diseases": len(input_json.get("diseases", []) or []),
        "genes_proteins": len(input_json.get("genes_proteins", []) or []),
        "pathways": len(input_json.get("pathways", []) or []),
    }
    output_counts = {
        "diseases": len(output_categories.get("diseases", [])),
        "genes_proteins": len(output_categories.get("genes_proteins", [])),
        "pathways": len(output_categories.get("pathways", [])),
    }

    removed_reason_counts = dict(Counter(item.get("reason", "") for item in removed))
    total_input = sum(input_counts.values())
    total_output = sum(output_counts.values())

    label_change_count = sum(1 for c in changes if any(ch.get("field") == "mapped_label" for ch in c.get("changes", [])))
    category_change_count = sum(1 for c in changes if any(ch.get("field") == "category" for ch in c.get("changes", [])))
    span_or_text_change_count = sum(
        1
        for c in changes
        if any(ch.get("field") in {"span", "text"} for ch in c.get("changes", []))
    )

    return {
        "summary": {
            "biobert_available": biobert_available,
            "input_counts": input_counts,
            "output_counts": output_counts,
            "total_input_entities": total_input,
            "total_output_entities": total_output,
            "total_removed_entities": len(removed),
            "changed_entities": len(changes),
            "unchanged_entities": max(0, total_output - len(changes)),
            "label_changed_entities": label_change_count,
            "category_changed_entities": category_change_count,
            "span_or_text_changed_entities": span_or_text_change_count,
            "removed_reason_counts": removed_reason_counts,
        },
        "timings": timings,
        "changes": changes,
        "removed": removed,
    }


def main_pipeline(
    text: str,
    input_json: Dict[str, Any],
    model_name: str = DEFAULT_MODEL_NAME,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    batch_size: int = DEFAULT_BATCH_SIZE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    span_threshold: float = DEFAULT_SPAN_THRESHOLD,
    device: Optional[int] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any], Dict[str, float]]:
    stage_start = time.perf_counter()
    flat_entities = flatten_input_entities(text or "", input_json, context_window=context_window)
    flatten_seconds = time.perf_counter() - stage_start

    if not flat_entities:
        empty_output = {"diseases": [], "genes_proteins": [], "pathways": []}
        timings = {
            "flatten_seconds": round(flatten_seconds, 3),
            "biobert_inference_seconds": 0.0,
            "reclassify_seconds": 0.0,
            "filter_seconds": 0.0,
            "dedup_seconds": 0.0,
            "restructure_seconds": 0.0,
            "total_pipeline_seconds": round(flatten_seconds, 3),
        }
        report = build_report(input_json, [], [], [], empty_output, timings, biobert_available=False)
        return empty_output, report, timings

    contexts = [entity.get("context_text", "") for entity in flat_entities]
    predictions: List[List[Dict[str, Any]]] = [[] for _ in flat_entities]
    biobert_available = True

    infer_start = time.perf_counter()
    try:
        nlp = load_model(model_name=model_name, device=device)
        predictions = run_biobert_batch(contexts, nlp=nlp, batch_size=batch_size)
    except Exception as exc:
        biobert_available = False
        logger.warning("BioBERT unavailable, using original labels with rule-based cleanup only: %s", exc)
    biobert_inference_seconds = time.perf_counter() - infer_start

    reclassify_start = time.perf_counter()
    refined = reclassify_entities(
        full_text=text or "",
        entities=flat_entities,
        predictions_by_context=predictions,
        confidence_threshold=confidence_threshold,
        span_threshold=span_threshold,
    )
    reclassify_seconds = time.perf_counter() - reclassify_start

    filter_start = time.perf_counter()
    filtered, removed_noise = filter_noise(refined)
    filter_seconds = time.perf_counter() - filter_start

    dedup_start = time.perf_counter()
    deduped, removed_dedup = deduplicate_entities(filtered)
    dedup_seconds = time.perf_counter() - dedup_start

    restructure_start = time.perf_counter()
    output_categories = restructure_output(deduped, full_text=text or "", context_window=context_window)
    restructure_seconds = time.perf_counter() - restructure_start

    timings = {
        "flatten_seconds": round(flatten_seconds, 3),
        "biobert_inference_seconds": round(biobert_inference_seconds, 3),
        "reclassify_seconds": round(reclassify_seconds, 3),
        "filter_seconds": round(filter_seconds, 3),
        "dedup_seconds": round(dedup_seconds, 3),
        "restructure_seconds": round(restructure_seconds, 3),
        "total_pipeline_seconds": round(
            flatten_seconds
            + biobert_inference_seconds
            + reclassify_seconds
            + filter_seconds
            + dedup_seconds
            + restructure_seconds,
            3,
        ),
    }
    report = build_report(
        input_json=input_json,
        flat_entities=flat_entities,
        final_entities=deduped,
        removed_events=[*removed_noise, *removed_dedup],
        output_categories=output_categories,
        timings=timings,
        biobert_available=biobert_available,
    )
    return output_categories, report, timings


def _build_text_from_args(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="BioBERT-based biomedical entity refinement")
    parser.add_argument("--text", help="Raw full text input (optional)")
    parser.add_argument("--text-file", help="Path to UTF-8 full text input (optional)")
    parser.add_argument(
        "--input-json",
        default="test/output_scispacy_ner.json",
        help="Path to extracted entities JSON",
    )
    parser.add_argument("--output", default="output_biobert_refined.json", help="Output JSON path")
    parser.add_argument("--report", default="report.json", help="Output report JSON path")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="HuggingFace NER model")
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW, help="Context chars on each side")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for label override",
    )
    parser.add_argument(
        "--span-threshold",
        type=float,
        default=DEFAULT_SPAN_THRESHOLD,
        help="Confidence threshold for span expansion",
    )
    parser.add_argument("--device", type=int, default=None, help="Transformers pipeline device id (-1 for CPU)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    args = parser.parse_args()

    logger.setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))

    start = time.perf_counter()
    text = _build_text_from_args(args)
    input_json = json.loads(Path(args.input_json).read_text(encoding="utf-8"))

    refined_categories, report, timings = main_pipeline(
        text=text,
        input_json=input_json,
        model_name=args.model_name,
        context_window=args.context_window,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        span_threshold=args.span_threshold,
        device=args.device,
    )

    payload = {
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "timings": timings,
        "counts": {
            "diseases": len(refined_categories["diseases"]),
            "genes_proteins": len(refined_categories["genes_proteins"]),
            "pathways": len(refined_categories["pathways"]),
        },
        **refined_categories,
    }

    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Refinement complete in %.2fs | diseases=%d genes_proteins=%d pathways=%d | output=%s | report=%s",
        payload["runtime_seconds"],
        len(refined_categories["diseases"]),
        len(refined_categories["genes_proteins"]),
        len(refined_categories["pathways"]),
        args.output,
        args.report,
    )


if __name__ == "__main__":
    main()
