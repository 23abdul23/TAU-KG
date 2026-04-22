import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import biobert_refinement
import scispacy_pipeline


logger = logging.getLogger("ner_extracion_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_mode_requested(args: argparse.Namespace) -> bool:
    return bool(args.text or args.text_file or args.source or args.url)


def _empty_entities() -> Dict[str, List[Dict[str, Any]]]:
    return {"diseases": [], "genes_proteins": [], "pathways": []}


def _resolve_pmcid_from_args(args: argparse.Namespace) -> str:
    source_value = (args.source or args.url or "").strip()
    if not source_value:
        return ""
    return scispacy_pipeline.normalize_pmcid(source_value)


def _resolve_output_paths(args: argparse.Namespace, pmcid: str) -> None:
    if pmcid:
        if not args.output:
            args.output = f"test/{pmcid}_output.json"
        if not args.report:
            args.report = f"test/{pmcid}_report.json"
    else:
        if not args.output:
            args.output = "test/output_ner_extracion.json"
        if not args.report:
            args.report = "test/report_ner_extracion.json"


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    pmcid = _resolve_pmcid_from_args(args)
    _resolve_output_paths(args, pmcid)

    input_build_seconds = 0.0
    text = ""
    scispacy_entities = _empty_entities()
    scispacy_timings: Dict[str, Any] = {}
    scispacy_used = False

    if _extract_mode_requested(args):
        input_start = time.perf_counter()
        if pmcid:
            xml_content = scispacy_pipeline.fetch_article_xml(pmcid)
            text = scispacy_pipeline.extract_article_text(xml_content)
        else:
            text = scispacy_pipeline.build_text_from_args(args)
        input_build_seconds = round(time.perf_counter() - input_start, 3)
        logger.info("Running SciSpaCy extraction stage...")
        scispacy_used = True
        scispacy_entities, scispacy_timings = scispacy_pipeline.main_pipeline(text)
    elif args.input_json:
        logger.info("Skipping SciSpaCy stage (using --input-json directly).")
        scispacy_entities = _load_json(args.input_json)
    else:
        raise ValueError("Provide extraction input (--text/--text-file/--source) or --input-json.")

    if args.scispacy_output:
        intermediate_payload = {
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "timings": {
                "input_build_seconds": input_build_seconds,
                **scispacy_timings,
            },
            "counts": {
                "diseases": len(scispacy_entities.get("diseases", [])),
                "genes_proteins": len(scispacy_entities.get("genes_proteins", [])),
                "pathways": len(scispacy_entities.get("pathways", [])),
            },
            **scispacy_entities,
        }
        _save_json(args.scispacy_output, intermediate_payload)

    logger.info("Running BioBERT refinement stage...")
    refined_categories, refinement_report, refinement_timings = biobert_refinement.main_pipeline(
        text=text,
        input_json=scispacy_entities,
        model_name=args.model_name,
        context_window=args.context_window,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        span_threshold=args.span_threshold,
        device=args.device,
    )

    final_payload = {
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "timings": {
            "input_build_seconds": input_build_seconds,
            "scispacy": scispacy_timings,
            "biobert_refinement": refinement_timings,
        },
        "counts": {
            "diseases": len(refined_categories["diseases"]),
            "genes_proteins": len(refined_categories["genes_proteins"]),
            "pathways": len(refined_categories["pathways"]),
        },
        **refined_categories,
    }

    report_payload = {
        "pipeline": {
            "scispacy_used": scispacy_used,
            "pmcid": pmcid or None,
            "scispacy_input_counts": {
                "diseases": len(scispacy_entities.get("diseases", [])),
                "genes_proteins": len(scispacy_entities.get("genes_proteins", [])),
                "pathways": len(scispacy_entities.get("pathways", [])),
            },
        },
        **refinement_report,
    }

    _save_json(args.output, final_payload)
    _save_json(args.report, report_payload)

    logger.info(
        "NER_extracion complete in %.2fs | diseases=%d genes_proteins=%d pathways=%d | output=%s | report=%s",
        final_payload["runtime_seconds"],
        final_payload["counts"]["diseases"],
        final_payload["counts"]["genes_proteins"],
        final_payload["counts"]["pathways"],
        args.output,
        args.report,
    )
    return {"output": final_payload, "report": report_payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="PMCID/URL -> SciSpaCy extraction -> BioBERT refinement")

    # Extraction inputs
    parser.add_argument("--text", help="Raw input text")
    parser.add_argument("--text-file", help="Path to UTF-8 text file")
    parser.add_argument("--source", help="PMCID/PMC URL (e.g., PMC1234567 or https://pmc.../PMC1234567)")
    parser.add_argument("--url", help="PMC article URL (alternative to --source)")

    # Optional direct refinement input
    parser.add_argument("--input-json", help="Existing extracted JSON (diseases/genes_proteins/pathways)")

    # Outputs
    parser.add_argument(
        "--output",
        default=None,
        help="Final refined JSON output path (default: test/PMC<id>_output.json for PMCID/URL)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Refinement report JSON output path (default: test/PMC<id>_report.json for PMCID/URL)",
    )
    parser.add_argument(
        "--scispacy-output",
        help="Optional path to save intermediate SciSpaCy output JSON before refinement",
    )

    # BioBERT refinement options
    parser.add_argument("--model-name", default=biobert_refinement.DEFAULT_MODEL_NAME, help="HuggingFace NER model")
    parser.add_argument("--context-window", type=int, default=biobert_refinement.DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--batch-size", type=int, default=biobert_refinement.DEFAULT_BATCH_SIZE)
    parser.add_argument("--confidence-threshold", type=float, default=biobert_refinement.DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--span-threshold", type=float, default=biobert_refinement.DEFAULT_SPAN_THRESHOLD)
    parser.add_argument("--device", type=int, default=None, help="Transformers pipeline device id (-1 for CPU)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    args = parser.parse_args()

    logger.setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))
    scispacy_pipeline.logger.setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))
    biobert_refinement.logger.setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))

    run_pipeline(args)


if __name__ == "__main__":
    main()
