"""
pages/paper_upload.py
====================
Streamlit page for uploading research papers in PDF format.

Features:
- Multi-file PDF upload
- Progress tracking
- Automatic/manual PMID entry
- Text extraction validation
- Metadata extraction preview
"""

import streamlit as st
import uuid
import os
import re
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_processor import (
    get_pdf_extraction_backends,
    process_pdf,
    is_pmc_url,
    clean_pdf_text
)
from src.pmc_service import (
    PMC_FETCH_WORKER_RECOMMENDATION,
    normalize_section_type,
    process_pmc_url_advanced,
    process_pmc_url_html_fallback,
)
from src.paper_entity_extractor import (
    extract_entities_from_text,
    format_extraction_for_review,
    warmup_entity_extraction_models,
)
from src.paper_schema import normalize_entities_for_paper, normalize_relationships_for_paper
from vector_db_manager import VectorDBManager
import deb_data_papers as papers_db
from logger_config import setup_logger
from dotenv import load_dotenv

load_dotenv()

logger = setup_logger(__name__)

PMC_FETCH_WORKERS = PMC_FETCH_WORKER_RECOMMENDATION
AI_EXTRACT_WORKERS = 4
ACTIVE_JOB_REFRESH_SECONDS = 2.0
DEFAULT_PAPER_SAVE_MODE = "UPSERT"
PAPER_SAVE_MODE_ENV_KEYS = ("PAPER_SAVE_MODE", "PAPER_SAVE_CONFIG", "CONFIG")


def _build_paper_entity_summary_rows():
    """Build live per-paper summary rows from saved and in-session extractions."""
    rows = []

    # Saved papers/entities from in-memory papers database.
    paper_edge_counts = {}
    for edge in papers_db.paper_edges:
        paper_id = edge.get("paper_id")
        if paper_id:
            paper_edge_counts[paper_id] = paper_edge_counts.get(paper_id, 0) + 1

    for paper in papers_db.papers_data:
        paper_id = paper.get("paper_id", "")
        entities = papers_db.paper_entities.get(paper_id, {})
        rows.append({
            "paper": paper.get("title") or paper_id,
            "status": paper.get("extraction_status", "saved"),
            "genes": len(entities.get("genes", [])),
            "proteins": len(entities.get("proteins", [])),
            "diseases": len(entities.get("diseases", [])),
            "pathways": len(entities.get("pathways", [])),
            "relationships": paper_edge_counts.get(paper_id, 0),
        })

    # Unsaved, completed extraction rows from current session.
    for key, value in st.session_state.items():
        if not str(key).startswith("current_extraction_"):
            continue
        meta = value.get("paper_metadata", {})
        extracted = value.get("extracted_entities", {})
        rows.append({
            "paper": meta.get("title") or key.replace("current_extraction_", ""),
            "status": "ready_to_save",
            "genes": len(extracted.get("genes", [])),
            "proteins": len(extracted.get("proteins", [])),
            "diseases": len(extracted.get("diseases", [])),
            "pathways": len(extracted.get("pathways", [])),
            "relationships": len(extracted.get("relationships", [])),
        })

    # Running/failed async jobs that are not in the extraction cache yet.
    for source_key, job in st.session_state.ai_jobs.items():
        if f"current_extraction_{source_key}" in st.session_state:
            continue
        status = job.get("status")
        if status not in {"running", "error"}:
            continue
        rows.append({
            "paper": source_key,
            "status": status,
            "genes": 0,
            "proteins": 0,
            "diseases": 0,
            "pathways": 0,
            "relationships": 0,
        })

    return rows


def _render_live_entity_summary_panel():
    """Top-level consolidated per-paper extraction summary panel."""
    st.subheader("📚 Live Per-Paper Entity Summary")
    summary_rows = _build_paper_entity_summary_rows()
    if not summary_rows:
        st.info("No extracted paper entities yet. Start by fetching PMC links or uploading PDFs.")
        return

    st.dataframe(summary_rows, width="stretch", hide_index=True)

    totals = {
        "genes": sum(r["genes"] for r in summary_rows),
        "proteins": sum(r["proteins"] for r in summary_rows),
        "diseases": sum(r["diseases"] for r in summary_rows),
        "pathways": sum(r["pathways"] for r in summary_rows),
        "relationships": sum(r["relationships"] for r in summary_rows),
    }
    cols = st.columns(5)
    cols[0].metric("Genes", totals["genes"])
    cols[1].metric("Proteins", totals["proteins"])
    cols[2].metric("Diseases", totals["diseases"])
    cols[3].metric("Pathways", totals["pathways"])
    cols[4].metric("Relationships", totals["relationships"])


def _sync_paper_nodes_to_vector_db():
    """Rebuild the unified knowledge index from canonical paper storage."""
    manager = _get_vector_db_manager()
    rebuild_stats = manager.rebuild_knowledge_index_from_store(include_curated=True)
    db_stats = manager.get_database_stats()
    return rebuild_stats, db_stats


@st.cache_resource(show_spinner=False)
def _get_vector_db_manager():
    """Reuse the vector DB manager across reruns to avoid reloading embeddings."""
    return VectorDBManager()


def _get_vector_db_manager_safe() -> tuple[Optional[VectorDBManager], str]:
    """Best-effort vector DB manager accessor that does not break paper autosave."""
    try:
        return _get_vector_db_manager(), ""
    except Exception as exc:
        return None, str(exc)


def _flush_pending_toasts():
    """Show queued toast notifications once per rerun."""
    pending_toasts = st.session_state.get("pending_toasts", [])
    if not pending_toasts:
        return

    for toast in pending_toasts:
        st.toast(toast.get("message", "Paper saved"), icon=toast.get("icon", "✅"))
    st.session_state.pending_toasts = []


def _build_stable_paper_id(
    pmcid: str,
    source_key: str,
    pmid: str,
    doi: str,
    title: str,
    file_path: str,
    source_url: str,
) -> str:
    """Generate a deterministic paper ID so autosave stays idempotent."""
    normalized_pmcid = _extract_pmcid(pmcid)
    if normalized_pmcid:
        return normalized_pmcid

    normalized_pmid = str(pmid or "").strip()
    if normalized_pmid:
        return normalized_pmid

    normalized_doi = str(doi or "").strip().lower()
    if normalized_doi:
        return f"doi_{hashlib.sha1(normalized_doi.encode('utf-8')).hexdigest()[:16]}"

    seed = "|".join([
        str(source_key or ""),
        str(file_path or ""),
        str(source_url or ""),
        str(title or ""),
    ])
    return f"paper_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"


def _extract_pmcid(value: Any) -> str:
    match = re.search(r"(PMC\d+)", str(value or ""), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _resolve_paper_save_mode() -> str:
    """Return paper save mode from env: UPSERT (default) or OVERWRITE."""
    raw_value = ""
    for env_key in PAPER_SAVE_MODE_ENV_KEYS:
        candidate = str(os.getenv(env_key, "") or "").strip()
        if candidate:
            raw_value = candidate
            break

    normalized = raw_value.upper()
    if normalized in {"OVERWRITE", "OVERWITE"}:
        return "OVERWRITE"
    if normalized == "UPSERT":
        return "UPSERT"
    return DEFAULT_PAPER_SAVE_MODE


def _merge_entities_for_upsert(
    paper_id: str,
    existing_entities: Optional[Dict[str, List[Dict[str, Any]]]],
    new_entities: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Merge entities by stable entity id so UPSERT adds/refreshes without dropping unmatched rows."""
    merged = {"genes": [], "proteins": [], "diseases": [], "pathways": []}
    existing_normalized = normalize_entities_for_paper(paper_id, existing_entities or {})
    new_normalized = normalize_entities_for_paper(paper_id, new_entities or {})

    for bucket in ("genes", "proteins", "diseases", "pathways"):
        by_id: Dict[str, Dict[str, Any]] = {}
        for entity in existing_normalized.get(bucket, []):
            entity_id = str((entity or {}).get("id", "")).strip()
            if entity_id:
                by_id[entity_id] = dict(entity)
        for entity in new_normalized.get(bucket, []):
            entity_id = str((entity or {}).get("id", "")).strip()
            if entity_id:
                by_id[entity_id] = dict(entity)
        merged[bucket] = list(by_id.values())
    return merged


def _merge_relationships_for_upsert(
    paper_id: str,
    existing_relationships: Optional[List[Dict[str, Any]]],
    new_relationships: List[Dict[str, Any]],
    entities_for_lookup: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Merge relationships by semantic key so UPSERT keeps old + new without duplicates."""
    existing_normalized = normalize_relationships_for_paper(
        paper_id,
        existing_relationships or [],
        entities_for_lookup,
    )
    new_normalized = normalize_relationships_for_paper(
        paper_id,
        new_relationships or [],
        entities_for_lookup,
    )

    merged_map: Dict[str, Dict[str, Any]] = {}

    def _edge_key(edge: Dict[str, Any]) -> str:
        return "|".join(
            [
                str(edge.get("source_id", "")).strip().lower(),
                str(edge.get("target_id", "")).strip().lower(),
                str(edge.get("edge_type", "")).strip().upper(),
                str(edge.get("evidence", "")).strip().lower(),
            ]
        )

    for edge in existing_normalized:
        merged_map[_edge_key(edge)] = dict(edge)
    for edge in new_normalized:
        merged_map[_edge_key(edge)] = dict(edge)

    return list(merged_map.values())


def _prepare_entities_for_storage(extracted_entities: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize extracted entity payloads before storing them."""
    return normalize_entities_for_paper("paper", extracted_entities)


def _prepare_relationships_for_storage(extracted_entities: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize extracted relationship payloads before storing them."""
    return normalize_relationships_for_paper(
        "paper",
        extracted_entities.get("relationships", []),
        extracted_entities,
    )


def _build_autosave_toast_message(
    paper_title: str,
    prepared_entities: Dict[str, Any],
    relationships: List[Dict[str, Any]],
    sync_stats: Dict[str, Any],
) -> str:
    """Build the user-facing autosave notification."""
    indexed_entities = int(sync_stats.get("entities", 0))
    indexed_relationships = int(sync_stats.get("relationships", 0))
    return (
        f"Saved '{paper_title[:60] or 'Untitled paper'}' to the knowledge index. "
        f"Entities extracted: {sum(len(values) for values in prepared_entities.values())}. "
        f"Relationships extracted: {len(relationships)}. "
        f"Approved records indexed now: {indexed_entities} entities, {indexed_relationships} relationships."
    )


def _auto_save_paper_and_index(
    source_key: str,
    metadata: Dict[str, Any],
    title: str,
    authors: List[str],
    pmid_input: str,
    doi_input: str,
    publication_date,
    clean_text: str,
    file_path: str,
    source_url: str,
    extraction_state_key: str,
) -> Dict[str, Any]:
    """Persist a completed extraction and index it into the vector DB once."""
    autosaved_papers = st.session_state.autosaved_papers
    if source_key in autosaved_papers:
        return autosaved_papers[source_key]

    save_mode = _resolve_paper_save_mode()
    pmcid = _extract_pmcid(metadata.get("pmcid") or source_url)
    extracted_entities = st.session_state[extraction_state_key]["extracted_entities"]
    paper_id = _build_stable_paper_id(
        pmcid=pmcid,
        source_key=source_key,
        pmid=pmid_input,
        doi=doi_input,
        title=title,
        file_path=file_path,
        source_url=source_url,
    )
    prepared_entities = normalize_entities_for_paper(paper_id, extracted_entities)
    relationships = normalize_relationships_for_paper(
        paper_id,
        extracted_entities.get("relationships", []),
        prepared_entities,
    )

    manager, manager_error = _get_vector_db_manager_safe()
    existing_same_pmc = papers_db.find_paper_by_pmcid(pmcid) if pmcid else None
    existing_same_pmc_id = str((existing_same_pmc or {}).get("paper_id", "")).strip()
    existing_same_id = papers_db.get_paper_by_id(paper_id)

    # Ensure PMCID becomes canonical paper_id when available.
    if existing_same_pmc_id and existing_same_pmc_id != paper_id:
        legacy_entities = papers_db.get_paper_entities(existing_same_pmc_id) or {}
        legacy_relationships = papers_db.get_paper_relationships(existing_same_pmc_id) or []
        if save_mode == "UPSERT":
            prepared_entities = _merge_entities_for_upsert(paper_id, legacy_entities, prepared_entities)
            relationships = _merge_relationships_for_upsert(paper_id, legacy_relationships, relationships, prepared_entities)
        papers_db.delete_paper(existing_same_pmc_id)
        if manager is not None:
            manager.delete_paper_records_from_knowledge(existing_same_pmc_id)

    if existing_same_id:
        if save_mode == "UPSERT":
            existing_entities = papers_db.get_paper_entities(paper_id) or {}
            existing_relationships = papers_db.get_paper_relationships(paper_id) or []
            prepared_entities = _merge_entities_for_upsert(paper_id, existing_entities, prepared_entities)
            relationships = _merge_relationships_for_upsert(paper_id, existing_relationships, relationships, prepared_entities)
        else:
            papers_db.delete_paper(paper_id)
            if manager is not None:
                manager.delete_paper_records_from_knowledge(paper_id)

    stored_paper = papers_db.upsert_extracted_paper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        pmid=pmid_input,
        doi=doi_input,
        abstract=metadata.get("abstract", ""),
        pdf_path=file_path,
        publication_date=publication_date.isoformat(),
        entities_by_type=prepared_entities,
        relationships=relationships,
        source="pmc_link" if source_url else "user_uploaded",
        source_url=source_url,
        sections=metadata.get("sections", []),
        extraction_status="extracted",
        pmcid=pmcid,
    )

    sync_stats = {"papers": 0, "entities": 0, "relationships": 0}
    if manager is not None:
        sync_stats = manager.upsert_paper_records_to_knowledge(paper_id, include_pending=False)

    save_result = {
        "paper_id": paper_id,
        "title": title,
        "entity_counts": {
            "genes": len(prepared_entities.get("genes", [])),
            "proteins": len(prepared_entities.get("proteins", [])),
            "diseases": len(prepared_entities.get("diseases", [])),
            "pathways": len(prepared_entities.get("pathways", [])),
            "relationships": len(relationships),
        },
        "sync_stats": sync_stats,
        "saved_at": datetime.now().isoformat(),
        "save_mode": save_mode,
        "indexing_error": manager_error if manager is None else "",
    }
    autosaved_papers[source_key] = save_result
    st.session_state.paper_save_status[source_key] = {"status": "saved", "details": save_result}
    st.session_state.pending_toasts.append({
        "icon": "✅",
        "message": (
            _build_autosave_toast_message(title, prepared_entities, relationships, sync_stats)
            if manager is not None
            else (
                f"Saved '{title[:60] or 'Untitled paper'}' to paper store, "
                "but vector indexing is temporarily unavailable (read-only DB path)."
            )
        ),
    })

    if paper_id not in st.session_state.uploaded_papers:
        st.session_state.uploaded_papers.append(paper_id)
    st.session_state.pop(extraction_state_key, None)
    return save_result


def _make_streamlit_key(prefix: str, source_key: str) -> str:
    """Build a stable, widget-safe Streamlit key for a specific paper source."""
    safe_source = re.sub(r"[^a-zA-Z0-9_]+", "_", source_key).strip("_")
    return f"{prefix}_{safe_source or 'paper'}"


def _make_pmc_source_key(pmc_url: str, index: int) -> str:
    """Build stable source keys so retries and re-submissions map to the same job."""
    normalized = str(pmc_url or "").strip().lower().rstrip("/")
    pmcid = _extract_pmcid(normalized)
    if pmcid:
        return f"pmc_{pmcid.lower()}"

    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"pmc_{digest}"


def _make_uploaded_file_source_key(uploaded_file) -> str:
    """Build a stable key for uploaded PDFs so reruns do not duplicate work."""
    digest = hashlib.sha1()
    digest.update(uploaded_file.name.encode("utf-8"))
    digest.update(str(uploaded_file.size).encode("utf-8"))
    digest.update(uploaded_file.getbuffer())
    return f"pdf_{digest.hexdigest()[:16]}"


def _extract_abstract_from_sections(sections: Optional[List[Dict[str, Any]]]) -> str:
    """Recover an abstract from normalized PMC sections when metadata does not provide one."""
    if not sections:
        return ""

    abstract_parts = []
    for section in sections:
        if normalize_section_type(section.get("type", "")) != "abstract":
            continue
        section_text = (section.get("text") or "").strip()
        if section_text:
            abstract_parts.append(section_text)

    return "\n\n".join(abstract_parts).strip()


def _fetch_pmc_payload(pmc_url: str):
    """Background fetch: HTML-first PMC ingestion with BioC fallback."""
    advanced = process_pmc_url_advanced(pmc_url)
    sections = advanced.get("sections", [])
    abstract_text = _extract_abstract_from_sections(sections)

    metadata = {
        "title": advanced.get("title", ""),
        "authors": advanced.get("authors", []),
        "abstract": abstract_text,
        "keywords": [],
        "pmcid": advanced.get("pmcid", ""),
        "pmid": advanced.get("pmid", ""),
        "doi": advanced.get("doi", ""),
        "journal": advanced.get("journal", ""),
        "publication_date": advanced.get("pubdate", ""),
        "sections": sections,
    }

    full_text = advanced.get("text")

    if not full_text:
        fallback = process_pmc_url_html_fallback(pmc_url)
        fallback_metadata = {
            "title": fallback.get("title", ""),
            "authors": fallback.get("authors", []),
            "abstract": _extract_abstract_from_sections(fallback.get("sections", [])),
            "keywords": [],
            "pmcid": fallback.get("pmcid", ""),
            "pmid": fallback.get("pmid", ""),
            "doi": fallback.get("doi", ""),
            "journal": fallback.get("journal", ""),
            "publication_date": fallback.get("pubdate", ""),
            "sections": fallback.get("sections", []),
        }
        metadata.update({k: v for k, v in fallback_metadata.items() if v})
        full_text = fallback.get("text", "")

    if not full_text:
        raise RuntimeError("No text extracted from PMC article via API or HTML fallback")

    clean_text = clean_pdf_text(full_text)
    return metadata, clean_text


def _submit_pmc_jobs(pmc_urls):
    """Submit PMC fetch jobs concurrently and keep stable ordering/state."""
    for index, pmc_url in enumerate(pmc_urls):
        source_key = _make_pmc_source_key(pmc_url, index)
        existing = st.session_state.pmc_jobs.get(source_key)
        if existing and existing.get("status") in {"running", "done"}:
            continue

        # Reset any stale extraction/saving state before retrying this same source key.
        st.session_state.ai_jobs.pop(source_key, None)
        st.session_state.autosaved_papers.pop(source_key, None)
        st.session_state.paper_save_status.pop(source_key, None)
        st.session_state.pop(f"current_extraction_{source_key}", None)

        future = st.session_state.pmc_executor.submit(_fetch_pmc_payload, pmc_url)
        st.session_state.pmc_jobs[source_key] = {
            "url": pmc_url,
            "status": "running",
            "stage": "fetch",
            "future": future,
            "metadata": None,
            "clean_text": "",
            "error": "",
        }
        if source_key not in st.session_state.pmc_job_order:
            st.session_state.pmc_job_order.append(source_key)


def _get_existing_entities_for_boosting():
    """Collect current graph entities to boost extraction confidence."""
    return {
        "genes": [n["id"] for n in __import__("deb_data").nodes_data if n.get("type") == "gene"],
        "proteins": [n["id"] for n in __import__("deb_data").nodes_data if n.get("type") == "protein"],
        "diseases": [n["id"] for n in __import__("deb_data").nodes_data if n.get("type") == "disease"],
        "pathways": [n["id"] for n in __import__("deb_data").nodes_data if n.get("type") == "pathway"],
    }


def _run_entity_extraction(
    clean_text: str,
    title: str,
    abstract: str,
    sections: Optional[List[Dict[str, Any]]] = None,
):
    """Worker function for background AI entity extraction."""
    existing_entities = _get_existing_entities_for_boosting()
    return extract_entities_from_text(
        clean_text,
        title=title,
        abstract=abstract,
        sections=sections,
        existing_entities=existing_entities,
    )


def _submit_ai_job(
    source_key: str,
    clean_text: str,
    title: str,
    abstract: str,
    sections: Optional[List[Dict[str, Any]]] = None,
    force: bool = False,
):
    """Submit asynchronous AI extraction for a paper if not already submitted."""
    existing = st.session_state.ai_jobs.get(source_key)
    if not force and existing and existing.get("status") in {"running", "done", "error"}:
        return

    future = st.session_state.ai_executor.submit(
        _run_entity_extraction,
        clean_text,
        title,
        abstract,
        sections,
    )
    st.session_state.ai_jobs[source_key] = {
        "status": "running",
        "future": future,
        "result": None,
        "error": "",
    }


def _poll_ai_jobs():
    """Update AI extraction job statuses from background futures."""
    for source_key, job in st.session_state.ai_jobs.items():
        if job.get("status") != "running":
            continue
        future = job.get("future")
        if future and future.done():
            try:
                result = future.result()
                if "error" in result:
                    job["status"] = "error"
                    job["error"] = str(result.get("error", "Unknown extraction error"))
                else:
                    job["status"] = "done"
                    job["result"] = result
                job["future"] = None
            except Exception as e:
                job["status"] = "error"
                job["error"] = str(e)
                job["future"] = None


def _poll_pmc_jobs():
    """Track PMC jobs across fetch -> extraction -> done lifecycle."""
    for source_key, job in st.session_state.pmc_jobs.items():
        if job.get("status") != "running":
            continue

        stage = str(job.get("stage", "fetch"))
        if stage == "fetch":
            future = job.get("future")
            if future is None and job.get("clean_text"):
                job["stage"] = "extract"
                metadata = job.get("metadata", {}) or {}
                _submit_ai_job(
                    source_key,
                    str(job.get("clean_text", "")),
                    str(metadata.get("title", "")),
                    str(metadata.get("abstract", "")),
                    metadata.get("sections", []),
                )
                continue

            if not future or not future.done():
                continue
            try:
                metadata, clean_text = future.result()
                job["metadata"] = metadata
                job["clean_text"] = clean_text
                job["future"] = None
                job["stage"] = "extract"
                _submit_ai_job(
                    source_key,
                    clean_text,
                    str((metadata or {}).get("title", "")),
                    str((metadata or {}).get("abstract", "")),
                    (metadata or {}).get("sections", []),
                )
            except Exception as e:
                job["status"] = "error"
                job["stage"] = "failed"
                job["error"] = str(e)
                job["future"] = None
            continue

        if stage == "extract":
            ai_job = st.session_state.ai_jobs.get(source_key, {})
            ai_status = ai_job.get("status")
            if ai_status == "done":
                job["status"] = "done"
                job["stage"] = "done"
            elif ai_status == "error":
                job["status"] = "error"
                job["stage"] = "failed"
                job["error"] = str(ai_job.get("error", "Unknown extraction error"))
            elif ai_status is None and job.get("clean_text"):
                metadata = job.get("metadata", {}) or {}
                _submit_ai_job(
                    source_key,
                    str(job.get("clean_text", "")),
                    str(metadata.get("title", "")),
                    str(metadata.get("abstract", "")),
                    metadata.get("sections", []),
                )


def _poll_model_warmup_job():
    """Update one-time entity model warmup status from background future."""
    warmup_job = st.session_state.get("model_warmup_job", {})
    if warmup_job.get("status") != "running":
        return

    future = warmup_job.get("future")
    if not future or not future.done():
        return

    try:
        warmup_job["result"] = future.result()
        warmup_job["status"] = "done"
    except Exception as exc:
        warmup_job["status"] = "error"
        warmup_job["error"] = str(exc)
    finally:
        warmup_job["future"] = None


def _has_running_background_jobs() -> bool:
    """Return True when any PMC or AI background work is still in progress."""
    model_warmup_running = st.session_state.get("model_warmup_job", {}).get("status") == "running"
    return (
        any(job.get("status") == "running" for job in st.session_state.pmc_jobs.values())
        or any(job.get("status") == "running" for job in st.session_state.ai_jobs.values())
        or model_warmup_running
    )


def _auto_refresh_background_jobs():
    """Trigger periodic reruns while background jobs are active."""
    if not _has_running_background_jobs():
        return

    st.caption(f"Auto-refreshing every {ACTIVE_JOB_REFRESH_SECONDS:.0f}s while background jobs are running.")
    time.sleep(ACTIVE_JOB_REFRESH_SECONDS)
    st.rerun()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "uploaded_papers" not in st.session_state:
        st.session_state.uploaded_papers = []
    if "current_extraction" not in st.session_state:
        st.session_state.current_extraction = None
    if "extraction_progress" not in st.session_state:
        st.session_state.extraction_progress = 0
    if "uploaded_file_payloads" not in st.session_state:
        st.session_state.uploaded_file_payloads = {}
    if "pmc_executor" not in st.session_state:
        st.session_state.pmc_executor = ThreadPoolExecutor(max_workers=PMC_FETCH_WORKERS)
    if "pmc_jobs" not in st.session_state:
        st.session_state.pmc_jobs = {}
    if "pmc_job_order" not in st.session_state:
        st.session_state.pmc_job_order = []
    if "ai_executor" not in st.session_state:
        st.session_state.ai_executor = ThreadPoolExecutor(max_workers=AI_EXTRACT_WORKERS)
    if "ai_jobs" not in st.session_state:
        st.session_state.ai_jobs = {}
    if "model_warmup_job" not in st.session_state:
        warmup_future = st.session_state.ai_executor.submit(warmup_entity_extraction_models)
        st.session_state.model_warmup_job = {
            "status": "running",
            "future": warmup_future,
            "result": None,
            "error": "",
        }
    if "autosaved_papers" not in st.session_state:
        st.session_state.autosaved_papers = {}
    if "pending_toasts" not in st.session_state:
        st.session_state.pending_toasts = []
    if "paper_save_status" not in st.session_state:
        st.session_state.paper_save_status = {}


def create_upload_directory():
    """Ensure uploaded_papers directory exists."""
    upload_dir = "./uploaded_papers"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        logger.info(f"Created upload directory: {upload_dir}")
    return upload_dir


def save_uploaded_pdf(uploaded_file, upload_dir: str) -> str:
    """
    Save uploaded PDF file to local storage.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        upload_dir: Directory to save file
    
    Returns:
        str: Path to saved file
    """
    # Generate unique filename
    file_extension = ".pdf"
    unique_name = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_name)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    logger.info(f"Saved PDF: {file_path}")
    return file_path


def _get_or_create_pdf_payload(uploaded_file, upload_dir: str, pmid: Optional[str] = None):
    """Cache PDF extraction results in session state so reruns stay idempotent."""
    source_key = _make_uploaded_file_source_key(uploaded_file)
    cached_payload = st.session_state.uploaded_file_payloads.get(source_key)
    if cached_payload:
        return cached_payload

    file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
    if file_size_mb > 50:
        raise ValueError(f"File size {file_size_mb:.1f}MB exceeds 50MB limit")

    file_path = save_uploaded_pdf(uploaded_file, upload_dir)
    metadata, full_text = process_pdf(file_path, max_pages=10)
    clean_text = clean_pdf_text(full_text)

    if pmid:
        metadata["pmid"] = pmid

    payload = {
        "file_path": file_path,
        "metadata": metadata,
        "clean_text": clean_text,
        "source_key": source_key,
    }
    st.session_state.uploaded_file_payloads[source_key] = payload
    return payload


def process_extracted_paper(metadata, clean_text: str, source_label: str, file_path: str, source_url: str = "", source_key: str = "default"):
    """Render the common extraction and save workflow for PDFs and PMC links."""
    st.success(f"✓ {source_label} extracted successfully")
    widget_key = _make_streamlit_key("paper", source_key)
    extraction_state_key = f"current_extraction_{source_key}"

    # Display extracted metadata
    st.subheader("📄 Extracted Metadata")
    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input(
            "Title",
            value=metadata.get("title", ""),
            help="Paper title",
            key=f"{widget_key}_title"
        )
        authors_str = st.text_area(
            "Authors",
            value="\n".join(metadata.get("authors", [])),
            height=100,
            help="One author per line",
            key=f"{widget_key}_authors"
        )
        authors = [a.strip() for a in authors_str.split("\n") if a.strip()]

    with col2:
        pmid_input = st.text_input(
            "PubMed ID (PMID)",
            value=metadata.get("pmid", ""),
            help="8-digit PMID",
            key=f"{widget_key}_pmid"
        )
        doi_input = st.text_input(
            "DOI",
            value=metadata.get("doi", ""),
            help="Digital Object Identifier",
            key=f"{widget_key}_doi"
        )

    publication_date = st.date_input(
        "Publication Date",
        value=datetime.now().date(),
        help="When the paper was published",
        key=f"{widget_key}_publication_date"
    )

    if metadata.get("pmcid"):
        st.info(f"PMCID: {metadata['pmcid']}")

    # Display abstract preview
    if metadata.get("abstract"):
        with st.expander("📝 Abstract Preview"):
            st.text(metadata.get("abstract", "")[:500])

    with st.expander("📖 Extracted Full Paper Text"):
        st.text_area(
            "Full extracted text",
            value=clean_text,
            height=420,
            disabled=True,
            key=f"{widget_key}_full_text_preview"
        )

    if source_url:
        st.markdown(f"**Source URL:** [{source_url}]({source_url})")

    # Autonomous asynchronous entity extraction (no manual click required).
    _submit_ai_job(
        source_key,
        clean_text,
        title,
        metadata.get("abstract", ""),
        metadata.get("sections", []),
    )

    ai_job = st.session_state.ai_jobs.get(source_key, {})
    ai_status = ai_job.get("status")

    if ai_status == "running":
        st.info("⏳ AI entity extraction is running in background for this paper...")
    elif ai_status == "error":
        st.error(f"❌ Extraction error: {ai_job.get('error', 'Unknown error')}")
        if st.button("Retry AI extraction", key=f"retry_ai_{source_key}"):
            _submit_ai_job(
                source_key,
                clean_text,
                title,
                metadata.get("abstract", ""),
                metadata.get("sections", []),
                force=True,
            )
            st.rerun()
    elif ai_status == "done":
        extracted = ai_job.get("result", {})
        st.session_state[extraction_state_key] = {
            "paper_metadata": {
                "title": title,
                "authors": authors,
                "pmid": pmid_input,
                "doi": doi_input,
                "publication_date": publication_date.isoformat(),
                "abstract": metadata.get("abstract", ""),
                "pdf_path": file_path,
                "source_url": source_url,
                "source": "pmc_link" if source_url else "user_uploaded",
                "upload_date": datetime.now().isoformat()
            },
            "extracted_entities": extracted,
            "full_text": clean_text
        }

        st.success("✓ Entities extracted successfully")

        st.subheader("🎯 Extraction Summary")
        summary_cols = st.columns(4)

        with summary_cols[0]:
            st.metric("Genes", len(extracted.get("genes", [])))
        with summary_cols[1]:
            st.metric("Proteins", len(extracted.get("proteins", [])))
        with summary_cols[2]:
            st.metric("Diseases", len(extracted.get("diseases", [])))
        with summary_cols[3]:
            st.metric("Pathways", len(extracted.get("pathways", [])))

        if extracted.get("relationships"):
            with st.expander(f"🔗 Relationships ({len(extracted['relationships'])} found)"):
                for i, rel in enumerate(extracted["relationships"][:5]):
                    st.write(
                        f"**{rel.get('source_name', rel.get('source', 'N/A'))}** "
                        f"({rel.get('source_type', 'Entity')}) "
                        f"→ *{rel.get('edge_type', rel.get('relation', 'ASSOCIATES'))}* → "
                        f"**{rel.get('target_name', rel.get('target', 'N/A'))}** "
                        f"({rel.get('target_type', 'Entity')}) "
                        f"(weight: {float(rel.get('edge_weight', rel.get('confidence', 0.0))):.2f})"
                    )

    save_state = st.session_state.paper_save_status.get(source_key, {})
    if extraction_state_key in st.session_state and source_key not in st.session_state.autosaved_papers:
        try:
            save_result = _auto_save_paper_and_index(
                source_key=source_key,
                metadata=metadata,
                title=title,
                authors=authors,
                pmid_input=pmid_input,
                doi_input=doi_input,
                publication_date=publication_date,
                clean_text=clean_text,
                file_path=file_path,
                source_url=source_url,
                extraction_state_key=extraction_state_key,
            )
            logger.info(f"Paper autosaved and indexed: {save_result['paper_id']}")
            st.rerun()
        except Exception as e:
            save_state = {"status": "error", "error": str(e)}
            st.session_state.paper_save_status[source_key] = save_state
            logger.error(f"Autosave error for {source_key}: {e}", exc_info=True)

    save_state = st.session_state.paper_save_status.get(source_key, save_state)
    if save_state.get("status") == "saved":
        details = save_state.get("details", {})
        entity_counts = details.get("entity_counts", {})
        sync_stats = details.get("sync_stats", {})
        indexing_error = str(details.get("indexing_error", "")).strip()
        st.success(
            (
                "Auto-saved to paper store and unified knowledge index. "
                if not indexing_error
                else "Auto-saved to paper store (indexing pending). "
            )
            + f"Paper ID: `{details.get('paper_id', '')}` | "
            + f"Indexed approved entities: {sync_stats.get('entities', 0)} | "
            + f"Indexed approved relationships: {sync_stats.get('relationships', 0)}"
        )
        if indexing_error:
            st.warning(
                "Paper was saved, but vector indexing failed due to DB write access. "
                f"Details: {indexing_error}"
            )
        st.caption(
            "Entities indexed: "
            f"genes {entity_counts.get('genes', 0)}, "
            f"proteins {entity_counts.get('proteins', 0)}, "
            f"diseases {entity_counts.get('diseases', 0)}, "
            f"pathways {entity_counts.get('pathways', 0)}, "
            f"relationships {entity_counts.get('relationships', 0)}."
        )
        return
    if save_state.get("status") == "error":
        st.error(f"Automatic save failed: {save_state.get('error', 'Unknown error')}")
        if st.button("Retry automatic save", key=f"retry_save_{source_key}"):
            st.session_state.paper_save_status.pop(source_key, None)
            st.session_state.autosaved_papers.pop(source_key, None)
            st.rerun()
        return

    # Save paper to database
    if extraction_state_key in st.session_state:
        if st.button("💾 Save Paper & Entities to Database", key=f"save_btn_{source_key}"):
            try:
                save_mode = _resolve_paper_save_mode()
                pmcid = _extract_pmcid(metadata.get("pmcid") or source_url)
                extracted_entities = st.session_state[extraction_state_key]["extracted_entities"]
                paper_id = _build_stable_paper_id(
                    pmcid=pmcid,
                    source_key=source_key,
                    pmid=pmid_input,
                    doi=doi_input,
                    title=title,
                    file_path=file_path,
                    source_url=source_url,
                )

                prepared_entities = normalize_entities_for_paper(paper_id, extracted_entities)
                relationships = normalize_relationships_for_paper(
                    paper_id,
                    extracted_entities.get("relationships", []),
                    prepared_entities,
                )

                manager, manager_error = _get_vector_db_manager_safe()
                existing_same_pmc = papers_db.find_paper_by_pmcid(pmcid) if pmcid else None
                existing_same_pmc_id = str((existing_same_pmc or {}).get("paper_id", "")).strip()
                existing_same_id = papers_db.get_paper_by_id(paper_id)

                if existing_same_pmc_id and existing_same_pmc_id != paper_id:
                    legacy_entities = papers_db.get_paper_entities(existing_same_pmc_id) or {}
                    legacy_relationships = papers_db.get_paper_relationships(existing_same_pmc_id) or []
                    if save_mode == "UPSERT":
                        prepared_entities = _merge_entities_for_upsert(paper_id, legacy_entities, prepared_entities)
                        relationships = _merge_relationships_for_upsert(paper_id, legacy_relationships, relationships, prepared_entities)
                    papers_db.delete_paper(existing_same_pmc_id)
                    if manager is not None:
                        manager.delete_paper_records_from_knowledge(existing_same_pmc_id)

                if existing_same_id:
                    if save_mode == "UPSERT":
                        existing_entities = papers_db.get_paper_entities(paper_id) or {}
                        existing_relationships = papers_db.get_paper_relationships(paper_id) or []
                        prepared_entities = _merge_entities_for_upsert(paper_id, existing_entities, prepared_entities)
                        relationships = _merge_relationships_for_upsert(
                            paper_id,
                            existing_relationships,
                            relationships,
                            prepared_entities,
                        )
                    else:
                        papers_db.delete_paper(paper_id)
                        if manager is not None:
                            manager.delete_paper_records_from_knowledge(paper_id)

                papers_db.upsert_extracted_paper(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    pmid=pmid_input,
                    doi=doi_input,
                    abstract=metadata.get("abstract", ""),
                    pdf_path=file_path,
                    publication_date=publication_date.isoformat(),
                    entities_by_type=prepared_entities,
                    relationships=relationships,
                    source="pmc_link" if source_url else "user_uploaded",
                    source_url=source_url,
                    sections=metadata.get("sections", []),
                    extraction_status="extracted",
                    pmcid=pmcid,
                )
                if manager is not None:
                    manager.upsert_paper_records_to_knowledge(paper_id, include_pending=False)

                st.success(f"✅ Paper saved successfully! Paper ID: `{paper_id}`")
                if manager is None and manager_error:
                    st.warning(
                        "Paper was saved to the JSON store, but vector indexing is currently unavailable: "
                        f"{manager_error}"
                    )
                if paper_id not in st.session_state.uploaded_papers:
                    st.session_state.uploaded_papers.append(paper_id)
                st.session_state.pop(extraction_state_key, None)

                logger.info(f"Paper saved: {paper_id}")

            except Exception as e:
                st.error(f"❌ Error saving paper: {str(e)}")
                logger.error(f"Save error: {e}", exc_info=True)


def process_single_pdf(file_path: str, pmid: Optional[str] = None):
    """
    Process a single PDF file through the full pipeline.
    
    Args:
        file_path: Path to PDF file
        pmid: Optional PubMed ID (if known)
    """
    try:
        # Extract metadata and text
        with st.spinner("Reading PDF... (🔄 Processing)"):
            metadata, full_text = process_pdf(file_path, max_pages=10)
            clean_text = clean_pdf_text(full_text)

        if pmid:
            metadata["pmid"] = pmid

        process_extracted_paper(metadata, clean_text, "PDF", file_path, source_key=os.path.basename(file_path))
    
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        logger.error(f"PDF processing error: {e}", exc_info=True)


def main():
    """Main Streamlit app for paper upload."""
    st.set_page_config(
        page_title="📤 Paper Upload",
        page_icon="📤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    _poll_ai_jobs()
    _poll_pmc_jobs()
    _poll_model_warmup_job()
    _flush_pending_toasts()
    
    # Page header
    st.title("📤 Upload Research Papers")
    st.markdown(
        "Upload PDF research papers to extract genes, proteins, diseases, and pathways. "
        "The system will automatically extract entities and relationships for manual review."
    )
    warmup_job = st.session_state.get("model_warmup_job", {})
    if warmup_job.get("status") == "running":
        st.caption("Preparing NER models once for this app session...")
    elif warmup_job.get("status") == "error":
        st.warning(
            "Model warmup failed; extraction will continue with lazy loading. "
            f"Details: {warmup_job.get('error', 'Unknown warmup error')}"
        )

    _render_live_entity_summary_panel()

    if st.button("🧠 Sync New Nodes To Vector DB", key="sync_nodes_vector_db"):
        try:
            with st.spinner("Rebuilding unified knowledge index..."):
                sync_stats, db_stats = _sync_paper_nodes_to_vector_db()
            st.success(
                "✅ Knowledge index rebuild complete: "
                f"curated entities {sync_stats.get('curated_entities', 0)}, "
                f"papers {sync_stats.get('papers', 0)}, "
                f"paper entities {sync_stats.get('entities', 0)}, "
                f"relationships {sync_stats.get('relationships', 0)}."
            )
            st.info(
                "Knowledge index documents - "
                f"total: {db_stats.get('total_documents', 0)}, "
                f"papers: {db_stats.get('papers_documents', 0)}, "
                f"paper_entities: {db_stats.get('paper_entities_documents', 0)}, "
                f"relationships: {db_stats.get('paper_edges_documents', 0)}, "
                f"curated_entities: {db_stats.get('curated_entities_documents', 0)}"
            )
        except Exception as e:
            st.error(f"❌ Vector DB sync failed: {e}")
            logger.error(f"Vector DB sync error: {e}", exc_info=True)
    
    # Create upload directory
    upload_dir = create_upload_directory()
    
    source_mode = st.radio(
        "Choose paper source",
        ["Paste PMC/NLM links", "Upload PDF files"],
        horizontal=True
    )

    if source_mode == "Upload PDF files":
        st.subheader("📁 Upload PDF Files")
        pdf_backends = get_pdf_extraction_backends()
        if not pdf_backends:
            st.error(
                "PDF extraction backends are missing in this Python environment. "
                "Install `pdfplumber` and `pypdf` before uploading PDFs."
            )
        else:
            st.caption(f"PDF text extraction backends: {', '.join(pdf_backends)}")
        uploaded_files = st.file_uploader(
            "Choose PDF files to upload",
            type="pdf",
            accept_multiple_files=True,
            help="Supported formats: PDF. Max 50MB per file."
        )

        if uploaded_files:
            st.info(f"📌 {len(uploaded_files)} file(s) selected for processing")

            # Process each uploaded file
            for uploaded_file in uploaded_files:
                st.divider()
                st.subheader(f"Processing: {uploaded_file.name}")

                try:
                    payload = _get_or_create_pdf_payload(uploaded_file, upload_dir)
                    st.success(f"✓ File ready: `{os.path.basename(payload['file_path'])}`")
                    process_extracted_paper(
                        payload["metadata"],
                        payload["clean_text"],
                        "PDF",
                        payload["file_path"],
                        source_key=payload["source_key"],
                    )

                except ImportError as e:
                    st.error(f"❌ PDF extraction backend error: {str(e)}")
                    logger.error(f"PDF backend error: {e}", exc_info=True)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"File processing error: {e}", exc_info=True)
    else:
        st.subheader("🔗 Paste PMC / NLM Links")
        st.markdown(
            "Paste one or more PMC links from the National Library of Medicine, one per line. "
            "Example: `https://pmc.ncbi.nlm.nih.gov/articles/PMC12330721/`"
        )
        pmc_urls_text = st.text_area(
            "PMC article URLs",
            height=150,
            placeholder="https://pmc.ncbi.nlm.nih.gov/articles/PMC12330721/\nhttps://pmc.ncbi.nlm.nih.gov/articles/PMC12512994/",
            key="pmc_urls_input"
        )

        if st.button("Fetch and Process PMC Links", key="pmc_fetch_btn"):
            pmc_urls = [line.strip() for line in pmc_urls_text.splitlines() if line.strip()]

            if not pmc_urls:
                st.warning("Enter at least one PMC link.")
            else:
                _submit_pmc_jobs(pmc_urls)

        if st.session_state.pmc_jobs:
            running = sum(1 for j in st.session_state.pmc_jobs.values() if j.get("status") == "running")
            done = sum(1 for j in st.session_state.pmc_jobs.values() if j.get("status") == "done")
            failed = sum(1 for j in st.session_state.pmc_jobs.values() if j.get("status") == "error")
            st.info(f"PMC jobs - running: {running}, done: {done}, failed: {failed}")

        for source_key in st.session_state.pmc_job_order:
            job = st.session_state.pmc_jobs.get(source_key)
            if not job:
                continue

            pmc_url = job.get("url", "")
            st.divider()
            st.subheader(pmc_url)

            if not is_pmc_url(pmc_url):
                st.warning("This link does not look like a PMC article URL, but processing was attempted.")

            status = job.get("status")
            if status == "running":
                stage = str(job.get("stage", "fetch"))
                if stage == "fetch":
                    st.info("⏳ Fetching and normalizing article... this runs in background.")
                    continue

                st.info("⏳ Article fetched. Running entity extraction...")
                process_extracted_paper(
                    job.get("metadata", {}),
                    job.get("clean_text", ""),
                    "PMC article",
                    "",
                    source_url=pmc_url,
                    source_key=source_key,
                )
                continue

            if status == "error":
                st.error(f"❌ Error processing PMC link: {job.get('error', 'Unknown error')}")
                if st.button("Retry", key=f"retry_{source_key}"):
                    _submit_pmc_jobs([pmc_url])
                    st.rerun()
                continue

            if status == "done":
                process_extracted_paper(
                    job.get("metadata", {}),
                    job.get("clean_text", ""),
                    "PMC article",
                    "",
                    source_url=pmc_url,
                    source_key=source_key,
                )
    
    # Show upload statistics in sidebar
    with st.sidebar:
        st.subheader("📊 Statistics")
        st.metric("Total Papers", papers_db.paper_metadata["total_papers"])
        st.metric("Pending Review", papers_db.paper_metadata["papers_pending_review"])
        st.metric("Approved", papers_db.paper_metadata["papers_approved"])
        st.metric("Total Entities", papers_db.paper_metadata["total_entities_extracted"])

        if st.button("📈 Refresh Vector DB Stats", key="refresh_vector_stats"):
            try:
                vdb_stats = VectorDBManager().get_database_stats()
                st.session_state["vector_stats_cache"] = vdb_stats
            except Exception as e:
                st.error(f"Stats refresh failed: {e}")

        vdb_stats = st.session_state.get("vector_stats_cache")
        if vdb_stats:
            st.metric("Vector Main Docs", vdb_stats.get("total_documents", 0))
            st.metric("Vector Papers Docs", vdb_stats.get("papers_documents", 0))
            st.metric("Vector Entity Docs", vdb_stats.get("paper_entities_documents", 0))
            st.metric("Vector All Docs", vdb_stats.get("total_documents_all_collections", 0))
        
        st.divider()
        
        if st.button("🔄 Refresh Statistics"):
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Next Steps:**")
        st.markdown("1. Upload PDF papers or paste PMC links")
        st.markdown("2. Review extracted entities")
        st.markdown("3. Approve and merge to graph")
        st.markdown("4. Use in chat and visualization")

    _auto_refresh_background_jobs()


if __name__ == "__main__":
    main()
