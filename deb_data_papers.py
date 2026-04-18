"""
deb_data_papers.py
=================
Paper-centric knowledge graph data structure for TAU-KG.
Stores papers, extracted entities, relationships, and metadata.

Structure:
- papers_data: Metadata for each uploaded paper
- paper_entities: Extracted entities from papers
- paper_edges: Relationships found in papers
- paper_metadata: Aggregated statistics and tracking information
"""

from __future__ import annotations

import copy
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.paper_schema import normalize_entities_for_paper, normalize_relationships_for_paper


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORE_PATH = os.path.join(DATA_DIR, "paper_store.json")
STORE_VERSION = 2

ENTITY_TYPES = ("genes", "proteins", "diseases", "pathways")


def _empty_entity_bucket() -> Dict[str, List[Dict[str, Any]]]:
    return {entity_type: [] for entity_type in ENTITY_TYPES}


def _default_paper_metadata() -> Dict[str, Any]:
    return {
        "total_papers": 0,
        "papers_pending_extraction": 0,
        "papers_pending_review": 0,
        "papers_approved": 0,
        "total_entities_extracted": 0,
        "total_entities_approved": 0,
        "entity_coverage": {
            "genes": 0,
            "proteins": 0,
            "diseases": 0,
            "pathways": 0,
            "total": 0,
        },
        "genes_from_papers": [],
        "proteins_from_papers": [],
        "diseases_from_papers": [],
        "pathways_from_papers": [],
        "last_extraction_date": None,
        "last_review_date": None,
    }


# In-memory collections populated from the JSON store on import.
papers_data: List[Dict[str, Any]] = []
paper_entities: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
paper_edges: List[Dict[str, Any]] = []
paper_metadata: Dict[str, Any] = _default_paper_metadata()

_STORE_LOADED = False
_STORE_LOCK = threading.RLock()


def _normalize_entity_bucket(raw_bucket: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    bucket = _empty_entity_bucket()
    if not isinstance(raw_bucket, dict):
        return bucket

    for entity_type in ENTITY_TYPES:
        values = raw_bucket.get(entity_type, [])
        if isinstance(values, list):
            bucket[entity_type] = normalize_entities_for_paper("paper", {entity_type: values}).get(entity_type, [])
        else:
            bucket[entity_type] = []
    return bucket


def _normalize_edge(
    edge: Dict[str, Any],
    paper_id: Optional[str] = None,
    entities_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    normalized_paper_id = str(paper_id if paper_id is not None else edge.get("paper_id", "")).strip() or "paper"
    normalized_edges = normalize_relationships_for_paper(
        normalized_paper_id,
        [copy.deepcopy(edge)],
        entities_by_type=entities_by_type or {},
    )
    return normalized_edges[0] if normalized_edges else {
        "id": f"{normalized_paper_id}_rel_0",
        "paper_id": normalized_paper_id,
        "source_id": "",
        "source_name": "",
        "source_type": "Entity",
        "target_id": "",
        "target_name": "",
        "target_type": "Entity",
        "edge_type": "ASSOCIATES",
        "edge_weight": 0.5,
        "evidence": "",
        "source_chromosome": "",
        "original_relation": "",
        "extraction_method": "gpt4",
        "approved": False,
        "notes": "",
    }


def _recompute_metadata(existing_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata = _default_paper_metadata()
    if isinstance(existing_metadata, dict):
        metadata["last_extraction_date"] = existing_metadata.get("last_extraction_date")
        metadata["last_review_date"] = existing_metadata.get("last_review_date")

    metadata["total_papers"] = len(papers_data)
    metadata["papers_pending_extraction"] = sum(
        1 for paper in papers_data if paper.get("extraction_status") == "pending"
    )
    metadata["papers_pending_review"] = sum(
        1 for paper in papers_data if paper.get("extraction_status") == "reviewed"
    )
    metadata["papers_approved"] = sum(
        1 for paper in papers_data if paper.get("extraction_status") == "approved"
    )

    unique_entities = {entity_type: set() for entity_type in ENTITY_TYPES}
    for paper_id, entities in paper_entities.items():
        bucket = _normalize_entity_bucket(entities)
        paper_entities[paper_id] = bucket

        for entity_type in ENTITY_TYPES:
            values = bucket[entity_type]
            metadata["entity_coverage"][entity_type] += len(values)
            metadata["total_entities_extracted"] += len(values)
            metadata["total_entities_approved"] += sum(1 for entity in values if entity.get("approved", False))
            unique_entities[entity_type].update(
                str(entity.get("name", "")).strip()
                for entity in values
                if str(entity.get("name", "")).strip()
            )

    metadata["entity_coverage"]["total"] = sum(
        metadata["entity_coverage"][entity_type] for entity_type in ENTITY_TYPES
    )
    metadata["genes_from_papers"] = sorted(unique_entities["genes"])
    metadata["proteins_from_papers"] = sorted(unique_entities["proteins"])
    metadata["diseases_from_papers"] = sorted(unique_entities["diseases"])
    metadata["pathways_from_papers"] = sorted(unique_entities["pathways"])

    return metadata


def _save_store_locked() -> str:
    """Persist the in-memory paper store to disk. Caller must hold `_STORE_LOCK`."""
    global paper_metadata

    paper_metadata = _recompute_metadata(paper_metadata)
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {
        "version": STORE_VERSION,
        "papers_data": papers_data,
        "paper_entities": paper_entities,
        "paper_edges": paper_edges,
        "paper_metadata": paper_metadata,
    }

    temp_path = f"{STORE_PATH}.{threading.get_ident()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, STORE_PATH)
    return STORE_PATH


def load_store() -> Dict[str, Any]:
    """Load persisted paper data from disk into the in-memory store."""
    global papers_data, paper_entities, paper_edges, paper_metadata, _STORE_LOADED
    with _STORE_LOCK:
        if os.path.exists(STORE_PATH):
            with open(STORE_PATH, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = {}

        papers_raw = payload.get("papers_data", [])
        entities_raw = payload.get("paper_entities", {})
        edges_raw = payload.get("paper_edges", [])
        metadata_raw = payload.get("paper_metadata", {})

        papers_data = copy.deepcopy(papers_raw) if isinstance(papers_raw, list) else []
        paper_entities = {}
        if isinstance(entities_raw, dict):
            for paper_id, bucket in entities_raw.items():
                normalized_paper_id = str(paper_id)
                normalized_bucket = _normalize_entity_bucket(bucket)
                paper_entities[normalized_paper_id] = normalize_entities_for_paper(
                    normalized_paper_id,
                    normalized_bucket,
                )

        paper_edges = []
        if isinstance(edges_raw, list):
            for edge in edges_raw:
                if isinstance(edge, dict):
                    normalized_paper_id = str(edge.get("paper_id", "")).strip() or "paper"
                    paper_edges.append(
                        _normalize_edge(
                            edge,
                            paper_id=normalized_paper_id,
                            entities_by_type=paper_entities.get(normalized_paper_id, {}),
                        )
                    )

        paper_metadata = _recompute_metadata(metadata_raw)
        _STORE_LOADED = True
        return {
            "papers_data": papers_data,
            "paper_entities": paper_entities,
            "paper_edges": paper_edges,
            "paper_metadata": paper_metadata,
        }


def ensure_loaded() -> None:
    """Ensure the JSON-backed store has been loaded."""
    with _STORE_LOCK:
        if not _STORE_LOADED:
            load_store()


def save_store() -> str:
    """Persist the in-memory paper store to disk atomically."""
    with _STORE_LOCK:
        ensure_loaded()
        return _save_store_locked()


def add_paper(
    paper_id,
    title,
    authors,
    pmid,
    doi,
    abstract,
    pdf_path,
    publication_date,
    source="user_uploaded",
    source_url="",
    sections=None,
):
    """
    Add or update a paper record.

    Args:
        paper_id (str): Unique identifier
        title (str): Paper title
        authors (list): List of author names
        pmid (str): PubMed ID
        doi (str): DOI
        abstract (str): Paper abstract
        pdf_path (str): Path to stored PDF or empty string
        publication_date (str): Publication date in YYYY-MM-DD format
        source (str): Origin of the paper data
        source_url (str): Original import URL

    Returns:
        dict: The stored paper record
    """
    with _STORE_LOCK:
        ensure_loaded()

        normalized_authors = authors if isinstance(authors, list) else [authors] if authors else []
        existing = next((paper for paper in papers_data if paper.get("paper_id") == paper_id), None)

        if existing:
            existing.update(
                {
                    "title": title,
                    "authors": normalized_authors,
                    "pmid": pmid,
                    "doi": doi,
                    "abstract": abstract,
                    "pdf_path": pdf_path,
                    "publication_date": publication_date,
                    "source": source,
                    "source_url": source_url,
                    "sections": sections or [],
                }
            )
            paper = existing
        else:
            paper = {
                "paper_id": paper_id,
                "title": title,
                "authors": normalized_authors,
                "pmid": pmid,
                "doi": doi,
                "abstract": abstract,
                "pdf_path": pdf_path,
                "publication_date": publication_date,
                "source": source,
                "source_url": source_url,
                "sections": sections or [],
                "upload_date": datetime.now().isoformat(),
                "extraction_status": "pending",
                "notes": "",
            }
            papers_data.append(paper)

        _save_store_locked()
        return copy.deepcopy(paper)


def add_entities(paper_id, entity_type, entities):
    """
    Add extracted entities for a paper.

    Args:
        paper_id (str): Paper ID
        entity_type (str): One of genes, proteins, diseases, pathways
        entities (list): List of extracted entities
    """
    with _STORE_LOCK:
        ensure_loaded()
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        if paper_id not in paper_entities:
            paper_entities[paper_id] = _empty_entity_bucket()

        normalized_bucket = normalize_entities_for_paper(
            paper_id,
            {entity_type: copy.deepcopy(entities) if isinstance(entities, list) else []},
        )
        paper_entities[paper_id][entity_type] = normalized_bucket.get(entity_type, [])
        paper_metadata["last_extraction_date"] = datetime.now().isoformat()
        _save_store_locked()


def set_paper_entities(paper_id: str, entities_by_type: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Replace all entity buckets for a paper and persist them."""
    with _STORE_LOCK:
        ensure_loaded()
        current = normalize_entities_for_paper(paper_id, _normalize_entity_bucket(paper_entities.get(paper_id)))

        for entity_type in ENTITY_TYPES:
            if entity_type in entities_by_type:
                values = entities_by_type.get(entity_type, [])
                normalized_bucket = normalize_entities_for_paper(
                    paper_id,
                    {entity_type: copy.deepcopy(values) if isinstance(values, list) else []},
                )
                current[entity_type] = normalized_bucket.get(entity_type, [])

        paper_entities[paper_id] = current
        paper_metadata["last_review_date"] = datetime.now().isoformat()
        _save_store_locked()
        return copy.deepcopy(current)


def add_edges(paper_id, edges):
    """
    Append discovered relationships for a paper.

    Args:
        paper_id (str): Paper ID
        edges (list): List of relationship dicts
    """
    with _STORE_LOCK:
        ensure_loaded()
        if not isinstance(edges, list):
            return

        for edge in edges:
            if isinstance(edge, dict):
                paper_edges.append(_normalize_edge(edge, paper_id=paper_id, entities_by_type=paper_entities.get(paper_id, {})))

        paper_metadata["last_extraction_date"] = datetime.now().isoformat()
        _save_store_locked()


def set_paper_relationships(paper_id: str, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace all stored relationships for a paper and persist them."""
    with _STORE_LOCK:
        ensure_loaded()
        retained_edges = [edge for edge in paper_edges if edge.get("paper_id") != paper_id]

        normalized_relationships: List[Dict[str, Any]] = []
        if isinstance(relationships, list):
            for relationship in relationships:
                if isinstance(relationship, dict):
                    normalized_relationships.append(
                        _normalize_edge(
                            relationship,
                            paper_id=paper_id,
                            entities_by_type=paper_entities.get(paper_id, {}),
                        )
                    )

        paper_edges[:] = retained_edges + normalized_relationships
        paper_metadata["last_review_date"] = datetime.now().isoformat()
        _save_store_locked()
        return copy.deepcopy(normalized_relationships)


def update_paper_status(paper_id, status):
    """
    Update extraction status of a paper.

    Args:
        paper_id (str): Paper ID
        status (str): pending, extracted, reviewed, approved, skipped
    """
    with _STORE_LOCK:
        ensure_loaded()

        for paper in papers_data:
            if paper.get("paper_id") == paper_id:
                paper["extraction_status"] = status
                if status == "extracted":
                    paper_metadata["last_extraction_date"] = datetime.now().isoformat()
                if status in {"reviewed", "approved", "skipped"}:
                    paper_metadata["last_review_date"] = datetime.now().isoformat()
                _save_store_locked()
                return copy.deepcopy(paper)
        return None


def upsert_extracted_paper(
    paper_id: str,
    title: str,
    authors: List[str],
    pmid: str,
    doi: str,
    abstract: str,
    pdf_path: str,
    publication_date: str,
    entities_by_type: Dict[str, List[Dict[str, Any]]],
    relationships: List[Dict[str, Any]],
    source: str = "user_uploaded",
    source_url: str = "",
    sections: Optional[List[Dict[str, Any]]] = None,
    extraction_status: str = "extracted",
) -> Dict[str, Any]:
    """Atomically upsert a paper, its entities, and relationships in one disk write."""
    with _STORE_LOCK:
        ensure_loaded()

        normalized_authors = authors if isinstance(authors, list) else [authors] if authors else []
        existing = next((paper for paper in papers_data if paper.get("paper_id") == paper_id), None)

        if existing:
            existing.update(
                {
                    "title": title,
                    "authors": normalized_authors,
                    "pmid": pmid,
                    "doi": doi,
                    "abstract": abstract,
                    "pdf_path": pdf_path,
                    "publication_date": publication_date,
                    "source": source,
                    "source_url": source_url,
                    "sections": sections or [],
                    "extraction_status": extraction_status,
                }
            )
            paper = existing
        else:
            paper = {
                "paper_id": paper_id,
                "title": title,
                "authors": normalized_authors,
                "pmid": pmid,
                "doi": doi,
                "abstract": abstract,
                "pdf_path": pdf_path,
                "publication_date": publication_date,
                "source": source,
                "source_url": source_url,
                "sections": sections or [],
                "upload_date": datetime.now().isoformat(),
                "extraction_status": extraction_status,
                "notes": "",
            }
            papers_data.append(paper)

        bucket = normalize_entities_for_paper(
            paper_id,
            entities_by_type if isinstance(entities_by_type, dict) else {},
        )
        paper_entities[paper_id] = bucket

        retained_edges = [edge for edge in paper_edges if edge.get("paper_id") != paper_id]
        normalized_relationships: List[Dict[str, Any]] = []
        if isinstance(relationships, list):
            for relationship in relationships:
                if isinstance(relationship, dict):
                    normalized_relationships.append(
                        _normalize_edge(
                            relationship,
                            paper_id=paper_id,
                            entities_by_type=paper_entities.get(paper_id, {}),
                        )
                    )
        paper_edges[:] = retained_edges + normalized_relationships

        now_iso = datetime.now().isoformat()
        paper_metadata["last_extraction_date"] = now_iso
        if extraction_status in {"reviewed", "approved", "skipped"}:
            paper_metadata["last_review_date"] = now_iso

        _save_store_locked()
        return copy.deepcopy(paper)


def get_paper_by_id(paper_id):
    """Get paper details by ID."""
    ensure_loaded()
    for paper in papers_data:
        if paper.get("paper_id") == paper_id:
            return copy.deepcopy(paper)
    return None


def get_paper_entities(paper_id):
    """Get extracted entities for a paper."""
    ensure_loaded()
    entities = paper_entities.get(paper_id)
    return copy.deepcopy(entities) if entities else None


def get_paper_relationships(paper_id: str) -> List[Dict[str, Any]]:
    """Get extracted relationships for a paper."""
    ensure_loaded()
    return [
        copy.deepcopy(edge)
        for edge in paper_edges
        if edge.get("paper_id") == paper_id
    ]


def get_papers_for_entity(entity_name, entity_type=None):
    """
    Find all papers mentioning a specific entity.

    Args:
        entity_name (str): Entity name to search
        entity_type (str): Optional entity type filter

    Returns:
        list: List of (paper_id, entity_dict) tuples
    """
    ensure_loaded()
    results = []

    for paper_id, entities in paper_entities.items():
        if entity_type:
            types_to_search = [entity_type]
        else:
            types_to_search = list(ENTITY_TYPES)

        for current_type in types_to_search:
            for entity in entities.get(current_type, []):
                if str(entity.get("name", "")).lower() == str(entity_name).lower():
                    results.append((paper_id, copy.deepcopy(entity)))

    return results


def get_approved_entities():
    """Get all approved entities across all papers."""
    ensure_loaded()
    approved = _empty_entity_bucket()

    for paper_id, entities in paper_entities.items():
        for entity_type in ENTITY_TYPES:
            for entity in entities.get(entity_type, []):
                if entity.get("approved", False):
                    approved[entity_type].append({"paper_id": paper_id, **copy.deepcopy(entity)})

    return approved


def get_approved_edges():
    """Get all approved edges across all papers."""
    ensure_loaded()
    return [copy.deepcopy(edge) for edge in paper_edges if edge.get("approved", False)]


load_store()
