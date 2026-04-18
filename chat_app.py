import json
import os
import random
import re
import time
from collections import defaultdict
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from typing import List, Dict, Any, Optional, Set, Tuple

from vector_db_manager import VectorDBManager

# Paper integration imports
try:
    import deb_data_papers as papers_db
    PAPERS_AVAILABLE = True
except ImportError:
    PAPERS_AVAILABLE = False

try:
    import deb_data
    GRAPH_DATA_AVAILABLE = True
except ImportError:
    GRAPH_DATA_AVAILABLE = False

try:
    from citations import extract_entities_with_llm as extract_query_entities
except Exception:
    extract_query_entities = None

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Knowledge Chat",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = Path("data")
CHAT_STATE_DIR = DATA_DIR / "chat_sessions"
CHAT_STATE_DIR.mkdir(parents=True, exist_ok=True)

SESSION_STATE_DEFAULTS = {
    "db_manager": None,
    "chat_history": [],
    "chat_citations": {},
    "chat_papers": {},
    "db_loaded": False,
    "db_error": "",
    "gene_network": defaultdict(set),
    "query_genes": [],
    "search_result_limit": 15,
    "selected_chat_turn_index": 0,
    "next_query_input": "",
    "clear_next_query_input_pending": False,
    "persisted_state_loaded": False,
}

for session_key, default_value in SESSION_STATE_DEFAULTS.items():
    if session_key not in st.session_state:
        st.session_state[session_key] = deepcopy(default_value)


def get_or_create_browser_session_id() -> str:
    """Persist a lightweight browser session id in the URL so refresh keeps the same chat state."""
    session_id = st.session_state.get("browser_session_id", "")
    if session_id:
        return session_id

    raw_value = st.query_params.get("session_id", "")
    if isinstance(raw_value, list):
        raw_value = raw_value[0] if raw_value else ""

    session_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(raw_value or "").strip())
    if not session_id:
        session_id = uuid4().hex
        st.query_params["session_id"] = session_id

    st.session_state.browser_session_id = session_id
    return session_id


def get_chat_state_path(session_id: Optional[str] = None) -> Path:
    active_session_id = session_id or get_or_create_browser_session_id()
    return CHAT_STATE_DIR / f"{active_session_id}.json"


def _serialize_citation(citation: Any) -> Dict[str, Any]:
    if hasattr(citation, "to_dict"):
        try:
            return dict(citation.to_dict())
        except Exception:
            pass

    if isinstance(citation, dict):
        return dict(citation)

    return {
        "title": str(getattr(citation, "title", "")),
        "authors": str(getattr(citation, "authors", "")),
        "journal": str(getattr(citation, "journal", "")),
        "year": str(getattr(citation, "year", "")),
        "pmid": str(getattr(citation, "pmid", "")),
        "doi": str(getattr(citation, "doi", "")),
        "is_review": bool(getattr(citation, "is_review", False)),
    }


def _serialize_chat_state() -> Dict[str, Any]:
    search_results = {
        key: value
        for key, value in st.session_state.items()
        if key.startswith("search_results_")
    }

    serialized_citations = {
        key: [_serialize_citation(item) for item in value]
        for key, value in st.session_state.chat_citations.items()
    }

    return {
        "chat_history": list(st.session_state.chat_history),
        "chat_citations": serialized_citations,
        "chat_papers": dict(st.session_state.chat_papers),
        "query_genes": list(st.session_state.query_genes),
        "search_result_limit": int(st.session_state.search_result_limit),
        "selected_chat_turn_index": int(st.session_state.selected_chat_turn_index),
        "next_query_input": str(st.session_state.next_query_input),
        "stored_search_results": search_results,
    }


def persist_chat_state() -> None:
    """Write the current browser session's chat state to disk."""
    session_path = get_chat_state_path()
    payload = _serialize_chat_state()
    session_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_persisted_chat_state() -> None:
    """Restore persisted chat state exactly once per Streamlit server session."""
    if st.session_state.persisted_state_loaded:
        return

    st.session_state.persisted_state_loaded = True
    session_path = get_chat_state_path()
    if not session_path.exists():
        return

    try:
        payload = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception:
        return

    for key in [item for item in list(st.session_state.keys()) if item.startswith("search_results_")]:
        del st.session_state[key]

    st.session_state.chat_history = payload.get("chat_history", [])
    st.session_state.chat_citations = payload.get("chat_citations", {})
    st.session_state.chat_papers = payload.get("chat_papers", {})
    st.session_state.query_genes = payload.get("query_genes", [])
    st.session_state.search_result_limit = int(payload.get("search_result_limit", 15))
    st.session_state.selected_chat_turn_index = int(payload.get("selected_chat_turn_index", 0))
    st.session_state.next_query_input = str(payload.get("next_query_input", ""))

    for key, value in payload.get("stored_search_results", {}).items():
        st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_cached_db_manager() -> VectorDBManager:
    return VectorDBManager()


def ensure_database_ready(show_spinner: bool = False) -> bool:
    """Automatically bootstrap the persisted Chroma database and optional paper collection."""
    if st.session_state.db_loaded and st.session_state.db_manager is not None:
        return True

    try:
        spinner_context = st.spinner("Loading vector database...") if show_spinner else nullcontext()
        with spinner_context:
            db_manager = get_cached_db_manager()
            db_manager.load_csv_to_vectordb("nodes_main.csv")

            if PAPERS_AVAILABLE and getattr(papers_db, "papers_data", None):
                db_manager.create_papers_collection()
                if db_manager.papers_collection.count() == 0:
                    db_manager.load_papers_to_vectordb(papers_db.papers_data)

            st.session_state.db_manager = db_manager
            st.session_state.db_loaded = True
            st.session_state.db_error = ""
        return True
    except Exception as exc:
        st.session_state.db_manager = None
        st.session_state.db_loaded = False
        st.session_state.db_error = str(exc)
        return False


get_or_create_browser_session_id()
load_persisted_chat_state()

if st.session_state.clear_next_query_input_pending:
    st.session_state.next_query_input = ""
    st.session_state.clear_next_query_input_pending = False


QUERY_COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

NODE_TYPE_COLORS = {
    "gene": "#1f77b4",
    "protein": "#2ca02c",
    "gene/protein": "#ff7f0e",
    "disease": "#d62728",
    "pathway": "#8c564b",
    "paper": "#7f7f7f",
    "entity": "#9467bd",
}

NODE_TYPE_SHAPES = {
    "gene": "dot",
    "protein": "square",
    "gene/protein": "hexagon",
    "disease": "diamond",
    "pathway": "triangle",
    "paper": "database",
    "entity": "dot",
}


def schedule_next_query_input_clear() -> None:
    """Clear the query box on the next rerun before the widget is recreated."""
    st.session_state.clear_next_query_input_pending = True


def clear_stored_search_results() -> None:
    """Remove persisted per-turn search payloads from session state."""
    for key in [item for item in list(st.session_state.keys()) if item.startswith("search_results_")]:
        del st.session_state[key]


def reset_chat_state() -> None:
    """Clear all persisted chat artifacts for the current browser session."""
    st.session_state.chat_history = []
    st.session_state.chat_citations = {}
    st.session_state.chat_papers = {}
    st.session_state.query_genes = []
    st.session_state.selected_chat_turn_index = 0
    st.session_state.next_query_input = ""
    clear_stored_search_results()
    schedule_next_query_input_clear()
    persist_chat_state()


def normalize_node_type(node_type: Optional[str], default: str = "entity") -> str:
    """Canonicalize node type labels across curated, vector, and paper-derived sources."""
    raw = re.sub(r"\s+", "", str(node_type or "")).strip().lower()
    mapping = {
        "gene": "gene",
        "genes": "gene",
        "protein": "protein",
        "proteins": "protein",
        "disease": "disease",
        "diseases": "disease",
        "pathway": "pathway",
        "pathways": "pathway",
        "gene/protein": "gene/protein",
        "geneprotein": "gene/protein",
        "gene-protein": "gene/protein",
    }
    return mapping.get(raw, str(node_type or default).strip() or default)


def distance_to_similarity_percent(distance) -> float:
    """Convert vector distance into a bounded similarity percentage for display."""
    try:
        numeric_distance = max(float(distance), 0.0)
    except (TypeError, ValueError):
        return 0.0
    return 100.0 / (1.0 + numeric_distance)


def _normalize_result_key(result: Dict[str, Any]) -> str:
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    return re.sub(r"\s+", " ", str(metadata.get("node_name", "")).strip()).lower()


def _merge_search_results(results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for result in results:
        key = _normalize_result_key(result)
        if not key:
            continue

        current = merged.get(key)
        similarity = result.get("similarity_score", distance_to_similarity_percent(result.get("distance")))
        if not current or similarity > current.get("similarity_score", 0.0):
            merged[key] = result

    ranked = sorted(merged.values(), key=lambda item: item.get("similarity_score", 0.0), reverse=True)
    return ranked[:limit]


def _build_query_variants(query: str) -> List[str]:
    base_query = re.sub(r"\s+", " ", str(query or "")).strip()
    variants = [base_query] if base_query else []

    extracted_entities: Dict[str, List[str]] = {}
    if extract_query_entities and base_query:
        try:
            extracted = extract_query_entities(base_query) or {}
            if isinstance(extracted, dict):
                extracted_entities = {
                    key: [str(item).strip() for item in value if str(item).strip()]
                    for key, value in extracted.items()
                    if isinstance(value, list)
                }
        except Exception:
            extracted_entities = {}

    ordered_terms: List[str] = []
    for entity_type in ("genes", "proteins", "diseases", "pathways", "keywords"):
        ordered_terms.extend(extracted_entities.get(entity_type, []))

    if ordered_terms:
        variants.append(" ".join(ordered_terms[:8]))

    genes = extracted_entities.get("genes", [])
    proteins = extracted_entities.get("proteins", [])
    diseases = extracted_entities.get("diseases", [])
    pathways = extracted_entities.get("pathways", [])

    anchor_terms = genes[:3] + proteins[:2] + diseases[:2] + pathways[:2]
    for term in anchor_terms:
        variants.append(f"{base_query} {term}")

    if genes and diseases:
        variants.append(f"{' '.join(genes[:2])} {' '.join(diseases[:2])} mechanism pathway interaction")
    if genes and pathways:
        variants.append(f"{' '.join(genes[:2])} {' '.join(pathways[:2])} regulation network")
    if proteins and diseases:
        variants.append(f"{' '.join(proteins[:2])} {' '.join(diseases[:2])} disease mechanism")

    deduped_variants = list(dict.fromkeys(variant for variant in variants if variant))
    return deduped_variants[:8]


def _search_query_seed_nodes(db_manager: VectorDBManager, query: str, limit: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Run several focused searches so one prompt can seed a denser graph."""
    query_variants = _build_query_variants(query)
    if not query_variants:
        return [], []

    per_query_limit = max(5, min(limit, 10))
    collected: List[Dict[str, Any]] = []

    for index, variant in enumerate(query_variants):
        try:
            variant_limit = limit if index == 0 else per_query_limit
            collected.extend(db_manager.search_similar(variant, n_results=variant_limit))
        except Exception:
            continue

    return _merge_search_results(collected, limit), query_variants


def render_pyvis_network(net: Network, height: int) -> None:
    """Render a PyVis graph directly as HTML without temporary files."""
    html_content = net.generate_html(notebook=False)
    components.html(html_content, height=height)


def get_paper_display_fields(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize paper search results whether fields are flat or nested under metadata."""
    metadata = paper.get("metadata", {}) if isinstance(paper, dict) else {}
    merged = dict(metadata)
    if isinstance(paper, dict):
        merged.update({k: v for k, v in paper.items() if k != "metadata"})

    authors = merged.get("authors", [])
    if isinstance(authors, str):
        authors = [author for author in authors.split("|") if author]
    elif not isinstance(authors, list):
        authors = []
    merged["authors"] = authors

    if PAPERS_AVAILABLE:
        paper_id = str(merged.get("paper_id", "")).strip()
        if paper_id:
            paper_record = papers_db.get_paper_by_id(paper_id)
            if paper_record:
                merged["title"] = paper_record.get("title") or merged.get("title", "")
                merged["authors"] = paper_record.get("authors") or merged.get("authors", [])
                merged["abstract"] = paper_record.get("abstract", "")
                merged["pmid"] = paper_record.get("pmid") or merged.get("pmid", "")
                merged["doi"] = paper_record.get("doi") or merged.get("doi", "")
                merged["publication_date"] = paper_record.get("publication_date") or merged.get("publication_date", "")
                merged["source_url"] = paper_record.get("source_url", "")

    publication_date = str(merged.get("publication_date", "") or "")
    merged["publication_year"] = merged.get("publication_year") or (publication_date[:4] if len(publication_date) >= 4 else "")
    return merged


def normalize_node_key(name: str) -> str:
    """Normalize node names so duplicate hits collapse into one network node."""
    text = re.sub(r"\s+", " ", str(name or "").strip())
    return text.lower()


def singularize_entity_type(entity_type: str) -> str:
    mapping = {
        "genes": "gene",
        "proteins": "protein",
        "diseases": "disease",
        "pathways": "pathway",
    }
    return normalize_node_type(mapping.get(entity_type, entity_type or "entity"))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_source_name(value: Optional[str], default: str = "CURATED_GRAPH") -> str:
    text = str(value or "").strip()
    return text or default


def build_graph_node_lookup() -> Dict[str, Dict[str, Any]]:
    """Create a unified lookup for curated and paper-derived graph nodes."""
    node_lookup: Dict[str, Dict[str, Any]] = {}

    if GRAPH_DATA_AVAILABLE:
        for node in getattr(deb_data, "nodes_data", []):
            name = str(node.get("id", "")).strip()
            if not name:
                continue
            key = normalize_node_key(name)
            node_lookup[key] = {
                "key": key,
                "name": name,
                "type": normalize_node_type(node.get("type", "entity")),
                "source": normalize_source_name(node.get("source_type") or node.get("node_source"), "CURATED_GRAPH"),
                "cluster": node.get("cluster", ""),
            }

    if PAPERS_AVAILABLE:
        for paper_id, entities in getattr(papers_db, "paper_entities", {}).items():
            for entity_type, values in entities.items():
                for entity in values:
                    name = str(entity.get("name", "")).strip()
                    if not name:
                        continue
                    key = normalize_node_key(name)
                    existing = node_lookup.get(key, {})
                    node_lookup[key] = {
                        "key": key,
                        "name": existing.get("name", name),
                        "type": normalize_node_type(existing.get("type", singularize_entity_type(entity_type))),
                        "source": existing.get("source", "PAPER_INGEST"),
                        "cluster": existing.get("cluster", ""),
                    }

    return node_lookup


def build_graph_adjacency() -> Dict[str, List[Dict[str, Any]]]:
    """Create adjacency lists from curated graph edges and paper-derived edges."""
    adjacency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    if GRAPH_DATA_AVAILABLE:
        for edge in getattr(deb_data, "edges_data", []):
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue

            edge_record = {
                "source_name": source,
                "target_name": target,
                "relation": str(edge.get("relation", "related")),
                "score": safe_float(edge.get("score", 0.5), 0.5),
                "provenance": "CURATED_GRAPH",
            }
            source_key = normalize_node_key(source)
            target_key = normalize_node_key(target)
            adjacency[source_key].append({**edge_record, "neighbor_key": target_key, "neighbor_name": target})
            adjacency[target_key].append({**edge_record, "neighbor_key": source_key, "neighbor_name": source})

    if PAPERS_AVAILABLE:
        for edge in getattr(papers_db, "paper_edges", []):
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue

            edge_record = {
                "source_name": source,
                "target_name": target,
                "relation": str(edge.get("relation", "related")),
                "score": safe_float(edge.get("confidence", edge.get("score", 0.5)), 0.5),
                "provenance": "PAPER_INGEST",
                "paper_id": str(edge.get("paper_id", "")),
            }
            source_key = normalize_node_key(source)
            target_key = normalize_node_key(target)
            adjacency[source_key].append({**edge_record, "neighbor_key": target_key, "neighbor_name": target})
            adjacency[target_key].append({**edge_record, "neighbor_key": source_key, "neighbor_name": source})

    return adjacency


def merge_search_results(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate search hits by normalized node name while keeping the strongest match."""
    merged: Dict[str, Dict[str, Any]] = {}

    for result in raw_results:
        metadata = dict(result.get("metadata", {}))
        node_name = str(metadata.get("node_name", "")).strip()
        if not node_name:
            continue

        key = normalize_node_key(node_name)
        similarity = result.get("similarity_score", distance_to_similarity_percent(result.get("distance")))
        source = normalize_source_name(metadata.get("node_source"), "UNKNOWN")
        node_type = normalize_node_type(metadata.get("node_type", "entity"))

        current = merged.get(key)
        if not current or similarity > current["similarity_score"]:
            merged[key] = {
                "key": key,
                "document": result.get("document", ""),
                "metadata": metadata,
                "distance": result.get("distance"),
                "similarity_score": similarity,
                "sources": {source},
                "types": {node_type},
            }
        else:
            current["sources"].add(source)
            current["types"].add(node_type)
            if metadata.get("node_id") and not current["metadata"].get("node_id"):
                current["metadata"]["node_id"] = metadata["node_id"]

    merged_results = list(merged.values())
    merged_results.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
    return merged_results


def collect_network_filter_options() -> Tuple[List[str], List[str]]:
    """Collect the available node types and sources from current state and graph data."""
    node_lookup = build_graph_node_lookup()
    node_types: Set[str] = {normalize_node_type(node.get("type", "entity")) for node in node_lookup.values()}
    sources: Set[str] = {node.get("source", "CURATED_GRAPH") for node in node_lookup.values()}

    for query_data in st.session_state.query_genes:
        for result in query_data.get("genes", []):
            metadata = result.get("metadata", {})
            node_types.add(normalize_node_type(metadata.get("node_type", "entity")))
            sources.add(normalize_source_name(metadata.get("node_source"), "UNKNOWN"))

    node_types.discard("")
    sources.discard("")
    return sorted(node_types), sorted(sources)


def filter_seed_results(raw_results: List[Dict[str, Any]], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter, deduplicate, and rank vector-search results before graph expansion."""
    filtered: List[Dict[str, Any]] = []

    for result in merge_search_results(raw_results):
        metadata = result.get("metadata", {})
        similarity = result.get("similarity_score", 0.0)
        node_type = normalize_node_type(metadata.get("node_type", "entity"))
        source = normalize_source_name(metadata.get("node_source"), "UNKNOWN")

        if similarity < settings["min_similarity"]:
            continue
        if settings["allowed_types"] and node_type not in settings["allowed_types"]:
            continue
        if settings["allowed_sources"] and source not in settings["allowed_sources"]:
            continue

        filtered.append(result)

    return filtered[: settings["seed_limit"]]


def build_node_payload(
    name: str,
    key: str,
    query: str,
    node_lookup: Dict[str, Dict[str, Any]],
    metadata_override: Optional[Dict[str, Any]] = None,
    similarity_score: float = 0.0,
    is_seed: bool = False,
) -> Dict[str, Any]:
    """Create a consistent node payload used by both single and compound graphs."""
    lookup = node_lookup.get(key, {})
    metadata_override = metadata_override or {}
    node_type = normalize_node_type(metadata_override.get("node_type") or lookup.get("type") or "entity")
    node_source = normalize_source_name(
        metadata_override.get("node_source") or lookup.get("source"),
        "CURATED_GRAPH" if lookup else "UNKNOWN",
    )
    node_id = metadata_override.get("node_id") or lookup.get("name") or name

    return {
        "key": key,
        "name": name,
        "metadata": {
            "node_name": name,
            "node_id": node_id,
            "node_type": node_type,
            "node_source": node_source,
        },
        "similarity_score": similarity_score,
        "queries": {query},
        "is_seed": is_seed,
    }


def build_network_payload(query_entries: List[Dict[str, Any]], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Build a real graph network from retrieved seeds plus graph-neighbor expansion."""
    node_lookup = build_graph_node_lookup()
    adjacency = build_graph_adjacency()
    node_registry: Dict[str, Dict[str, Any]] = {}
    edge_registry: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    seed_counts: Dict[str, int] = {}

    for entry in query_entries:
        query = entry.get("query", "")
        raw_results = entry.get("genes", [])
        seed_results = filter_seed_results(raw_results, settings)
        seed_counts[query] = len(seed_results)

        for seed in seed_results:
            metadata = seed.get("metadata", {})
            seed_name = str(metadata.get("node_name", "")).strip()
            if not seed_name:
                continue

            seed_key = normalize_node_key(seed_name)
            if seed_key not in node_registry and len(node_registry) >= settings["max_nodes"]:
                break

            similarity = seed.get("similarity_score", 0.0)
            seed_payload = build_node_payload(
                name=seed_name,
                key=seed_key,
                query=query,
                node_lookup=node_lookup,
                metadata_override=metadata,
                similarity_score=similarity,
                is_seed=True,
            )

            existing_seed = node_registry.get(seed_key)
            if existing_seed:
                existing_seed["queries"].add(query)
                existing_seed["similarity_score"] = max(existing_seed["similarity_score"], similarity)
                existing_seed["is_seed"] = True
            else:
                node_registry[seed_key] = seed_payload

            neighbors_added = 0
            neighbor_candidates = sorted(
                adjacency.get(seed_key, []),
                key=lambda item: item.get("score", 0.0),
                reverse=True,
            )

            for edge in neighbor_candidates:
                if neighbors_added >= settings["neighbors_per_seed"]:
                    break
                if len(edge_registry) >= settings["max_edges"]:
                    break

                neighbor_key = edge.get("neighbor_key", "")
                neighbor_name = edge.get("neighbor_name", "")
                if not neighbor_key or not neighbor_name:
                    continue

                neighbor_lookup = node_lookup.get(neighbor_key, {})
                neighbor_type = normalize_node_type(neighbor_lookup.get("type", "entity"))
                neighbor_source = normalize_source_name(neighbor_lookup.get("source"), edge.get("provenance", "CURATED_GRAPH"))

                if settings["allowed_types"] and neighbor_type not in settings["allowed_types"]:
                    continue
                if settings["allowed_sources"] and neighbor_source not in settings["allowed_sources"]:
                    continue

                if neighbor_key not in node_registry and len(node_registry) >= settings["max_nodes"]:
                    continue

                neighbor_similarity = max(edge.get("score", 0.0) * 100.0, similarity * 0.7)
                if neighbor_key in node_registry:
                    node_registry[neighbor_key]["queries"].add(query)
                    node_registry[neighbor_key]["similarity_score"] = max(
                        node_registry[neighbor_key]["similarity_score"],
                        neighbor_similarity,
                    )
                else:
                    node_registry[neighbor_key] = build_node_payload(
                        name=neighbor_name,
                        key=neighbor_key,
                        query=query,
                        node_lookup=node_lookup,
                        similarity_score=neighbor_similarity,
                    )

                edge_key = tuple(sorted([seed_key, neighbor_key])) + (str(edge.get("relation", "related")),)
                existing_edge = edge_registry.get(edge_key)
                edge_payload = {
                    "source_key": seed_key,
                    "target_key": neighbor_key,
                    "score": safe_float(edge.get("score", 0.5), 0.5),
                    "relation": str(edge.get("relation", "related")),
                    "provenance": {str(edge.get("provenance", "CURATED_GRAPH"))},
                    "queries": {query},
                }

                if existing_edge:
                    existing_edge["score"] = max(existing_edge["score"], edge_payload["score"])
                    existing_edge["queries"].update(edge_payload["queries"])
                    existing_edge["provenance"].update(edge_payload["provenance"])
                else:
                    edge_registry[edge_key] = edge_payload
                    neighbors_added += 1

    edges = sorted(edge_registry.values(), key=lambda item: item["score"], reverse=True)[: settings["max_edges"]]
    kept_node_keys = {edge["source_key"] for edge in edges} | {edge["target_key"] for edge in edges}
    for key, node in node_registry.items():
        if node.get("is_seed"):
            kept_node_keys.add(key)

    nodes = [node_registry[key] for key in kept_node_keys if key in node_registry]
    nodes.sort(key=lambda item: (len(item["queries"]), item["similarity_score"]), reverse=True)
    nodes = nodes[: settings["max_nodes"]]
    allowed_node_keys = {node["key"] for node in nodes}
    edges = [
        edge for edge in edges
        if edge["source_key"] in allowed_node_keys and edge["target_key"] in allowed_node_keys
    ]

    seed_total = sum(1 for node in nodes if node.get("is_seed"))
    multi_query_nodes = sum(1 for node in nodes if len(node.get("queries", set())) > 1)

    return {
        "nodes": nodes,
        "edges": edges,
        "seed_counts": seed_counts,
        "total_queries": len(query_entries),
        "multi_query_nodes": multi_query_nodes,
        "seed_total": seed_total,
    }


def _merge_node_metadata(current: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer more specific metadata when the same node appears in multiple query graphs."""
    merged = dict(current)
    current_type = normalize_node_type(merged.get("node_type", "entity"))
    incoming_type = normalize_node_type(incoming.get("node_type", "entity"))

    if current_type == "entity" and incoming_type != "entity":
        merged["node_type"] = incoming_type
    elif current_type != incoming_type and incoming_type == "gene/protein":
        merged["node_type"] = incoming_type
    if not merged.get("node_id") and incoming.get("node_id"):
        merged["node_id"] = incoming["node_id"]
    if merged.get("node_source") in {"", "UNKNOWN"} and incoming.get("node_source"):
        merged["node_source"] = incoming["node_source"]
    return merged


def merge_network_payloads(payloads: List[Dict[str, Any]], max_nodes: int, max_edges: int) -> Dict[str, Any]:
    """Merge several independently-built query payloads into one compound network."""
    node_registry: Dict[str, Dict[str, Any]] = {}
    edge_registry: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    seed_counts: Dict[str, int] = {}

    for payload in payloads:
        for query, count in payload.get("seed_counts", {}).items():
            seed_counts[query] = count

        for node in payload.get("nodes", []):
            key = node.get("key")
            if not key:
                continue

            current = node_registry.get(key)
            if current:
                current["queries"].update(node.get("queries", set()))
                current["similarity_score"] = max(current.get("similarity_score", 0.0), node.get("similarity_score", 0.0))
                current["is_seed"] = current.get("is_seed", False) or node.get("is_seed", False)
                current["metadata"] = _merge_node_metadata(current.get("metadata", {}), node.get("metadata", {}))
            else:
                node_registry[key] = {
                    "key": key,
                    "name": node.get("name", ""),
                    "metadata": dict(node.get("metadata", {})),
                    "similarity_score": node.get("similarity_score", 0.0),
                    "queries": set(node.get("queries", set())),
                    "is_seed": bool(node.get("is_seed", False)),
                }

        for edge in payload.get("edges", []):
            source_key = edge.get("source_key")
            target_key = edge.get("target_key")
            relation = str(edge.get("relation", "related"))
            if not source_key or not target_key:
                continue

            edge_key = tuple(sorted([source_key, target_key])) + (relation,)
            current = edge_registry.get(edge_key)
            if current:
                current["score"] = max(current.get("score", 0.0), edge.get("score", 0.0))
                current["queries"].update(edge.get("queries", set()))
                current["provenance"].update(edge.get("provenance", set()))
            else:
                edge_registry[edge_key] = {
                    "source_key": source_key,
                    "target_key": target_key,
                    "score": edge.get("score", 0.0),
                    "relation": relation,
                    "provenance": set(edge.get("provenance", set())),
                    "queries": set(edge.get("queries", set())),
                }

    edges = list(edge_registry.values())
    edges.sort(key=lambda item: (len(item.get("queries", set())), item.get("score", 0.0)), reverse=True)
    edges = edges[:max_edges]

    connected_node_keys = {edge["source_key"] for edge in edges} | {edge["target_key"] for edge in edges}
    for node in node_registry.values():
        if node.get("is_seed"):
            connected_node_keys.add(node["key"])

    nodes = [node_registry[key] for key in connected_node_keys if key in node_registry]

    node_edge_counts: Dict[str, int] = {key: 0 for key in connected_node_keys}
    for edge in edges:
        if edge["source_key"] in node_edge_counts:
            node_edge_counts[edge["source_key"]] += 1
        if edge["target_key"] in node_edge_counts:
            node_edge_counts[edge["target_key"]] += 1

    nodes.sort(
        key=lambda item: (
            len(item.get("queries", set())),
            node_edge_counts.get(item["key"], 0),
            1 if item.get("is_seed") else 0,
            item.get("similarity_score", 0.0),
        ),
        reverse=True,
    )
    nodes = nodes[:max_nodes]
    allowed_node_keys = {node["key"] for node in nodes}
    edges = [
        edge for edge in edges
        if edge["source_key"] in allowed_node_keys and edge["target_key"] in allowed_node_keys
    ]

    multi_query_nodes = sum(1 for node in nodes if len(node.get("queries", set())) > 1)
    seed_total = sum(1 for node in nodes if node.get("is_seed"))

    return {
        "nodes": nodes,
        "edges": edges,
        "seed_counts": seed_counts,
        "total_queries": len(payloads),
        "multi_query_nodes": multi_query_nodes,
        "seed_total": seed_total,
    }


def get_query_network_settings() -> Dict[str, Any]:
    """Render configurable graph controls and return the active settings."""
    st.subheader("Network Controls")
    available_types, available_sources = collect_network_filter_options()

    col1, col2, col3 = st.columns(3)
    with col1:
        seed_limit = st.slider("Seed Nodes", min_value=3, max_value=25, value=10, help="Top retrieved nodes used as graph seeds.")
    with col2:
        neighbors_per_seed = st.slider("Neighbors / Seed", min_value=1, max_value=10, value=3, help="Real graph neighbors added per seed.")
    with col3:
        min_similarity = st.slider("Min Similarity %", min_value=0, max_value=100, value=20, help="Hide weak vector hits before graph expansion.")

    col4, col5 = st.columns(2)
    with col4:
        max_nodes = st.slider("Max Nodes", min_value=5, max_value=60, value=25)
    with col5:
        max_edges = st.slider("Max Edges", min_value=5, max_value=120, value=40)

    allowed_types = st.multiselect(
        "Node Types",
        options=available_types,
        default=available_types,
        help="Filter which node types may appear in the network.",
    )
    allowed_sources = st.multiselect(
        "Node Sources",
        options=available_sources,
        default=available_sources,
        help="Filter which data sources may appear in the network.",
    )

    return {
        "seed_limit": seed_limit,
        "neighbors_per_seed": neighbors_per_seed,
        "min_similarity": float(min_similarity),
        "max_nodes": max_nodes,
        "max_edges": max_edges,
        "allowed_types": set(allowed_types),
        "allowed_sources": set(allowed_sources),
    }

def initialize_database():
    """Initialize the vector database"""
    if ensure_database_ready(show_spinner=True):
        st.success("Vector database ready.")
        return True

    st.error(f"Error initializing database: {st.session_state.db_error}")
    return False


def split_batch_queries(raw_text: str) -> List[str]:
    """Support one query per line or one query per paragraph when users paste batches."""
    normalized = raw_text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    paragraph_queries = [chunk.strip() for chunk in re.split(r"\n\s*\n+", normalized) if chunk.strip()]
    if len(paragraph_queries) > 1:
        return paragraph_queries

    line_queries = [line.strip() for line in normalized.split("\n") if line.strip()]
    if len(line_queries) > 1 and all(line.endswith(("?", ".", ":")) for line in line_queries):
        return line_queries

    return [normalized]


def process_single_query(prompt: str) -> bool:
    """Run retrieval and answer generation for a single query turn."""
    st.session_state.chat_history.append(("user", prompt))

    try:
        search_results, query_variants = _search_query_seed_nodes(
            st.session_state.db_manager,
            prompt,
            st.session_state.search_result_limit,
        )

        if search_results:
            st.session_state.query_genes.append({
                "query": prompt,
                "genes": search_results,
                "query_variants": query_variants,
            })

        result = generate_enhanced_response(prompt, search_results, st.session_state.db_manager)
        st.session_state.chat_history.append(("assistant", result["response"]))

        assistant_index = len(st.session_state.chat_history) - 1
        message_key = f"message_{assistant_index}"
        if result.get("citations"):
            st.session_state.chat_citations[message_key] = result["citations"]
        if result.get("papers"):
            st.session_state.chat_papers[message_key] = result["papers"]
        if search_results:
            st.session_state[f"search_results_{assistant_index + 1}"] = search_results

        return True
    except Exception as exc:
        error_msg = f"Sorry, I encountered an error while searching: {str(exc)}"
        st.session_state.chat_history.append(("assistant", error_msg))
        return False

def generate_enhanced_response(query: str, search_results: List[Dict[str, Any]], db_manager: VectorDBManager) -> Dict[str, Any]:
    """Generate enhanced response using GPT-4 and PubMed citations"""
    if not search_results:
        return {
            'response': "I couldn't find any relevant information in the gene/protein database for your query.",
            'citations': [],
            'papers': [],
            'has_enhanced': False
        }
    
    # Try to get enhanced response with GPT-4 and citations
    try:
        enhanced_result = db_manager.generate_enhanced_response(query, search_results, max_tokens=1024)
        
        # Search for related papers
        papers = search_papers_for_query(query, n_results=3)
        
        return {
            'response': enhanced_result['gpt_response'],
            'citations': enhanced_result['citations'],
            'papers': papers,
            'has_enhanced': enhanced_result['has_openai'],
            'has_citations': enhanced_result['has_citations'],
            'search_results': search_results
        }
    except Exception as e:
        # Fallback to basic response if enhanced fails
        st.error(f"Enhanced response failed, using basic mode: {str(e)}")
        print(f"DEBUG: Exception in generate_enhanced_response: {str(e)}")
        return generate_basic_response(query, search_results)

def generate_basic_response(query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate basic response (fallback when GPT-4 is not available)"""
    if not search_results:
        return {
            'response': "I couldn't find any relevant information in the gene/protein database for your query.",
            'citations': [],
            'papers': [],
            'has_enhanced': False
        }
    
    # Create a comprehensive response
    response = f"Based on your query about '{query}', I found the following relevant gene/protein information:\n\n"
    
    for i, result in enumerate(search_results, 1):
        metadata = result['metadata']
        similarity = result.get('similarity_score', distance_to_similarity_percent(result.get('distance')))
        
        response += f"**{i}. {metadata['node_name']}**\n"
        response += f"   - **Gene/Protein ID**: {metadata['node_id']}\n"
        response += f"   - **Type**: {metadata['node_type']}\n"
        response += f"   - **Source**: {metadata['node_source']}\n"
        response += f"   - **Similarity Score**: {similarity:.1f}%\n\n"
    
    # Add contextual information
    if len(search_results) > 1:
        response += f"I found {len(search_results)} relevant matches. "
    
    # Add suggestions for further queries
    gene_names = [result['metadata']['node_name'] for result in search_results]
    response += f"\n**Related genes/proteins you might want to ask about**: {', '.join(gene_names[:3])}"
    
    # Search for related papers
    papers = search_papers_for_query(query, n_results=3)
    
    return {
        'response': response,
        'citations': [],
        'papers': papers,
        'has_enhanced': False,
        'has_citations': False,
        'search_results': search_results
    }

def display_citations(citations):
    """Display PubMed citations in a formatted way."""
    if not citations:
        return

    st.markdown("### Supporting Literature")

    for i, citation in enumerate(citations, 1):
        title = citation.get("title", "") if isinstance(citation, dict) else getattr(citation, "title", "")
        pmid = citation.get("pmid", "") if isinstance(citation, dict) else getattr(citation, "pmid", "")
        authors = citation.get("authors", "") if isinstance(citation, dict) else getattr(citation, "authors", "")
        journal = citation.get("journal", "") if isinstance(citation, dict) else getattr(citation, "journal", "")
        year = citation.get("year", "") if isinstance(citation, dict) else getattr(citation, "year", "")
        is_review = citation.get("is_review", False) if isinstance(citation, dict) else getattr(citation, "is_review", False)

        with st.container():
            if pmid:
                pmid_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                st.markdown(f"**{i}. [{title}]({pmid_link})**")
            else:
                st.markdown(f"**{i}. {title}**")

            if authors:
                st.write(f"**Authors:** {authors}")

            if journal and year:
                st.write(f"**Journal:** {journal} ({year})")
            elif journal:
                st.write(f"**Journal:** {journal}")
            elif year:
                st.write(f"**Year:** {year}")

            if pmid:
                st.write(f"**PMID:** {pmid}")

            if is_review:
                st.markdown("**Review Article**")

def create_gene_network_graph(genes_data: List[Dict], query: str = ""):
    """Create an interactive network graph of related genes using pyvis"""
    if not genes_data:
        return None
    
    # Create pyvis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.force_atlas_2based()
    
    # Color scheme for different node types
    color_scheme = {
        "gene": "#1f77b4",      # Blue
        "protein": "#2ca02c",   # Green
        "gene/protein": "#ff7f0e",  # Orange
        "default": "#9467bd"    # Purple
    }
    
    # Add nodes for each gene
    for gene in genes_data:
        metadata = gene['metadata']
        gene_name = metadata['node_name']
        gene_id = metadata['node_id']
        gene_type = metadata['node_type']
        gene_source = metadata['node_source']
        relevance = gene.get('similarity_score', distance_to_similarity_percent(gene.get('distance')))
        
        # Determine color based on type
        color = color_scheme.get(gene_type, color_scheme["default"])
        
        # Size based on relevance
        size = max(15, relevance / 2)
        
        # Enhanced tooltip
        title = (f"Gene: {gene_name}<br>"
                f"ID: {gene_id}<br>"
                f"Type: {gene_type}<br>"
                f"Source: {gene_source}<br>"
                f"Relevance: {relevance:.1f}%<br>"
                f"Query: {query}")
        
        net.add_node(
            gene_name,
            label=gene_name,
            color=color,
            title=title,
            size=size,
            font={'size': 12}
        )
    
    # Add edges between genes (connect all genes from same query)
    gene_names = [gene['metadata']['node_name'] for gene in genes_data]
    for i, gene1 in enumerate(gene_names):
        for gene2 in gene_names[i+1:]:
            # Edge width based on combined relevance
            gene1_relevance = (1 - genes_data[i].get('distance', 0)) * 100
            gene2_idx = next(j for j, g in enumerate(genes_data) if g['metadata']['node_name'] == gene2)
            gene2_relevance = (1 - genes_data[gene2_idx].get('distance', 0)) * 100
            
            edge_width = max(1, (gene1_relevance + gene2_relevance) / 50)
            
            net.add_edge(
                gene1, 
                gene2, 
                title=f"Co-occurrence in query: {query}",
                width=edge_width,
                color="#666666"
            )
    
    # Configure physics
    net.set_options("""
    var options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08,
                "damping": 0.4
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": true,
                "iterations": 1000
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "multiselect": true
        }
    }
    """)
    
    return net

def create_compound_gene_network_graph(selected_queries: List[str], query_genes_data: List[Dict]) -> Network:
    """Create a compound network graph combining genes from multiple queries"""
    if not selected_queries or not query_genes_data:
        return None
    
    # Create pyvis network
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    net.force_atlas_2based()
    
    # Color scheme for different queries and node types
    query_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange  
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf"   # Cyan
    ]
    
    # Track all genes and their query associations
    all_genes = {}  # gene_name -> {queries: set, data: gene_data, relevance: max_relevance}
    query_gene_map = {}  # query -> [gene_names]
    
    # Collect genes from selected queries
    for i, query in enumerate(selected_queries):
        query_gene_map[query] = []
        
        # Find genes for this query
        for query_data in query_genes_data:
            if isinstance(query_data, dict) and query_data.get('query') == query:
                genes = query_data.get('genes', [])
                
                for gene in genes:
                    metadata = gene['metadata']
                    gene_name = metadata['node_name']
                    relevance = (1 - gene.get('distance', 0)) * 100
                    
                    query_gene_map[query].append(gene_name)
                    
                    if gene_name not in all_genes:
                        all_genes[gene_name] = {
                            'queries': set(),
                            'data': gene,
                            'relevance': relevance
                        }
                    else:
                        # Update with higher relevance if found
                        all_genes[gene_name]['relevance'] = max(
                            all_genes[gene_name]['relevance'], 
                            relevance
                        )
                    
                    all_genes[gene_name]['queries'].add(query)
                break
    
    # Add nodes to the network
    for gene_name, gene_info in all_genes.items():
        metadata = gene_info['data']['metadata']
        relevance = gene_info['relevance']
        associated_queries = gene_info['queries']
        
        # Determine node color based on query association
        if len(associated_queries) > 1:
            # Multi-query gene - use a special color (gold)
            color = "#FFD700"
            border_color = "#FFA500"
            size = max(20, relevance / 2 + 10)  # Larger for multi-query genes
        else:
            # Single query gene - use query-specific color
            query = list(associated_queries)[0]
            query_index = selected_queries.index(query) % len(query_colors)
            color = query_colors[query_index]
            border_color = color
            size = max(15, relevance / 2)
        
        # Enhanced tooltip
        queries_text = "<br>".join([f"• {q}" for q in associated_queries])
        title = (f"Gene: {gene_name}<br>"
                f"ID: {metadata['node_id']}<br>"
                f"Type: {metadata['node_type']}<br>"
                f"Source: {metadata['node_source']}<br>"
                f"Max Relevance: {relevance:.1f}%<br>"
                f"Found in queries:<br>{queries_text}")
        
        net.add_node(
            gene_name,
            label=gene_name,
            color=color,
            title=title,
            size=size,
            font={'size': 12},
            borderWidth=2,
            borderWidthSelected=4,
            chosen={'node': True}
        )
    
    # Add edges
    # 1. Intra-query edges (genes from same query)
    for query, gene_names in query_gene_map.items():
        query_index = selected_queries.index(query) % len(query_colors)
        edge_color = query_colors[query_index]
        
        for i, gene1 in enumerate(gene_names):
            for gene2 in gene_names[i+1:]:
                if gene1 in all_genes and gene2 in all_genes:
                    # Edge width based on combined relevance
                    relevance1 = all_genes[gene1]['relevance']
                    relevance2 = all_genes[gene2]['relevance']
                    edge_width = max(1, (relevance1 + relevance2) / 50)
                    
                    net.add_edge(
                        gene1, 
                        gene2, 
                        title=f"Co-occurrence in: {query}",
                        width=edge_width,
                        color=edge_color,
                        alpha=0.7
                    )
    
    # 2. Inter-query edges (genes that appear in multiple queries)
    multi_query_genes = [name for name, info in all_genes.items() if len(info['queries']) > 1]
    for i, gene1 in enumerate(multi_query_genes):
        for gene2 in multi_query_genes[i+1:]:
            # Connect multi-query genes with special edges
            shared_queries = all_genes[gene1]['queries'].intersection(all_genes[gene2]['queries'])
            if shared_queries:
                net.add_edge(
                    gene1,
                    gene2,
                    title=f"Shared in: {', '.join(shared_queries)}",
                    width=3,
                    color="#FFD700",  # Gold for multi-query connections
                    dashes=True,
                    alpha=0.8
                )
    
    # Configure physics for better layout with more nodes
    net.set_options("""
    var options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.005,
                "springLength": 120,
                "springConstant": 0.05,
                "damping": 0.6
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": true,
                "iterations": 1500
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "multiselect": true,
            "selectConnectedEdges": false
        }
    }
    """)
    
    return net


def build_pyvis_network_graph(network_payload: Dict[str, Any], query_order: List[str], height_px: int) -> Optional[Network]:
    """Render a real graph network from retrieved seeds plus actual graph edges."""
    nodes = network_payload.get("nodes", [])
    if not nodes:
        return None

    edges = network_payload.get("edges", [])
    node_map = {node["key"]: node for node in nodes}
    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="black")
    net.force_atlas_2based()

    for node in nodes:
        metadata = node["metadata"]
        query_membership = sorted(node.get("queries", set()))
        node_type = normalize_node_type(metadata.get("node_type", "entity"))
        base_color = NODE_TYPE_COLORS.get(node_type, NODE_TYPE_COLORS["entity"])
        node_shape = NODE_TYPE_SHAPES.get(node_type, NODE_TYPE_SHAPES["entity"])

        if len(query_membership) > 1:
            border_color = "#FFD700"
        elif query_membership:
            border_color = QUERY_COLOR_PALETTE[query_order.index(query_membership[0]) % len(QUERY_COLOR_PALETTE)]
        else:
            border_color = "#444444"

        similarity = node.get("similarity_score", 0.0)
        size = max(16, min(40, 10 + (similarity / 6)))
        tooltip = (
            f"Name: {metadata.get('node_name')}<br>"
            f"ID: {metadata.get('node_id')}<br>"
            f"Type: {node_type}<br>"
            f"Source: {metadata.get('node_source')}<br>"
            f"Similarity: {similarity:.1f}%<br>"
            f"Role: {'Seed match' if node.get('is_seed') else 'Expanded neighbor'}<br>"
            f"Queries: {', '.join(query_membership) if query_membership else 'N/A'}"
        )

        net.add_node(
            node["key"],
            label=metadata.get("node_name", node["name"]),
            color={
                "background": base_color,
                "border": border_color,
                "highlight": {"background": base_color, "border": "#111111"},
                "hover": {"background": base_color, "border": "#111111"},
            },
            title=tooltip,
            size=size,
            shape=node_shape,
            font={"size": 12},
            borderWidth=3 if node.get("is_seed") else 1,
            borderWidthSelected=5,
            chosen={"node": True},
        )

    for edge in edges:
        source = node_map.get(edge["source_key"])
        target = node_map.get(edge["target_key"])
        if not source or not target:
            continue

        query_membership = sorted(edge.get("queries", set()))
        if len(query_membership) > 1:
            color = "#FFD700"
        elif query_membership:
            color = QUERY_COLOR_PALETTE[query_order.index(query_membership[0]) % len(QUERY_COLOR_PALETTE)]
        else:
            color = "#666666"

        provenance = ", ".join(sorted(edge.get("provenance", set())))
        tooltip = (
            f"{source['metadata']['node_name']} -> {target['metadata']['node_name']}<br>"
            f"Relation: {edge.get('relation', 'related')}<br>"
            f"Confidence/Score: {edge.get('score', 0.0):.2f}<br>"
            f"Source: {provenance}<br>"
            f"Queries: {', '.join(query_membership) if query_membership else 'N/A'}"
        )

        net.add_edge(
            edge["source_key"],
            edge["target_key"],
            title=tooltip,
            width=max(1.5, min(8.0, edge.get("score", 0.0) * 6.0)),
            color=color,
            dashes="PAPER_INGEST" in edge.get("provenance", set()),
            arrows="to",
        )

    net.set_options("""
    var options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -70,
                "centralGravity": 0.01,
                "springLength": 120,
                "springConstant": 0.06,
                "damping": 0.6
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": true,
                "iterations": 1200
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 150,
            "multiselect": true,
            "selectConnectedEdges": false
        }
    }
    """)

    return net


def render_network_legend() -> None:
    """Render a compact legend beside the graph."""
    st.markdown("**Legend**")
    st.markdown("**Node Types**")

    type_rows = [
        ("gene", "Gene"),
        ("protein", "Protein"),
        ("gene/protein", "Gene/Protein"),
        ("pathway", "Pathway"),
        ("disease", "Disease"),
    ]
    for node_type, label in type_rows:
        color = NODE_TYPE_COLORS.get(node_type, NODE_TYPE_COLORS["entity"])
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{color};border:1px solid #333;'></span>"
            f"<span>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("**Query Role**")
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
        "<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:#bbbbbb;border:3px solid #FFD700;'></span>"
        "<span>Appears in multiple selected queries</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
        "<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:#bbbbbb;border:3px solid #1f77b4;'></span>"
        "<span>Single-query node</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Node fill color shows biological type. Border color shows query membership.")


def render_network_with_legend(net: Network, height: int) -> None:
    """Render the graph with a right-side legend."""
    graph_col, legend_col = st.columns([5, 2])
    with graph_col:
        render_pyvis_network(net, height=height)
    with legend_col:
        render_network_legend()


def build_single_query_network_graph(query_entry: Dict[str, Any], settings: Dict[str, Any]) -> Tuple[Optional[Network], Dict[str, Any]]:
    """Create a single-query graph using configured seeds and real edge expansion."""
    payload = build_network_payload([query_entry], settings)
    graph = build_pyvis_network_graph(payload, [query_entry.get("query", "")], height_px=650)
    return graph, payload


def build_compound_query_network_graph(selected_queries: List[str], query_genes_data: List[Dict], settings: Dict[str, Any]) -> Tuple[Optional[Network], Dict[str, Any]]:
    """Create a multi-query graph by merging multiple expanded query subgraphs."""
    selected_entries = [
        query_data
        for query_data in query_genes_data
        if isinstance(query_data, dict) and query_data.get("query") in selected_queries
    ]
    per_query_payloads = [build_network_payload([entry], settings) for entry in selected_entries]
    payload = merge_network_payloads(per_query_payloads, settings["max_nodes"], settings["max_edges"])
    graph = build_pyvis_network_graph(payload, selected_queries, height_px=700)
    return graph, payload

def display_gene_network_tab():
    """Display the gene network visualization tab with compound network option"""
    st.header("🕸️ Gene Network Visualization")
    
    if not st.session_state.query_genes:
        st.info("🔍 No gene networks to display yet. Start by asking questions in the Chat tab to build gene relationships!")
        return
    
    # Extract query strings from the stored data
    query_options = []
    for query_data in st.session_state.query_genes:
        if isinstance(query_data, dict) and 'query' in query_data:
            query_options.append(query_data['query'])
    
    if not query_options:
        return
    
    # Network visualization mode selection
    st.subheader("🎛️ Network Visualization Mode")
    
    viz_mode = st.radio(
        "Choose visualization mode:",
        ["Single Query Network", "Compound Network (Multiple Queries)"],
        key="viz_mode_selector"
    )
    
    if viz_mode == "Single Query Network":
        # Original single query functionality
        st.subheader("Available Gene Networks")
        
        selected_query = st.selectbox(
            "Select a query to visualize its gene network:",
            options=query_options,
            key="single_network_query_selector"
        )
        
        if selected_query:
            # Find the corresponding genes data
            genes_data = None
            for query_data in st.session_state.query_genes:
                if isinstance(query_data, dict) and query_data.get('query') == selected_query:
                    genes_data = query_data.get('genes', [])
                    break
            
            if genes_data:
                # Create and display the network graph
                net = create_gene_network_graph(genes_data, selected_query)
                if net:
                    render_pyvis_network(net, height=650)
                    
                    # Show gene details below the graph
                    st.subheader("📋 Genes in Network")
                    
                    cols = st.columns(min(3, len(genes_data)))
                    for i, gene in enumerate(genes_data):
                        with cols[i % 3]:
                            metadata = gene['metadata']
                            relevance = gene.get('similarity_score', distance_to_similarity_percent(gene.get('distance')))
                            
                            st.markdown(f"""
                            **{metadata['node_name']}**
                            - ID: {metadata['node_id']}
                            - Type: {metadata['node_type']}
                            - Source: {metadata['node_source']}
                            - Relevance: {relevance:.1f}%
                            """)
                else:
                    st.error("Failed to create network graph")
            else:
                st.warning("No gene data found for selected query")
    
    else:  # Compound Network mode
        st.subheader("🔗 Compound Network Builder")
        
        # Multi-select for queries
        selected_queries = st.multiselect(
            "Select multiple queries to combine into one network:",
            options=query_options,
            default=query_options[:min(3, len(query_options))],  # Default to first 3 queries
            key="compound_network_query_selector",
            help="Select 2 or more queries to see how their gene networks interconnect"
        )
        
        if len(selected_queries) < 2:
            st.warning("⚠️ Please select at least 2 queries to create a compound network.")
        else:
            st.info(f"🔬 Creating compound network from {len(selected_queries)} queries...")
            
            # Create compound network
            compound_net = create_compound_gene_network_graph(selected_queries, st.session_state.query_genes)
            
            if compound_net:
                render_pyvis_network(compound_net, height=700)
                
                # Show legend and analysis
                st.subheader("🎨 Network Legend")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Node Colors:**")
                    st.markdown("🟡 **Gold**: Genes found in multiple queries")
                    st.markdown("🔵 **Colored**: Query-specific genes")
                    st.markdown("**Node Size**: Based on relevance score")
                
                with col2:
                    st.markdown("**Edge Types:**")
                    st.markdown("**Solid lines**: Genes from same query")
                    st.markdown("**Dashed gold lines**: Multi-query connections")
                    st.markdown("**Edge Width**: Based on combined relevance")
                
                # Compound network analysis
                st.subheader("🔍 Compound Network Analysis")
                
                # Collect analysis data
                all_genes = set()
                multi_query_genes = set()
                query_gene_counts = {}
                
                for query in selected_queries:
                    query_gene_counts[query] = 0
                    for query_data in st.session_state.query_genes:
                        if isinstance(query_data, dict) and query_data.get('query') == query:
                            genes = query_data.get('genes', [])
                            query_genes = set()
                            for gene in genes:
                                gene_name = gene['metadata']['node_name']
                                all_genes.add(gene_name)
                                query_genes.add(gene_name)
                            query_gene_counts[query] = len(query_genes)
                            
                            # Check for multi-query genes
                            for other_query in selected_queries:
                                if other_query != query:
                                    for other_query_data in st.session_state.query_genes:
                                        if isinstance(other_query_data, dict) and other_query_data.get('query') == other_query:
                                            other_genes = set(g['metadata']['node_name'] for g in other_query_data.get('genes', []))
                                            multi_query_genes.update(query_genes.intersection(other_genes))
                            break
                
                # Display analysis metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Unique Genes", len(all_genes))
                
                with col2:
                    st.metric("Multi-Query Genes", len(multi_query_genes))
                
                with col3:
                    overlap_percentage = (len(multi_query_genes) / len(all_genes) * 100) if all_genes else 0
                    st.metric("Overlap %", f"{overlap_percentage:.1f}%")
                
                with col4:
                    st.metric("Selected Queries", len(selected_queries))
                
                # Query breakdown
                if st.expander("📊 Query Breakdown", expanded=False):
                    for query, count in query_gene_counts.items():
                        st.write(f"**{query}**: {count} genes")
                
            else:
                st.error("Failed to create compound network graph")
    
    # Network statistics (common for both modes)
    if st.session_state.query_genes:
        st.markdown("---")
        st.subheader("📊 Overall Network Statistics")
        
        total_queries = len(st.session_state.query_genes)
        total_unique_genes = set()
        
        for query_data in st.session_state.query_genes:
            if isinstance(query_data, dict):
                genes = query_data.get('genes', [])
                for gene in genes:
                    total_unique_genes.add(gene['metadata']['node_name'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("Unique Genes", len(total_unique_genes))
        with col3:
            avg_genes = sum(len(q.get('genes', [])) for q in st.session_state.query_genes if isinstance(q, dict)) / max(1, total_queries)
            st.metric("Avg Genes/Query", f"{avg_genes:.1f}")


def display_gene_network_tab_v2():
    """Display the improved gene network tab using real graph expansion and filters."""
    st.header("Gene Network Visualization")

    if not st.session_state.query_genes:
        st.info("No gene networks to display yet. Ask a question in the Chat tab first.")
        return

    query_options = [
        query_data["query"]
        for query_data in st.session_state.query_genes
        if isinstance(query_data, dict) and query_data.get("query")
    ]
    if not query_options:
        return

    settings = get_query_network_settings()

    st.subheader("Visualization Mode")
    viz_mode = st.radio(
        "Choose visualization mode:",
        ["Single Query Network", "Compound Network (Multiple Queries)"],
        key="viz_mode_selector_v2",
    )

    if viz_mode == "Single Query Network":
        selected_query = st.selectbox(
            "Select a query to visualize:",
            options=query_options,
            key="single_network_query_selector_v2",
        )

        query_entry = next(
            (
                query_data
                for query_data in st.session_state.query_genes
                if isinstance(query_data, dict) and query_data.get("query") == selected_query
            ),
            None,
        )

        if query_entry:
            graph, payload = build_single_query_network_graph(query_entry, settings)
            if graph:
                render_network_with_legend(graph, height=650)

                stats_cols = st.columns(4)
                stats_cols[0].metric("Displayed Nodes", len(payload.get("nodes", [])))
                stats_cols[1].metric("Displayed Edges", len(payload.get("edges", [])))
                stats_cols[2].metric("Seed Matches", payload.get("seed_total", 0))
                stats_cols[3].metric("Retrieved Candidates", len(query_entry.get("genes", [])))

                rows = []
                for node in payload.get("nodes", []):
                    metadata = node["metadata"]
                    rows.append({
                        "Name": metadata.get("node_name", ""),
                        "Type": metadata.get("node_type", ""),
                        "Source": metadata.get("node_source", ""),
                        "Similarity %": f"{node.get('similarity_score', 0.0):.1f}",
                        "Role": "Seed" if node.get("is_seed") else "Neighbor",
                        "Queries": ", ".join(sorted(node.get("queries", set()))),
                    })
                if rows:
                    st.subheader("Nodes in Network")
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.warning("No graph could be built with the current filters. Relax the node/source/similarity filters.")

    else:
        selected_queries = st.multiselect(
            "Select multiple queries to combine:",
            options=query_options,
            default=query_options[:min(3, len(query_options))],
            key="compound_network_query_selector_v2",
            help="Each query contributes seed matches, then the graph expands through real curated or paper-derived edges.",
        )

        if len(selected_queries) < 2:
            st.warning("Please select at least 2 queries to create a compound network.")
        else:
            graph, payload = build_compound_query_network_graph(selected_queries, st.session_state.query_genes, settings)
            if graph:
                render_network_with_legend(graph, height=700)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Displayed Nodes", len(payload.get("nodes", [])))
                col2.metric("Displayed Edges", len(payload.get("edges", [])))
                col3.metric("Multi-Query Nodes", payload.get("multi_query_nodes", 0))
                overlap_percentage = payload.get("multi_query_nodes", 0) / max(1, len(payload.get("nodes", []))) * 100
                col4.metric("Overlap %", f"{overlap_percentage:.1f}%")

                with st.expander("Query Breakdown", expanded=False):
                    for query in selected_queries:
                        st.write(f"**{query}**: {payload.get('seed_counts', {}).get(query, 0)} seed matches kept")
            else:
                st.warning("No compound graph could be built with the current filters.")

    st.markdown("---")
    st.subheader("Overall Network Statistics")
    deduped_query_nodes = set()
    total_queries = len(st.session_state.query_genes)
    total_candidates = 0

    for query_data in st.session_state.query_genes:
        if not isinstance(query_data, dict):
            continue
        merged = merge_search_results(query_data.get("genes", []))
        total_candidates += len(merged)
        for gene in merged:
            deduped_query_nodes.add(normalize_node_key(gene.get("metadata", {}).get("node_name", "")))

    col1, col2, col3 = st.columns(3)
    col1.metric("Stored Queries", total_queries)
    col2.metric("Unique Retrieved Nodes", len(deduped_query_nodes))
    avg_candidates = total_candidates / max(1, total_queries)
    col3.metric("Avg Candidates/Query", f"{avg_candidates:.1f}")

def search_papers_for_query(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Search for relevant papers using vector similarity"""
    if not PAPERS_AVAILABLE or not st.session_state.db_manager:
        return []
    
    try:
        papers = st.session_state.db_manager.search_papers(query, n_results=n_results)
        return papers if papers else []
    except Exception as e:
        st.warning(f"Could not search papers: {str(e)}")
        return []

def display_papers(papers: List[Dict[str, Any]]):
    """Display paper results in a formatted way"""
    if not papers:
        return
    
    st.markdown("### 📄 Related Research Papers")
    
    for i, paper in enumerate(papers, 1):
        with st.container():
            paper_fields = get_paper_display_fields(paper)
            # Paper title and PMID link
            title = paper_fields.get('title') or 'Untitled'
            pmid = paper_fields.get('pmid', '')
            
            if pmid:
                pmid_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                st.markdown(f"**{i}. [{title}]({pmid_link})**")
            else:
                st.markdown(f"**{i}. {title}**")
            
            # Authors
            if paper_fields.get('authors'):
                st.write(f"👥 **Authors:** {', '.join(paper_fields['authors'])}")
            
            # Publication info
            info_parts = []
            if paper_fields.get('publication_year'):
                info_parts.append(f"({paper_fields['publication_year']})")
            if paper_fields.get('journal'):
                info_parts.append(f"*{paper_fields['journal']}*")
            
            if info_parts:
                st.write(" ".join(info_parts))
            
            # Abstract
            if paper_fields.get('abstract'):
                with st.expander("📖 Abstract"):
                    abstract = paper_fields['abstract']
                    st.write(abstract[:300] + "..." if len(abstract) > 300 else abstract)
            
            st.divider()


def get_chat_turns() -> List[Dict[str, Any]]:
    """Group flat chat history into user/assistant turns for compact rendering."""
    turns: List[Dict[str, Any]] = []
    pending_user: Optional[Dict[str, Any]] = None

    for index, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            pending_user = {
                "user_message": message,
                "user_index": index,
            }
            continue

        if role == "assistant":
            pending_user = pending_user or {"user_message": "", "user_index": None}
            turns.append(
                {
                    "user_message": pending_user["user_message"],
                    "user_index": pending_user["user_index"],
                    "assistant_message": message,
                    "assistant_index": index,
                }
            )
            pending_user = None

    if pending_user:
        turns.append(
            {
                "user_message": pending_user["user_message"],
                "user_index": pending_user["user_index"],
                "assistant_message": "",
                "assistant_index": None,
            }
        )

    return turns


def render_assistant_message_content(assistant_message: str, assistant_index: Optional[int]) -> None:
    """Render assistant content plus any stored citations and related papers."""
    st.write(assistant_message)

    if assistant_index is None:
        return

    message_key = f"message_{assistant_index}"
    if message_key in st.session_state.chat_citations:
        display_citations(st.session_state.chat_citations[message_key])
    if message_key in st.session_state.chat_papers:
        display_papers(st.session_state.chat_papers[message_key])
    render_stored_search_results(assistant_index)


def render_chat_turn(turn: Dict[str, Any]) -> None:
    """Render one compact user/assistant chat exchange."""
    st.chat_message("user").write(turn.get("user_message", ""))

    assistant_message = turn.get("assistant_message", "")
    if assistant_message:
        with st.chat_message("assistant"):
            render_assistant_message_content(assistant_message, turn.get("assistant_index"))


def render_chat_turns(turns: List[Dict[str, Any]]) -> None:
    """Render several chat turns in order."""
    for turn in turns:
        render_chat_turn(turn)


def format_chat_turn_label(turn: Dict[str, Any], turn_number: int) -> str:
    """Create a compact label for the query selector."""
    query = str(turn.get("user_message", "")).strip() or f"Query {turn_number}"
    if len(query) > 90:
        query = f"{query[:90]}..."
    return f"{turn_number}. {query}"


def render_stored_search_results(assistant_index: Optional[int]) -> None:
    """Render stored vector-search details for a previously answered query."""
    if assistant_index is None:
        return

    current_query_key = f"search_results_{assistant_index + 1}"
    stored_results = st.session_state.get(current_query_key, [])
    if not stored_results:
        return

    with st.expander("Retrieved Genes/Proteins from Database", expanded=False):
        node_options = []
        for result_item in stored_results:
            metadata = result_item['metadata']
            relevance = result_item.get('similarity_score', distance_to_similarity_percent(result_item.get('distance')))
            option_text = f"{metadata['node_name']} (ID: {metadata['node_id']}) - {relevance:.1f}% relevance"
            node_options.append(option_text)

        if node_options:
            session_id = st.session_state.get("browser_session_id", "default")
            selector_key = f"node_selector_{session_id}_{assistant_index}"
            selected_node = st.selectbox(
                "Select a gene/protein to view details:",
                options=["Select a gene/protein..."] + node_options,
                key=selector_key
            )

            if selected_node != "Select a gene/protein...":
                selected_index = node_options.index(selected_node)
                selected_result = stored_results[selected_index]
                metadata = selected_result['metadata']

                with st.container():
                    st.markdown("### Gene/Protein Details")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Name:** {metadata['node_name']}")
                        st.write(f"**Node ID:** {metadata['node_id']}")
                        st.write(f"**Type:** {metadata['node_type']}")

                    with col2:
                        st.write(f"**Source:** {metadata['node_source']}")
                        relevance = selected_result.get('similarity_score', distance_to_similarity_percent(selected_result.get('distance')))
                        st.write(f"**Relevance:** {relevance:.1f}%")

                    st.write(f"**Full Document:** {selected_result['document']}")
                    st.info("This information is retrieved from your local gene/protein database.")

    with st.expander("Raw Search Results (Technical)", expanded=False):
        for i, result_item in enumerate(stored_results, 1):
            st.write(f"**Result {i}:**")
            st.write(f"Document: {result_item['document']}")
            st.write(f"Similarity Score: {result_item.get('similarity_score', distance_to_similarity_percent(result_item.get('distance'))):.2f}%")
            st.json(result_item['metadata'])
            st.markdown("---")

def main():
    ensure_database_ready()

    st.title("Gene/Protein Knowledge Chat")
    st.markdown("Ask questions about genes and proteins from your dataset!")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Chat", "Gene Network"])
    
    # Sidebar
    with st.sidebar:
        st.header("Database Information")
        
        if not st.session_state.db_loaded:
            if st.session_state.db_error:
                st.error(f"Database load failed: {st.session_state.db_error}")
            if st.button("Retry Database Load", type="primary"):
                initialize_database()
        else:
            st.success("Database Ready")
            
            st.subheader("Search Settings")
            st.session_state.search_result_limit = st.slider(
                "Query results to retrieve",
                min_value=5,
                max_value=50,
                value=st.session_state.search_result_limit,
                help="Controls how many vector-search matches are retrieved for chat responses and network seeds.",
            )
            persist_chat_state()

            # Show database statistics
            if st.session_state.db_manager:
                try:
                    stats = st.session_state.db_manager.get_database_stats()
                    st.metric("Total Genes/Proteins", stats['total_documents'])
                    
                    st.subheader("Data Sources")
                    for source, count in stats['sources'].items():
                        st.write(f"• {source}: {count}")
                    
                    st.subheader("Node Types")
                    for node_type, count in stats['node_types'].items():
                        st.write(f"• {node_type}: {count}")
                        
                except Exception as e:
                    st.error(f"Error loading stats: {str(e)}")
        
        st.markdown("---")
        st.subheader("Sample Queries")
        st.markdown("""
        Try asking:
        - "Tell me about MYC protein"
        - "What genes are related to kinase?"
        - "Show me information about BRCA2"
        - "Find proteins involved in transcription"
        - "What is EGFR?"
        """)
        
        if st.button("Clear Chat History"):
            reset_chat_state()
            st.rerun()
    
    # Tab 1: Chat Interface
    with tab1:
        # Main chat interface
        if not st.session_state.db_loaded:
            st.info("The database is loading. If it failed, use the sidebar retry button.")
        else:
            chat_turns = get_chat_turns()
            if chat_turns:
                max_turn_index = len(chat_turns) - 1
                st.session_state.selected_chat_turn_index = min(
                    st.session_state.selected_chat_turn_index,
                    max_turn_index,
                )
            else:
                st.session_state.selected_chat_turn_index = 0

            control_col1, control_col2, control_col3 = st.columns([1.5, 3.2, 0.8])

            with control_col1:
                if len(chat_turns) > 1:
                    turn_options = list(range(len(chat_turns)))
                    st.session_state.selected_chat_turn_index = st.selectbox(
                        "Answered Queries",
                        options=turn_options,
                        index=min(st.session_state.selected_chat_turn_index, len(turn_options) - 1),
                        format_func=lambda idx: format_chat_turn_label(chat_turns[idx], idx + 1),
                        key="active_chat_turn_selector",
                        on_change=persist_chat_state,
                    )
                elif len(chat_turns) == 1:
                    st.caption("Viewing the current answered query")
                else:
                    st.caption("No answered queries yet")

            with control_col2:
                st.text_area(
                    "Ask next query",
                    key="next_query_input",
                    height=120,
                    placeholder="Ask about genes, proteins, or related information. Use one line per query or separate longer queries with a blank line.",
                    label_visibility="collapsed",
                    on_change=persist_chat_state,
                )

            with control_col3:
                st.caption(" ")
                ask_clicked = st.button("Ask", key="ask_next_query_button")

            batch_status_placeholder = st.empty()
            batch_results_placeholder = st.empty()

            if ask_clicked:
                prompt = st.session_state.next_query_input.strip()
                if not prompt:
                    st.warning("Enter a query before asking.")
                else:
                    prompts = split_batch_queries(prompt)
                    initial_turn_count = len(get_chat_turns())

                    for query_index, current_prompt in enumerate(prompts, start=1):
                        if len(prompts) > 1:
                            batch_status_placeholder.info(
                                f"Processing query {query_index} of {len(prompts)}..."
                            )
                        else:
                            batch_status_placeholder.info("Generating response...")

                        process_single_query(current_prompt)
                        st.session_state.selected_chat_turn_index = max(0, len(get_chat_turns()) - 1)
                        persist_chat_state()

                        completed_turns = get_chat_turns()[initial_turn_count:]
                        with batch_results_placeholder.container():
                            if len(prompts) > 1:
                                st.caption(
                                    f"Completed {query_index} of {len(prompts)} queries. New answers appear below as they finish."
                                )
                            render_chat_turns(completed_turns)

                    if len(prompts) > 1:
                        batch_status_placeholder.success(f"Completed {len(prompts)} queries.")
                    else:
                        batch_status_placeholder.empty()

                    st.session_state.selected_chat_turn_index = max(0, len(get_chat_turns()) - 1)
                    schedule_next_query_input_clear()
                    persist_chat_state()
                    st.rerun()

                    st.session_state.chat_history.append(("user", prompt))

                    with st.spinner("Searching database and generating enhanced response..."):
                        try:
                            # Search the vector database with entity-aware query expansion.
                            search_results, query_variants = _search_query_seed_nodes(
                                st.session_state.db_manager,
                                prompt,
                                st.session_state.search_result_limit,
                            )
                            
                            # Store query and genes for network visualization
                            if search_results:
                                st.session_state.query_genes.append({
                                    'query': prompt,
                                    'genes': search_results,
                                    'query_variants': query_variants,
                                })
                            
                            # Generate enhanced response with GPT-4 and citations
                            result = generate_enhanced_response(prompt, search_results, st.session_state.db_manager)
                            st.session_state.chat_history.append(("assistant", result['response']))

                            assistant_index = len(st.session_state.chat_history) - 1
                            message_key = f"message_{assistant_index}"
                            if result.get('citations'):
                                st.session_state.chat_citations[message_key] = result['citations']
                            if result.get('papers'):
                                st.session_state.chat_papers[message_key] = result['papers']
                            if search_results:
                                st.session_state[f"search_results_{assistant_index + 1}"] = search_results

                            st.session_state.selected_chat_turn_index = len(get_chat_turns()) - 1
                            schedule_next_query_input_clear()
                            st.rerun()
                            
                            # Display citations if available
                            if result.get('citations'):
                                display_citations(result['citations'])
                            
                            # Display papers if available
                            if result.get('papers'):
                                display_papers(result['papers'])
                            
                            # Add status indicators
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                provider = str(result.get('llm_provider', os.getenv('LLM_PROVIDER', 'openai'))).upper()
                                provider = str(result.get('llm_provider', os.getenv('LLM_PROVIDER', 'openai'))).upper()
                                if result.get('has_enhanced'):
                                    st.success(f"🤖 {provider} Enhanced")
                                    st.success(f"🤖 {provider} Enhanced")
                                else:
                                    st.info("📝 Basic Mode")
                            
                            with col2:
                                if result.get('has_citations') and result.get('citations'):
                                    st.success(f"📚 {len(result['citations'])} Citations")
                                elif result.get('has_citations'):
                                    st.warning("📚 No Citations Found")
                                else:
                                    st.info("📚 Citations Disabled")
                            
                            with col3:
                                # Show API key status
                                llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
                                if llm_provider == "gemini":
                                    llm_key = "✅" if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "❌"
                                    llm_label = "Gemini"
                                else:
                                    llm_key = "✅" if os.getenv("OPENAI_API_KEY") else "❌"
                                    llm_label = "OpenAI"
                                llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
                                if llm_provider == "gemini":
                                    llm_key = "✅" if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "❌"
                                    llm_label = "Gemini"
                                else:
                                    llm_key = "✅" if os.getenv("OPENAI_API_KEY") else "❌"
                                    llm_label = "OpenAI"
                                ncbi_key = "✅" if os.getenv("NCBI_API_KEY") else "❌"
                                st.info(f"🔑 {llm_label}: {llm_key} NCBI: {ncbi_key}")
                                st.info(f"🔑 {llm_label}: {llm_key} NCBI: {ncbi_key}")
                            
                            # Add to chat history (just the main response for cleaner history)
                            st.session_state.chat_history.append(("assistant", result['response']))
                            
                            # Store citations and papers for this message
                            message_key = f"message_{len(st.session_state.chat_history) - 1}"
                            if result.get('citations'):
                                st.session_state.chat_citations[message_key] = result['citations']
                            if result.get('papers'):
                                st.session_state.chat_papers[message_key] = result['papers']
                            
                            # Show retrieved nodes in a dropdown
                            if search_results:
                                with st.expander("🧬 Retrieved Genes/Proteins from Database", expanded=False):
                                    # Store search results in session state to persist across reruns
                                    current_query_key = f"search_results_{len(st.session_state.chat_history)}"
                                    if current_query_key not in st.session_state:
                                        st.session_state[current_query_key] = search_results
                                    
                                    # Create a selectbox for the retrieved nodes
                                    node_options = []
                                    stored_results = st.session_state[current_query_key]
                                    
                                    for i, result_item in enumerate(stored_results):
                                        metadata = result_item['metadata']
                                        relevance = result_item.get('similarity_score', distance_to_similarity_percent(result_item.get('distance')))
                                        option_text = f"{metadata['node_name']} (ID: {metadata['node_id']}) - {relevance:.1f}% relevance"
                                        node_options.append(option_text)
                                    
                                    if node_options:
                                        # Use a unique key that persists
                                        selector_key = f"node_selector_{current_query_key}"
                                        
                                        selected_node = st.selectbox(
                                            "Select a gene/protein to view details:",
                                            options=["Select a gene/protein..."] + node_options,
                                            key=selector_key
                                        )
                                        
                                        if selected_node != "Select a gene/protein...":
                                            # Find the selected result
                                            selected_index = node_options.index(selected_node)
                                            selected_result = stored_results[selected_index]
                                            metadata = selected_result['metadata']
                                            
                                            # Display detailed information in a container that won't cause rerun
                                            with st.container():
                                                st.markdown("### 📋 Gene/Protein Details")
                                                col1, col2 = st.columns(2)
                                                
                                                with col1:
                                                    st.write(f"**Name:** {metadata['node_name']}")
                                                    st.write(f"**Node ID:** {metadata['node_id']}")
                                                    st.write(f"**Type:** {metadata['node_type']}")
                                                
                                                with col2:
                                                    st.write(f"**Source:** {metadata['node_source']}")
                                                    relevance = selected_result.get('similarity_score', distance_to_similarity_percent(selected_result.get('distance')))
                                                    st.write(f"**Relevance:** {relevance:.1f}%")
                                                
                                                st.write(f"**Full Document:** {selected_result['document']}")
                                                
                                                # Add a small note
                                                st.info("💡 This information is retrieved from your local gene/protein database.")
                                
                                # Also show detailed search results in a separate expander
                                with st.expander("🔍 Raw Search Results (Technical)", expanded=False):
                                    for i, result_item in enumerate(search_results, 1):
                                        st.write(f"**Result {i}:**")
                                        st.write(f"Document: {result_item['document']}")
                                        st.write(f"Similarity Score: {result_item.get('similarity_score', distance_to_similarity_percent(result_item.get('distance'))):.2f}%")
                                        st.json(result_item['metadata'])
                                        st.markdown("---")
                        
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error while searching: {str(e)}"
                            st.session_state.chat_history.append(("assistant", error_msg))
                            st.session_state.selected_chat_turn_index = len(get_chat_turns()) - 1
                            schedule_next_query_input_clear()
                            st.rerun()

            st.markdown("---")

            if chat_turns:
                selected_turn = chat_turns[st.session_state.selected_chat_turn_index]
                render_chat_turn(selected_turn)
            else:
                st.info("Ask a question to start the conversation.")

            # Additional features
            st.markdown("---")
            
            # Quick actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🎲 Random Gene Info"):
                    if st.session_state.db_manager:
                        try:
                            gene_names = st.session_state.db_manager.get_all_gene_names()
                            import random
                            random_gene = random.choice(gene_names)
                            gene_info = st.session_state.db_manager.get_gene_info(random_gene)
                            if gene_info:
                                st.info(f"**Random Gene**: {gene_info['document']}")
                        except Exception as e:
                            st.error(f"Error getting random gene: {str(e)}")
            
            with col2:
                if st.button("📊 Database Stats"):
                    if st.session_state.db_manager:
                        try:
                            stats = st.session_state.db_manager.get_database_stats()
                            st.json(stats)
                        except Exception as e:
                            st.error(f"Error loading stats: {str(e)}")
            
            with col3:
                if st.button("📋 All Gene Names"):
                    if st.session_state.db_manager:
                        try:
                            gene_names = st.session_state.db_manager.get_all_gene_names()
                            st.write(f"**Total Genes**: {len(gene_names)}")
                            st.write(", ".join(gene_names[:20]) + ("..." if len(gene_names) > 20 else ""))
                        except Exception as e:
                            st.error(f"Error loading gene names: {str(e)}")
    
    # Tab 2: Gene Network Visualization
    with tab2:
        display_gene_network_tab_v2()

if __name__ == "__main__":
    main()
