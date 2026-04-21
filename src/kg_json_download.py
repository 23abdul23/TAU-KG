import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

try:
    import deb_data_papers as papers_db

    PAPERS_AVAILABLE = True
except ImportError:
    PAPERS_AVAILABLE = False
from typing import List, Dict, Any, Optional, Set, Tuple

def kg_slugify(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("/", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def kg_normalize_node_type(container_key: str, entity_type: str) -> str:
    if container_key:
        key = container_key.strip().lower()
        if key in {"genes", "proteins"}:
            return "gene/protein"
        if key == "diseases":
            return "disease"
        if key == "pathways":
            return "pathway"

    raw = (entity_type or "").strip().lower()
    if "gene" in raw or "protein" in raw:
        return "gene/protein"
    if "disease" in raw:
        return "disease"
    if "pathway" in raw:
        return "pathway"
    return "entity"


def kg_key_prefix_from_node_type(node_type: str) -> str:
    prefix_map = {
        "gene/protein": "gene_protein",
        "disease": "disease",
        "pathway": "pathway",
        "entity": "entity",
    }
    return prefix_map.get(node_type, "entity")


def kg_build_node_key(entity_id: str, label: str, node_type: str) -> str:
    base = kg_slugify(entity_id) if entity_id else kg_slugify(label)
    return f"{kg_key_prefix_from_node_type(node_type)}_{base}"


def build_kg_graph_from_store(data: Dict[str, Any]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    entity_id_to_node_key: Dict[str, str] = {}
    existing_node_keys: Set[str] = set()

    def upsert_node(entity: Dict[str, Any], container_key: str = "") -> str:
        entity_id = str(entity.get("id", "") or "")
        label = str(entity.get("name", "") or entity_id or "Unknown")
        node_type = kg_normalize_node_type(container_key, str(entity.get("entity_type", "") or ""))

        if entity_id and entity_id in entity_id_to_node_key:
            return entity_id_to_node_key[entity_id]

        node_key = kg_build_node_key(entity_id, label, node_type)
        if node_key in existing_node_keys:
            suffix = 2
            candidate = f"{node_key}_{suffix}"
            while candidate in existing_node_keys:
                suffix += 1
                candidate = f"{node_key}_{suffix}"
            node_key = candidate

        node_attrs: Dict[str, Any] = {
            "label": label,
            "nodeType": node_type,
        }
        chromosome = entity.get("chromosome", "")
        if chromosome:
            node_attrs["chromosome"] = chromosome

        nodes.append({"key": node_key, "attributes": node_attrs})
        existing_node_keys.add(node_key)
        if entity_id:
            entity_id_to_node_key[entity_id] = node_key
        return node_key

    paper_entities = data.get("paper_entities", {})
    if isinstance(paper_entities, dict):
        for _, bundle in paper_entities.items():
            if not isinstance(bundle, dict):
                continue
            for container_key in ("genes", "proteins", "diseases", "pathways"):
                entries = bundle.get(container_key, [])
                if not isinstance(entries, list):
                    continue
                for entity in entries:
                    if isinstance(entity, dict):
                        upsert_node(entity, container_key=container_key)

    for index, rel in enumerate(data.get("paper_edges", []), start=1):
        if not isinstance(rel, dict):
            continue

        source_id = str(rel.get("source_id", "") or "")
        target_id = str(rel.get("target_id", "") or "")

        source_key = entity_id_to_node_key.get(source_id)
        if not source_key:
            source_key = upsert_node(
                {
                    "id": source_id,
                    "name": rel.get("source_name", source_id),
                    "entity_type": rel.get("source_type", ""),
                    "paper_id": rel.get("paper_id", ""),
                }
            )

        target_key = entity_id_to_node_key.get(target_id)
        if not target_key:
            target_key = upsert_node(
                {
                    "id": target_id,
                    "name": rel.get("target_name", target_id),
                    "entity_type": rel.get("target_type", ""),
                    "paper_id": rel.get("paper_id", ""),
                }
            )

        edge_key = str(rel.get("id", "") or f"e{index}")
        edges.append(
            {
                "source": source_key,
                "target": target_key,
                "key": edge_key,
                "attributes": {
                    "edgeType": rel.get("edge_type", "OTHER"),
                    "original_relation": rel.get("original_relation", ""),
                    "edge_weight": rel.get("edge_weight", None),
                },
            }
        )

    return {
        "attributes": {
            "name": "Bio Knowledge Graph from papers",
            "organism": "Homo sapiens",
            "version": f"v{data.get('version', 1)}",
        },
        "options": {
            "type": "mixed",
            "multi": True,
            "allowSelfLoops": True,
        },
        "nodes": nodes,
        "edges": edges,
    }


def build_kg_export_from_current_store() -> Dict[str, Any]:
    store_payload: Dict[str, Any] = {}

    if PAPERS_AVAILABLE:
        try:
            loaded = papers_db.load_store()
            if isinstance(loaded, dict):
                store_payload = loaded
        except Exception:
            store_payload = {}

    if not store_payload:
        store_path = Path("data") / "paper_store.json"
        if store_path.exists():
            try:
                store_payload = json.loads(store_path.read_text(encoding="utf-8"))
            except Exception:
                store_payload = {}

    if not store_payload:
        store_payload = {
            "paper_entities": getattr(papers_db, "paper_entities", {}) if PAPERS_AVAILABLE else {},
            "paper_edges": getattr(papers_db, "paper_edges", []) if PAPERS_AVAILABLE else [],
            "version": 1,
        }

    return build_kg_graph_from_store(store_payload)


__all__ = [
    "build_kg_export_from_current_store",
    "build_kg_graph_from_store",
    "kg_build_node_key",
    "kg_key_prefix_from_node_type",
    "kg_normalize_node_type",
    "kg_slugify",
]

