import re
from typing import Any, Dict, Iterable, List, Optional


RELATION_EDGE_TYPE_RULES = [
    ("treat", "TREATS"),
    ("therapy", "TREATS"),
    ("therapeutic", "TREATS"),
    ("drug", "TREATS"),
    ("encode", "ENCODES"),
    ("express", "EXPRESSES"),
    ("participat", "PARTICIPATES"),
    ("component of", "PARTICIPATES"),
    ("pathway", "PARTICIPATES"),
    ("activate", "ACTIVATES"),
    ("stimulat", "ACTIVATES"),
    ("promote", "ACTIVATES"),
    ("upregulat", "ACTIVATES"),
    ("inhibit", "INHIBITS"),
    ("suppress", "INHIBITS"),
    ("block", "INHIBITS"),
    ("antagon", "INHIBITS"),
    ("regulat", "REGULATES"),
    ("control", "REGULATES"),
    ("modulat", "REGULATES"),
    ("mediate", "REGULATES"),
    ("interact", "INTERACTS"),
    ("bind", "INTERACTS"),
    ("compete", "INTERACTS"),
]

ENTITY_TYPE_CANONICAL = {
    "gene": "gene",
    "genes": "gene",
    "protein": "protein",
    "proteins": "protein",
    "disease": "disease",
    "diseases": "disease",
    "pathway": "pathway",
    "pathways": "pathway",
    "drug": "drug",
    "drugs": "drug",
    "tissue": "tissue",
    "tissues": "tissue",
    "entity": "entity",
    "paper": "paper",
}

ENTITY_TYPE_TITLE = {
    "gene": "Gene",
    "protein": "Protein",
    "disease": "Disease",
    "pathway": "Pathway",
    "drug": "Drug",
    "tissue": "Tissue",
    "entity": "Entity",
    "paper": "Paper",
}

ENTITY_TYPE_BUCKETS = {
    "gene": "genes",
    "protein": "proteins",
    "disease": "diseases",
    "pathway": "pathways",
}


def slugify_identifier(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return normalized or "unknown"


def normalize_entity_type(entity_type: Any, title_case: bool = False) -> str:
    normalized = ENTITY_TYPE_CANONICAL.get(str(entity_type or "").strip().lower(), "entity")
    return ENTITY_TYPE_TITLE.get(normalized, "Entity") if title_case else normalized


def bucket_from_entity_type(entity_type: Any) -> Optional[str]:
    return ENTITY_TYPE_BUCKETS.get(normalize_entity_type(entity_type))


def generate_entity_id(name: Any, entity_type: Any) -> str:
    return f"{normalize_entity_type(entity_type)}_{slugify_identifier(name)}"


def extract_chromosome(text: Any) -> str:
    content = str(text or "").strip()
    if not content:
        return ""

    match = re.search(r"\bchromosome\s+([0-9]{1,2}|[xX]|[yY]|[mM]|mt)\b", content)
    if match:
        return match.group(1).upper()

    match = re.search(r"\bchr(?:omosome)?\s*([0-9]{1,2}|[xX]|[yY]|[mM]|mt)\b", content)
    if match:
        return match.group(1).upper()

    return ""


def canonicalize_edge_type(raw_relation: Any, source_type: Any = "", target_type: Any = "") -> str:
    relation = str(raw_relation or "").strip().lower()
    if not relation:
        return "ASSOCIATES"

    for pattern, edge_type in RELATION_EDGE_TYPE_RULES:
        if pattern in relation:
            return edge_type

    normalized_source = normalize_entity_type(source_type)
    normalized_target = normalize_entity_type(target_type)
    if normalized_source == "gene" and normalized_target == "protein":
        return "ENCODES"
    if normalized_target == "pathway":
        return "PARTICIPATES"
    return "ASSOCIATES"


def build_entity_lookup(entities_by_type: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for entity_type, entities in (entities_by_type or {}).items():
        canonical_type = normalize_entity_type(entity_type, title_case=True)
        for entity in entities or []:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            lookup[name.lower()] = {
                "name": name,
                "type": canonical_type,
                "id": str(entity.get("id", "")).strip() or generate_entity_id(name, canonical_type),
                "chromosome": str(entity.get("chromosome", "")).strip(),
            }
    return lookup


def infer_relationship_entity_type(
    entity_name: Any,
    entity_type: Any,
    entity_lookup: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    explicit_type = normalize_entity_type(entity_type, title_case=True)
    if explicit_type != "Entity":
        return explicit_type

    lookup = entity_lookup or {}
    record = lookup.get(str(entity_name or "").strip().lower())
    if record:
        return record.get("type", "Entity")
    return "Entity"


def infer_relationship_entity_id(
    entity_name: Any,
    entity_id: Any,
    entity_type: Any,
    entity_lookup: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    explicit_id = str(entity_id or "").strip()
    if explicit_id:
        return explicit_id

    lookup = entity_lookup or {}
    record = lookup.get(str(entity_name or "").strip().lower())
    if record and record.get("id"):
        return record["id"]

    return generate_entity_id(entity_name, entity_type)


def normalize_paper_entity(
    entity: Dict[str, Any],
    entity_type: str,
    paper_id: str,
    entity_index: int,
) -> Dict[str, Any]:
    name = str(entity.get("name", "")).strip()
    canonical_type = normalize_entity_type(entity_type)
    chromosome = str(entity.get("chromosome", "")).strip()
    if canonical_type == "gene" and not chromosome:
        chromosome = extract_chromosome(entity.get("context", ""))

    return {
        "id": str(entity.get("id", "")).strip() or generate_entity_id(name, canonical_type),
        "name": name,
        "confidence": float(entity.get("confidence", 0.5) or 0.5),
        "context": str(entity.get("context", "")).strip(),
        "entity_type": canonical_type,
        "chromosome": chromosome,
        "approved": bool(entity.get("approved", True)),
        "mapped_to_existing": str(entity.get("mapped_to_existing", "")),
        "notes": str(entity.get("notes", "")),
        "paper_id": str(entity.get("paper_id", "")).strip() or paper_id,
        "legacy_index_id": f"{paper_id}_{entity_type}_{entity_index}",
    }


def normalize_paper_relationship(
    relationship: Dict[str, Any],
    paper_id: str,
    relationship_index: int,
    entity_lookup: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    lookup = entity_lookup or {}

    source_name = str(relationship.get("source_name", relationship.get("source", ""))).strip()
    target_name = str(relationship.get("target_name", relationship.get("target", ""))).strip()

    source_type = infer_relationship_entity_type(
        source_name,
        relationship.get("source_type", ""),
        lookup,
    )
    target_type = infer_relationship_entity_type(
        target_name,
        relationship.get("target_type", ""),
        lookup,
    )

    original_relation = str(
        relationship.get("original_relation", relationship.get("relation", relationship.get("edge_type", "")))
    ).strip()
    edge_type = str(relationship.get("edge_type", "")).strip() or canonicalize_edge_type(
        original_relation,
        source_type,
        target_type,
    )

    source_id = infer_relationship_entity_id(
        source_name,
        relationship.get("source_id", ""),
        source_type,
        lookup,
    )
    target_id = infer_relationship_entity_id(
        target_name,
        relationship.get("target_id", ""),
        target_type,
        lookup,
    )

    source_record = lookup.get(source_name.lower(), {})
    source_chromosome = str(
        relationship.get("source_chromosome", source_record.get("chromosome", ""))
    ).strip()

    return {
        "id": str(relationship.get("id", "")).strip() or f"{paper_id}_rel_{relationship_index}",
        "paper_id": str(relationship.get("paper_id", "")).strip() or paper_id,
        "source_id": source_id,
        "source_name": source_name,
        "source_type": source_type,
        "target_id": target_id,
        "target_name": target_name,
        "target_type": target_type,
        "edge_type": edge_type or "ASSOCIATES",
        "edge_weight": float(relationship.get("edge_weight", relationship.get("confidence", 0.5)) or 0.5),
        "evidence": str(relationship.get("evidence", "")).strip(),
        "source_chromosome": source_chromosome,
        "original_relation": original_relation,
        "extraction_method": str(relationship.get("extraction_method", "gpt4")),
        "approved": bool(relationship.get("approved", True)),
        "notes": str(relationship.get("notes", "")),
    }


def normalize_entities_for_paper(
    paper_id: str,
    entities_by_type: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    normalized: Dict[str, List[Dict[str, Any]]] = {bucket: [] for bucket in ENTITY_TYPE_BUCKETS.values()}
    for bucket in normalized:
        for index, entity in enumerate((entities_by_type or {}).get(bucket, [])):
            if not isinstance(entity, dict):
                continue
            normalized[bucket].append(normalize_paper_entity(entity, bucket, paper_id, index))
    return normalized


def normalize_relationships_for_paper(
    paper_id: str,
    relationships: Iterable[Dict[str, Any]],
    entities_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    lookup = build_entity_lookup(entities_by_type or {})
    normalized: List[Dict[str, Any]] = []
    for index, relationship in enumerate(relationships or []):
        if not isinstance(relationship, dict):
            continue
        normalized.append(normalize_paper_relationship(relationship, paper_id, index, lookup))
    return normalized
