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
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging
from src.llm_provider import LLMClient

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

Also extract KEY RELATIONSHIPS:
- source entity -> relationship -> target entity
- confidence: 0.0-1.0 score
- evidence: supporting sentence from paper

OUTPUT FORMAT (JSON):
{
    "genes": [
        {"name": "MAPT", "context": "MAPT mutations cause...", "confidence": 0.95},
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
            "source": "MAPT",
            "target": "Tau Protein",
            "relation": "encodes",
            "confidence": 0.87,
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


def extract_entities_from_text(
    paper_text: str,
    title: str = "",
    abstract: str = "",
    sections: Optional[List[Dict[str, Any]]] = None,
    existing_entities: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Extract entities from paper text using GPT-4.
    
    Args:
        paper_text: Full or partial paper text
        title: Paper title
        abstract: Paper abstract
        existing_entities: Dict with keys (genes, proteins, diseases, pathways) 
                          containing lists of known entity names for boosting confidence
    
    Returns:
        dict: Extracted entities with structure:
            {
                "genes": [...],
                "proteins": [...],
                "diseases": [...],
                "pathways": [...],
                "relationships": [...]
            }
    """
    if not existing_entities:
        existing_entities = {
            "genes": [],
            "proteins": [],
            "diseases": [],
            "pathways": []
        }
    
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
    
    # Build extraction prompt
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
        
        # Try to extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            extracted_data = json.loads(json_str)
        else:
            logger.warning("Could not find JSON in GPT response")
            return _parse_text_extraction(response_text)
        
        # Boost confidence scores for entities already in database
        extracted_data = _boost_confidence_for_known_entities(
            extracted_data, existing_entities
        )
        
        logger.info(f"Extracted {_count_entities(extracted_data)} entities total")
        return extracted_data
        
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
    {{"source": "GENE1", "target": "PROTEIN1", "relation": "encodes", "confidence": 0.9, "evidence": "..."}},
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
            return relationships
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
            confidence = rel.get("confidence", 0)
            if confidence >= min_confidence:
                if all(k in rel for k in ["source", "target", "relation"]):
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
    formatted = {}
    
    for entity_type in ["genes", "proteins", "diseases", "pathways"]:
        formatted[entity_type] = []
        
        for i, entity in enumerate(extracted_data.get(entity_type, [])):
            formatted_entity = {
                "id": f"{paper_id}_{entity_type}_{i}",
                "name": entity.get("name", ""),
                "confidence": entity.get("confidence", 0.5),
                "context": entity.get("context", ""),
                "entity_type": entity_type,
                "approved": False,
                "mapped_to_existing": "",
                "notes": ""
            }
            formatted[entity_type].append(formatted_entity)
    
    # Format relationships
    formatted["relationships"] = []
    for i, rel in enumerate(extracted_data.get("relationships", [])):
        formatted_rel = {
            "id": f"{paper_id}_rel_{i}",
            "source": rel.get("source", ""),
            "target": rel.get("target", ""),
            "relation": rel.get("relation", ""),
            "confidence": rel.get("confidence", 0.5),
            "evidence": rel.get("evidence", ""),
            "approved": False,
            "notes": ""
        }
        formatted["relationships"].append(formatted_rel)
    
    return formatted
