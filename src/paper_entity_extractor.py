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

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.error("OpenAI library not installed. Install with: pip install openai")


def get_openai_client():
    """Initialize and return OpenAI client."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available. Install with: pip install openai")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return openai.OpenAI(api_key=api_key)


def build_extraction_prompt(paper_text: str, title: str = "", abstract: str = "") -> str:
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

    user_message = f"""Please extract entities and relationships from this paper:

TITLE: {title if title else "Not provided"}

ABSTRACT: {abstract if abstract else "Not provided"}

FULL TEXT (first 5000 characters):
{paper_text[:5000]}

Extract entities and relationships. Respond ONLY with valid JSON."""

    return system_prompt, user_message


def extract_entities_from_text(
    paper_text: str,
    title: str = "",
    abstract: str = "",
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
        client = get_openai_client()
    except (ImportError, ValueError) as e:
        logger.error(f"Cannot initialize OpenAI client: {e}")
        return {
            "genes": [],
            "proteins": [],
            "diseases": [],
            "pathways": [],
            "relationships": [],
            "error": str(e)
        }
    
    # Build extraction prompt
    system_prompt, user_message = build_extraction_prompt(paper_text, title, abstract)
    
    try:
        logger.info("Calling GPT-4 for entity extraction...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info("GPT-4 response received")
        
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
        client = get_openai_client()
    except (ImportError, ValueError) as e:
        logger.error(f"Cannot initialize OpenAI: {e}")
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
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
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
