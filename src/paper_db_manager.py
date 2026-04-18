"""
src/paper_db_manager.py
=======================
Database integration module for merging approved paper entities into the main knowledge graph.

Handles:
- Merging approved entities from papers into deb_data.py
- Enriching existing nodes with paper evidence
- Creating paper-derived edges in the graph
- Versioning and tracking data provenance
"""

import os
import sys
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Add imports
try:
    import deb_data_papers as papers_db
    import deb_data
except ImportError as e:
    logger.error(f"Could not import data modules: {e}")


def get_approved_entities() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all approved entities from papers.
    
    Returns:
        dict: Approved entities by type
    """
    approved = {
        "genes": [],
        "proteins": [],
        "diseases": [],
        "pathways": []
    }
    
    for paper_id, entities in papers_db.paper_entities.items():
        for entity_type in ["genes", "proteins", "diseases", "pathways"]:
            for entity in entities.get(entity_type, []):
                if entity.get("approved", False):
                    approved[entity_type].append({
                        "paper_id": paper_id,
                        **entity
                    })
    
    return approved


def get_approved_edges() -> List[Dict[str, Any]]:
    """
    Get all approved relationships from papers.
    
    Returns:
        list: Approved edges/relationships
    """
    return [edge for edge in papers_db.paper_edges if edge.get("approved", False)]


def entity_exists_in_graph(entity_name: str, entity_type: str) -> bool:
    """
    Check if entity already exists in main graph.
    
    Args:
        entity_name: Name of entity
        entity_type: Type (gene, protein, disease, pathway)
    
    Returns:
        bool: True if exists
    """
    for node in deb_data.nodes_data:
        if (node.get("id") == entity_name and 
            node.get("type") == entity_type):
            return True
    return False


def edge_exists_in_graph(source: str, target: str, relation: str) -> bool:
    """
    Check if relationship already exists in main graph.
    
    Args:
        source: Source entity ID
        target: Target entity ID
        relation: Relationship type
    
    Returns:
        bool: True if exists
    """
    for edge in deb_data.edges_data:
        if (edge.get("source") == source and
            edge.get("target") == target and
            edge.get("relation") == relation):
            return True
    return False


def add_paper_node(paper_id: str, paper_title: str, publication_date: str) -> Dict[str, Any]:
    """
    Add paper as a node in the knowledge graph (optional feature).
    
    Args:
        paper_id: ID of paper
        paper_title: Title of paper
        publication_date: Publication date
    
    Returns:
        dict: Paper node
    """
    paper_node = {
        "id": f"paper_{paper_id}",
        "type": "paper",
        "cluster": "Publications",
        "label": paper_title[:50],
        "publication_date": publication_date,
        "source_type": "paper"
    }
    
    # Add if not exists
    if not entity_exists_in_graph(paper_node["id"], "paper"):
        deb_data.nodes_data.append(paper_node)
        logger.info(f"Added paper node: {paper_node['id']}")
    
    return paper_node


def merge_entity_to_graph(
    entity_name: str,
    entity_type: str,
    cluster: str = "Paper-derived",
    paper_id: Optional[str] = None
) -> bool:
    """
    Add a paper-derived entity to the main knowledge graph.
    
    Args:
        entity_name: Name of entity
        entity_type: Type (gene, protein, disease, pathway)
        cluster: Cluster/category
        paper_id: ID of source paper (optional)
    
    Returns:
        bool: True if added, False if already exists
    """
    if entity_exists_in_graph(entity_name, entity_type):
        logger.info(f"Entity already exists: {entity_name} ({entity_type})")
        return False
    
    node = {
        "id": entity_name,
        "type": entity_type,
        "cluster": cluster,
        "source_type": "paper",
        "source_paper_id": paper_id
    }
    
    deb_data.nodes_data.append(node)
    logger.info(f"Added entity from paper: {entity_name} ({entity_type})")
    return True


def merge_relationship_to_graph(
    source: str,
    target: str,
    relation: str,
    confidence: float = 0.80,
    evidence: str = "",
    paper_id: Optional[str] = None
) -> bool:
    """
    Add a paper-derived relationship to the main knowledge graph.
    
    Args:
        source: Source entity ID
        target: Target entity ID
        relation: Relationship description
        confidence: Confidence score (0.0-1.0)
        evidence: Supporting evidence/text
        paper_id: Source paper ID
    
    Returns:
        bool: True if added, False if already exists
    """
    if edge_exists_in_graph(source, target, relation):
        logger.info(f"Edge already exists: {source} → {target}")
        return False
    
    edge = {
        "source": source,
        "target": target,
        "relation": relation,
        "score": confidence,
        "source_type": "paper",
        "source_paper_id": paper_id,
        "evidence": evidence
    }
    
    deb_data.edges_data.append(edge)
    logger.info(f"Added relationship from paper: {source} → {target}")
    return True


def merge_paper_entities_to_graph(
    paper_id: str,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Merge all approved entities and relationships from a paper into main graph.
    
    Args:
        paper_id: ID of paper to merge
        dry_run: If True, only count what would be merged (don't actually merge)
    
    Returns:
        dict: Statistics on merge operation
    """
    stats = {
        "paper_id": paper_id,
        "entities_added": 0,
        "entities_skipped": 0,
        "relationships_added": 0,
        "relationships_skipped": 0,
        "dry_run": dry_run
    }
    
    # Get paper info
    paper = papers_db.get_paper_by_id(paper_id)
    if not paper:
        logger.error(f"Paper not found: {paper_id}")
        return stats
    
    # Add paper node (optional)
    if not dry_run:
        add_paper_node(paper_id, paper.get("title", ""), paper.get("publication_date", ""))
    
    # Merge entities
    entities = papers_db.get_paper_entities(paper_id)
    
    for entity_type in ["genes", "proteins", "diseases", "pathways"]:
        for entity in entities.get(entity_type, []):
            if entity.get("approved", False):
                entity_name = entity.get("name", "")
                if entity_name:
                    if not dry_run:
                        if merge_entity_to_graph(
                            entity_name,
                            entity_type,
                            cluster=entity.get("mapped_to_existing", "Paper-derived"),
                            paper_id=paper_id
                        ):
                            stats["entities_added"] += 1
                        else:
                            stats["entities_skipped"] += 1
                    else:
                        if entity_exists_in_graph(entity_name, entity_type):
                            stats["entities_skipped"] += 1
                        else:
                            stats["entities_added"] += 1
    
    # Merge edges
    all_edges = [e for e in papers_db.paper_edges if e.get("paper_id") == paper_id]
    
    for edge in all_edges:
        if edge.get("approved", False):
            source = edge.get("source_name", edge.get("source", ""))
            target = edge.get("target_name", edge.get("target", ""))
            relation = edge.get("edge_type", edge.get("relation", ""))
            confidence = edge.get("edge_weight", edge.get("confidence", 0.80))
            evidence = edge.get("evidence", "")
            
            if source and target and relation:
                if not dry_run:
                    if merge_relationship_to_graph(
                        source,
                        target,
                        relation,
                        confidence,
                        evidence,
                        paper_id
                    ):
                        stats["relationships_added"] += 1
                    else:
                        stats["relationships_skipped"] += 1
                else:
                    if edge_exists_in_graph(source, target, relation):
                        stats["relationships_skipped"] += 1
                    else:
                        stats["relationships_added"] += 1
    
    logger.info(f"Merge stats for {paper_id}: {stats}")
    return stats


def merge_multiple_papers(paper_ids: List[str], dry_run: bool = False) -> Dict[str, Any]:
    """
    Merge multiple papers into the graph.
    
    Args:
        paper_ids: List of paper IDs to merge
        dry_run: If True, only count what would be merged
    
    Returns:
        dict: Aggregated statistics
    """
    total_stats = {
        "total_papers": len(paper_ids),
        "total_entities_added": 0,
        "total_entities_skipped": 0,
        "total_relationships_added": 0,
        "total_relationships_skipped": 0,
        "dry_run": dry_run,
        "paper_stats": []
    }
    
    for paper_id in paper_ids:
        stats = merge_paper_entities_to_graph(paper_id, dry_run)
        total_stats["paper_stats"].append(stats)
        
        total_stats["total_entities_added"] += stats["entities_added"]
        total_stats["total_entities_skipped"] += stats["entities_skipped"]
        total_stats["total_relationships_added"] += stats["relationships_added"]
        total_stats["total_relationships_skipped"] += stats["relationships_skipped"]
    
    return total_stats


def enrich_existing_node_with_paper(
    node_id: str,
    paper_id: str,
    entity_type: str
) -> bool:
    """
    Add paper reference to an existing node in the graph.
    
    Args:
        node_id: ID of existing node
        paper_id: ID of paper providing evidence
        entity_type: Type of entity
    
    Returns:
        bool: True if enriched
    """
    # Find node
    for node in deb_data.nodes_data:
        if node.get("id") == node_id and node.get("type") == entity_type:
            # Add paper source if not already there
            if "source_papers" not in node:
                node["source_papers"] = []
            
            if paper_id not in node["source_papers"]:
                node["source_papers"].append(paper_id)
                logger.info(f"Enriched node {node_id} with paper {paper_id}")
                return True
    
    return False


def export_merged_data_to_file(output_path: str = "./deb_data_merged.py") -> bool:
    """
    Export merged deb_data to a file for backup/versioning.
    
    Args:
        output_path: Path to export file
    
    Returns:
        bool: True if successful
    """
    try:
        # Create Python file with data
        content = f"""# Auto-generated merged data file
# Generated: {datetime.now().isoformat()}
# This file contains merged paper entities and relationships

# NODES
nodes_data = {repr(deb_data.nodes_data)}

# EDGES
edges_data = {repr(deb_data.edges_data)}
"""
        
        with open(output_path, "w") as f:
            f.write(content)
        
        logger.info(f"Exported merged data to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return False


def print_merge_report(stats: Dict[str, Any]):
    """
    Print a formatted merge operation report.
    
    Args:
        stats: Statistics dictionary from merge operation
    """
    print("\n" + "="*60)
    print("PAPER MERGE REPORT")
    print("="*60)
    
    if stats.get("total_papers"):
        print(f"\nTotal Papers: {stats['total_papers']}")
        print(f"Mode: {'DRY RUN' if stats.get('dry_run') else 'REAL'}")
        print("\nSummary:")
        print(f"  Entities Added:      {stats['total_entities_added']}")
        print(f"  Entities Skipped:    {stats['total_entities_skipped']}")
        print(f"  Relationships Added: {stats['total_relationships_added']}")
        print(f"  Relationships Skip:  {stats['total_relationships_skipped']}")
        
        print("\nPaper Details:")
        for paper_stat in stats.get("paper_stats", []):
            print(f"\n  {paper_stat['paper_id']}:")
            print(f"    - Entities: +{paper_stat['entities_added']} skip{paper_stat['entities_skipped']}")
            print(f"    - Relations: +{paper_stat['relationships_added']} skip{paper_stat['relationships_skipped']}")
    else:
        print(f"Paper ID: {stats['paper_id']}")
        print(f"Mode: {'DRY RUN' if stats.get('dry_run') else 'REAL'}")
        print(f"Entities:      +{stats['entities_added']} skip−{stats['entities_skipped']}")
        print(f"Relationships: +{stats['relationships_added']} skip−{stats['relationships_skipped']}")
    
    print("\n" + "="*60 + "\n")
