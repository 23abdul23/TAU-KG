"""
pages/paper_review.py
====================
Streamlit page for reviewing and correcting extracted entities from papers.

Features:
- Display extracted entities in editable table
- Approve/reject individual entities
- Map to existing nodes
- Add/edit relationships
- Bulk operations
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deb_data_papers as papers_db
import deb_data
from logger_config import setup_logger

logger = setup_logger(__name__)


def initialize_session_state():
    """Initialize session state variables."""
    if "current_review_paper" not in st.session_state:
        st.session_state.current_review_paper = None
    if "reviewed_entities" not in st.session_state:
        st.session_state.reviewed_entities = {}


def get_all_pending_papers() -> List[Dict[str, Any]]:
    """Get all papers pending review."""
    pending = [
        p for p in papers_db.papers_data
        if p["extraction_status"] in ["extracted", "reviewed"]
    ]
    return pending


def get_existing_nodes_by_type(node_type: str) -> List[str]:
    """Get all existing node names by type from deb_data."""
    existing = [
        n["id"] for n in deb_data.nodes_data
        if n.get("type") == node_type
    ]
    return sorted(existing)


def render_entity_table(
    entities: List[Dict[str, Any]],
    entity_type: str,
    paper_id: str
) -> Dict[str, Any]:
    """
    Render editable entity table using Streamlit.
    
    Args:
        entities: List of entity dictionaries
        entity_type: Type of entity (gene, protein, disease, pathway)
        paper_id: ID of paper
    
    Returns:
        dict: Updated entities with user edits
    """
    if not entities:
        st.info(f"No {entity_type} found")
        return []
    
    st.subheader(f"🔹 {entity_type.upper()} ({len(entities)})")
    
    updated_entities = []
    existing_nodes = get_existing_nodes_by_type(entity_type.rstrip('s'))
    
    for idx, entity in enumerate(entities):
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 1, 1])
            
            with col1:
                # Confidence indicator
                conf = entity.get("confidence", 0.5)
                if conf >= 0.85:
                    st.write("🟢")
                elif conf >= 0.70:
                    st.write("🟡")
                else:
                    st.write("🔴")
            
            with col2:
                # Entity name (editable)
                entity_name = st.text_input(
                    "Entity",
                    value=entity.get("name", ""),
                    key=f"{paper_id}_{entity_type}_{idx}_name",
                    label_visibility="collapsed"
                )
            
            with col3:
                # Map to existing node
                mapped = st.selectbox(
                    "Map to",
                    options=[""] + existing_nodes,
                    index=0,
                    key=f"{paper_id}_{entity_type}_{idx}_map",
                    label_visibility="collapsed"
                )
            
            with col4:
                # Confidence setting
                conf_val = st.slider(
                    "📊",
                    min_value=0.0,
                    max_value=1.0,
                    value=entity.get("confidence", 0.5),
                    step=0.05,
                    key=f"{paper_id}_{entity_type}_{idx}_conf",
                    label_visibility="collapsed"
                )
            
            with col5:
                # Approve checkbox
                approved = st.checkbox(
                    "✓",
                    value=entity.get("approved", False),
                    key=f"{paper_id}_{entity_type}_{idx}_app",
                    label_visibility="collapsed"
                )
            
            # Entity context
            with st.expander(f"Context (confidence: {conf:.2f})", expanded=False):
                st.text(entity.get("context", "No context"))
            
            # Update entity
            updated_entity = {
                **entity,
                "name": entity_name,
                "confidence": conf_val,
                "approved": approved,
                "mapped_to_existing": mapped
            }
            updated_entities.append(updated_entity)
            
            st.divider()
    
    return updated_entities


def render_paper_header(paper: Dict[str, Any]):
    """Render paper information header."""
    st.subheader(f"📑 {paper['title']}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", paper["extraction_status"].upper())
    with col2:
        st.metric("PMID", paper.get("pmid", "N/A"))
    with col3:
        st.metric("Publication", paper.get("publication_date", "N/A"))
    with col4:
        st.metric("Upload Date", paper.get("upload_date", "N/A")[:10])
    
    if paper.get("authors"):
        st.caption(f"👥 Authors: {', '.join(paper['authors'][:3])}")


def main():
    """Main Streamlit app for paper review."""
    st.set_page_config(
        page_title="🔍 Entity Review",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.title("🔍 Review Extracted Entities")
    st.markdown(
        "Review and correct entities extracted from papers. Approve high-confidence entities "
        "and map them to existing nodes in the knowledge graph."
    )
    
    # Get pending papers
    pending_papers = get_all_pending_papers()
    
    if not pending_papers:
        st.info("✅ No papers pending review!")
        st.markdown("👉 Go to **📤 Paper Upload** to add papers.")
        return
    
    # Sidebar: Select paper
    st.sidebar.subheader("📋 Papers for Review")
    
    paper_options = {
        f"{p['title'][:40]}... ({p['paper_id'][:8]})": p
        for p in pending_papers
    }
    
    selected_paper_name = st.sidebar.selectbox(
        "Select Paper",
        options=list(paper_options.keys()),
        label_visibility="collapsed"
    )
    
    current_paper = paper_options[selected_paper_name]
    st.session_state.current_review_paper = current_paper["paper_id"]
    
    # Render paper header
    render_paper_header(current_paper)
    
    # Get entities
    entities = papers_db.get_paper_entities(current_paper["paper_id"]) or {
        "genes": [],
        "proteins": [],
        "diseases": [],
        "pathways": [],
    }
    relationships = papers_db.get_paper_relationships(current_paper["paper_id"])
    
    if not any(entities.values()) and not relationships:
        st.warning("❌ No entities extracted for this paper")
        return
    
    # Create tabs for entity types
    tabs = st.tabs(["🧬 Genes", "🧪 Proteins", "🏥 Diseases", "🛣️ Pathways", "🔗 Relationships"])
    
    updated_all_entities = {}
    
    with tabs[0]:
        updated_all_entities["genes"] = render_entity_table(
            entities.get("genes", []),
            "genes",
            current_paper["paper_id"]
        )
    
    with tabs[1]:
        updated_all_entities["proteins"] = render_entity_table(
            entities.get("proteins", []),
            "proteins",
            current_paper["paper_id"]
        )
    
    with tabs[2]:
        updated_all_entities["diseases"] = render_entity_table(
            entities.get("diseases", []),
            "diseases",
            current_paper["paper_id"]
        )
    
    with tabs[3]:
        updated_all_entities["pathways"] = render_entity_table(
            entities.get("pathways", []),
            "pathways",
            current_paper["paper_id"]
        )
    
    with tabs[4]:
        st.subheader("🔗 RELATIONSHIPS")
        if not relationships:
            st.info("No relationships found")
        else:
            rel_data = []
            for idx, rel in enumerate(relationships):
                rel_approved = st.checkbox(
                    f"Approve: {rel.get('source')} → {rel.get('target')}",
                    value=rel.get("approved", False),
                    key=f"rel_app_{idx}"
                )
                rel["approved"] = rel_approved
                rel_data.append(rel)
            
            updated_all_entities["relationships"] = rel_data
    
    # Action buttons
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Approve All Visible", width="stretch"):
            try:
                # Mark all as approved
                for entity_type in ["genes", "proteins", "diseases", "pathways"]:
                    for entity in updated_all_entities.get(entity_type, []):
                        entity["approved"] = True
                for relationship in updated_all_entities.get("relationships", relationships):
                    relationship["approved"] = True
                
                # Save to database
                papers_db.set_paper_entities(current_paper["paper_id"], updated_all_entities)
                papers_db.set_paper_relationships(
                    current_paper["paper_id"],
                    updated_all_entities.get("relationships", relationships)
                )
                
                st.success("✅ All visible entities marked for approval")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with col2:
        if st.button("💾 Save Changes", width="stretch"):
            try:
                # Update entities and relationships in database
                papers_db.set_paper_entities(current_paper["paper_id"], updated_all_entities)
                papers_db.set_paper_relationships(
                    current_paper["paper_id"],
                    updated_all_entities.get("relationships", relationships)
                )
                
                st.success("✅ Changes saved!")
                logger.info(f"Paper {current_paper['paper_id']} entities updated")
            except Exception as e:
                st.error(f"❌ Error saving: {e}")
                logger.error(f"Save error: {e}")
    
    with col3:
        if st.button("🔄 Merge to Graph", width="stretch"):
            try:
                # Count approved entities
                approved_count = sum(
                    sum(1 for e in updated_all_entities.get(etype, []) if e.get("approved"))
                    for etype in ["genes", "proteins", "diseases", "pathways"]
                )
                
                if approved_count == 0:
                    st.warning("⚠️ No approved entities to merge")
                else:
                    # Update status
                    papers_db.update_paper_status(current_paper["paper_id"], "approved")
                    st.success(f"✅ {approved_count} entities marked for merge to graph")
                    st.info("👉 Use **📊 Paper DB Manager** to complete the merge")
                    logger.info(f"Paper {current_paper['paper_id']} approved for merging")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with col4:
        if st.button("🗑️ Skip Paper", width="stretch"):
            try:
                papers_db.update_paper_status(current_paper["paper_id"], "skipped")
                st.warning("Paper skipped (no merge)")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Summary statistics
    st.sidebar.divider()
    st.sidebar.subheader("📊 Review Statistics")
    
    total_entities = sum(
        len(updated_all_entities.get(etype, []))
        for etype in ["genes", "proteins", "diseases", "pathways"]
    )
    
    approved_entities = sum(
        sum(1 for e in updated_all_entities.get(etype, []) if e.get("approved"))
        for etype in ["genes", "proteins", "diseases", "pathways"]
    )
    
    st.sidebar.metric("Total Entities", total_entities)
    st.sidebar.metric("Approved", approved_entities)
    
    if total_entities > 0:
        approval_pct = (approved_entities / total_entities) * 100
        st.sidebar.progress(approval_pct / 100, text=f"{approval_pct:.0f}% approved")


if __name__ == "__main__":
    main()
