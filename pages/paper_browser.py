"""
pages/paper_browser.py
======================
Streamlit page for browsing, searching, and visualizing papers and their entities.

Features:
- Search papers by title, authors, PMID
- Filter by entity
- View paper with highlighted entities
- Network visualization of paper-entity relationships
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
    """Initialize session state."""
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "selected_paper" not in st.session_state:
        st.session_state.selected_paper = None


def search_papers(query: str, search_type: str = "all") -> List[Dict[str, Any]]:
    """
    Search papers by title, authors, or PMID.
    
    Args:
        query: Search query
        search_type: Type of search (all, title, authors, pmid)
    
    Returns:
        list: Matching papers
    """
    query_lower = query.lower()
    results = []
    
    for paper in papers_db.papers_data:
        if search_type == "all":
            match = (
                query_lower in paper.get("title", "").lower() or
                any(query_lower in author.lower() for author in paper.get("authors", [])) or
                query_lower in str(paper.get("pmid", ""))
            )
        elif search_type == "title":
            match = query_lower in paper.get("title", "").lower()
        elif search_type == "authors":
            match = any(query_lower in author.lower() for author in paper.get("authors", []))
        elif search_type == "pmid":
            match = query_lower in str(paper.get("pmid", ""))
        else:
            match = False
        
        if match:
            results.append(paper)
    
    return results


def get_papers_with_entity(entity_name: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Find all papers containing a specific entity.
    
    Args:
        entity_name: Name of entity to search
        entity_type: Optional type filter
    
    Returns:
        list: Papers containing entity
    """
    papers_with_entity = []
    
    for paper_id, entities in papers_db.paper_entities.items():
        found = False
        
        if entity_type:
            types_to_search = [entity_type]
        else:
            types_to_search = ["genes", "proteins", "diseases", "pathways"]
        
        for etype in types_to_search:
            for entity in entities.get(etype, []):
                if entity.get("name", "").lower() == entity_name.lower():
                    found = True
                    break
            if found:
                break
        
        if found:
            paper = papers_db.get_paper_by_id(paper_id)
            if paper:
                papers_with_entity.append(paper)
    
    return papers_with_entity


def render_paper_card(paper: Dict[str, Any]):
    """Render a paper information card."""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**{paper['title']}**")
            st.caption(
                f"👥 {', '.join(paper.get('authors', [])[:2])} | "
                f"📅 {paper.get('publication_date', 'N/A')[:10]} | "
                f"PMID: {paper.get('pmid', 'N/A')}"
            )
        
        with col2:
            status_emoji = {
                "pending": "⏱️",
                "extracted": "🔄",
                "reviewed": "👁️",
                "approved": "✅"
            }
            st.write(status_emoji.get(paper.get("extraction_status"), "❓"))
        
        # Show extracted entities count
        entities = papers_db.get_paper_entities(paper["paper_id"])
        relationships = papers_db.get_paper_relationships(paper["paper_id"])
        if entities or relationships:
            entities = entities or {
                "genes": [],
                "proteins": [],
                "diseases": [],
                "pathways": [],
            }
            entity_counts = {
                "Genes": len(entities.get("genes", [])),
                "Proteins": len(entities.get("proteins", [])),
                "Diseases": len(entities.get("diseases", [])),
                "Pathways": len(entities.get("pathways", [])),
                "Relationships": len(relationships)
            }
            
            st.markdown(
                " | ".join([f"🧬 {entity_counts['Genes']}", 
                          f"🧪 {entity_counts['Proteins']}", 
                          f"🏥 {entity_counts['Diseases']}", 
                          f"🛣️ {entity_counts['Pathways']}"])
            )
    
    return paper["paper_id"]


def render_paper_detail(paper: Dict[str, Any]):
    """Render detailed paper view."""
    st.markdown("---")
    st.subheader(f"📑 {paper['title']}")
    
    # Paper metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Publication Date", paper.get("publication_date", "N/A")[:10])
    with col2:
        st.metric("PMID", paper.get("pmid", "N/A"))
    with col3:
        st.metric("Status", paper.get("extraction_status", "N/A").upper())
    
    # Authors
    if paper.get("authors"):
        with st.expander("👥 Authors"):
            for author in paper["authors"]:
                st.write(f"• {author}")
    
    # Abstract
    if paper.get("abstract"):
        with st.expander("📋 Abstract"):
            st.text(paper["abstract"])
    
    # DOI and link
    if paper.get("doi"):
        st.markdown(f"**DOI:** [{paper['doi']}](https://doi.org/{paper['doi']})")
    
    if paper.get("pmid"):
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"
        st.markdown(f"**PubMed:** [View on PubMed]({pubmed_url})")
    
    # Extracted entities
    st.subheader("🎯 Extracted Entities")
    
    entities = papers_db.get_paper_entities(paper["paper_id"]) or {
        "genes": [],
        "proteins": [],
        "diseases": [],
        "pathways": [],
    }
    relationships = papers_db.get_paper_relationships(paper["paper_id"])
    
    if any(entities.values()) or relationships:
        entity_tabs = st.tabs(["🧬 Genes", "🧪 Proteins", "🏥 Diseases", "🛣️ Pathways", "🔗 Relationships"])
        
        with entity_tabs[0]:
            genes = entities.get("genes", [])
            if genes:
                df_genes = pd.DataFrame([
                    {
                        "Gene": e.get("name", ""),
                        "Confidence": f"{e.get('confidence', 0):.2f}",
                        "Status": "✅" if e.get("approved") else "⏳",
                        "Mapped": e.get("mapped_to_existing", "-")
                    }
                    for e in genes
                ])
                st.dataframe(df_genes, width="stretch")
            else:
                st.info("No genes found")
        
        with entity_tabs[1]:
            proteins = entities.get("proteins", [])
            if proteins:
                df_proteins = pd.DataFrame([
                    {
                        "Protein": e.get("name", ""),
                        "Confidence": f"{e.get('confidence', 0):.2f}",
                        "Status": "✅" if e.get("approved") else "⏳",
                        "Mapped": e.get("mapped_to_existing", "-")
                    }
                    for e in proteins
                ])
                st.dataframe(df_proteins, width="stretch")
            else:
                st.info("No proteins found")
        
        with entity_tabs[2]:
            diseases = entities.get("diseases", [])
            if diseases:
                df_diseases = pd.DataFrame([
                    {
                        "Disease": e.get("name", ""),
                        "Confidence": f"{e.get('confidence', 0):.2f}",
                        "Status": "✅" if e.get("approved") else "⏳",
                        "Mapped": e.get("mapped_to_existing", "-")
                    }
                    for e in diseases
                ])
                st.dataframe(df_diseases, width="stretch")
            else:
                st.info("No diseases found")
        
        with entity_tabs[3]:
            pathways = entities.get("pathways", [])
            if pathways:
                df_pathways = pd.DataFrame([
                    {
                        "Pathway": e.get("name", ""),
                        "Confidence": f"{e.get('confidence', 0):.2f}",
                        "Status": "✅" if e.get("approved") else "⏳",
                        "Mapped": e.get("mapped_to_existing", "-")
                    }
                    for e in pathways
                ])
                st.dataframe(df_pathways, width="stretch")
            else:
                st.info("No pathways found")
        
        with entity_tabs[4]:
            if relationships:
                df_rels = pd.DataFrame([
                    {
                        "Source": r.get("source_name", r.get("source", "")),
                        "Source Type": r.get("source_type", "Entity"),
                        "Edge Type": r.get("edge_type", r.get("relation", "")),
                        "Target": r.get("target_name", r.get("target", "")),
                        "Target Type": r.get("target_type", "Entity"),
                        "Weight": f"{float(r.get('edge_weight', r.get('confidence', 0.0))):.2f}",
                        "Evidence": str(r.get("evidence", ""))[:180],
                        "Status": "✅" if r.get("approved") else "⏳"
                    }
                    for r in relationships
                ])
                st.dataframe(df_rels, width="stretch")
            else:
                st.info("No relationships found")
    else:
        st.warning("No entities extracted for this paper")


def main():
    """Main Streamlit app for paper browser."""
    st.set_page_config(
        page_title="📚 Paper Browser",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.title("📚 Paper Browser")
    st.markdown(
        "Search and browse uploaded papers. Filter by entities or explore paper relationships."
    )
    
    # Check if there are any papers
    if not papers_db.papers_data:
        st.info("📭 No papers uploaded yet. Go to **📤 Paper Upload** to add papers.")
        return
    
    # Search interface
    st.subheader("🔍 Search Papers")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "Search query",
            placeholder="Search by title, authors, or PMID...",
            label_visibility="collapsed"
        )
    
    with col2:
        search_type = st.selectbox(
            "Search type",
            options=["all", "title", "authors", "pmid"],
            label_visibility="collapsed"
        )
    
    with col3:
        search_entity = st.checkbox("Search by entity", value=False)
    
    # Perform search
    if search_query:
        if search_entity:
            # Search for papers containing entity
            matching_papers = get_papers_with_entity(search_query)
            if not matching_papers:
                st.warning(f"❌ No papers found containing entity: {search_query}")
        else:
            # Regular text search
            matching_papers = search_papers(search_query, search_type)
            if not matching_papers:
                st.warning(f"❌ No papers found matching: {search_query}")
    else:
        matching_papers = papers_db.papers_data
    
    # Display results
    if matching_papers:
        st.markdown(f"**Found {len(matching_papers)} paper(s)**")
        
        # Filter status
        status_filter = st.sidebar.multiselect(
            "Filter by status",
            options=["pending", "extracted", "reviewed", "approved"],
            default=["extracted", "reviewed", "approved"]
        )
        
        matching_papers = [
            p for p in matching_papers
            if p.get("extraction_status") in status_filter
        ]
        
        # Display papers
        for paper in matching_papers:
            paper_id = render_paper_card(paper)
            
            if st.button("View Details", key=f"details_{paper_id}"):
                st.session_state.selected_paper = paper_id
        
        # Show selected paper detail
        if st.session_state.selected_paper:
            paper = papers_db.get_paper_by_id(st.session_state.selected_paper)
            if paper:
                render_paper_detail(paper)
    
    # Statistics sidebar
    st.sidebar.divider()
    st.sidebar.subheader("📊 Database Statistics")
    
    st.sidebar.metric("Total Papers", papers_db.paper_metadata["total_papers"])
    st.sidebar.metric("Extracted", papers_db.paper_metadata["total_entities_extracted"])
    st.sidebar.metric("Approved", papers_db.paper_metadata["papers_approved"])
    
    # Status breakdown
    stats_by_status = {}
    for paper in papers_db.papers_data:
        status = paper.get("extraction_status", "unknown")
        stats_by_status[status] = stats_by_status.get(status, 0) + 1
    
    if stats_by_status:
        st.sidebar.markdown("**Status Breakdown:**")
        for status, count in sorted(stats_by_status.items()):
            st.sidebar.write(f"• {status.upper()}: {count}")


if __name__ == "__main__":
    main()
