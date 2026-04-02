"""
deb_data_papers.py
================
Paper-centric knowledge graph data structure for TAU-KG.
Stores papers, extracted entities, relationships, and metadata.

Structure:
- papers_data: Metadata for each uploaded paper (title, authors, PMID, date, etc.)
- paper_entities: Extracted entities from papers (genes, proteins, diseases, pathways) with confidence
- paper_edges: Relationships found in papers (source → target with evidence)
- paper_metadata: Aggregated statistics and tracking information
"""

# ========================
# PAPERS METADATA
# ========================
# Core information about each uploaded paper
papers_data = []

# Example structure:
# {
#     "paper_id": "uuid_or_pmid",
#     "title": "MAPT mutations and tau aggregation in neurodegeneration",
#     "authors": ["John Doe", "Jane Smith"],
#     "publication_date": "2024-01-15",
#     "pmid": "12345678",
#     "doi": "10.1234/example.2024",
#     "abstract": "Full abstract text...",
#     "pdf_path": "uploaded_papers/uuid_filename.pdf",
#     "source": "user_uploaded",  # user_uploaded | pmc_link | pmid_fetched
#     "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
#     "sections": [
#         {"type": "methods", "text": "..."},
#         {"type": "results", "text": "..."}
#     ],
#     "upload_date": "2024-01-20",
#     "extraction_status": "pending",  # pending | extracted | reviewed | approved
#     "notes": "Optional user notes"
# }


# ========================
# PAPER ENTITIES
# ========================
# Extracted entities from papers with confidence scores
paper_entities = {}

# Example structure:
# "paper_uuid_1": {
#     "genes": [
#         {
#             "name": "MAPT",
#             "entity_type": "gene",
#             "confidence": 0.92,
#             "context": "MAPT mutations cause tau aggregation...",
#             "sentence_index": 5,
#             "approved": False,
#             "mapped_to_existing": "MAPT"
#         }
#     ],
#     "proteins": [
#         {
#             "name": "Tau Protein",
#             "entity_type": "protein",
#             "confidence": 0.88,
#             "context": "The tau protein undergoes hyperphosphorylation...",
#             "sentence_index": 7,
#             "approved": False,
#             "mapped_to_existing": "Tau Protein"
#         }
#     ],
#     "diseases": [
#         {
#             "name": "Alzheimer's Disease",
#             "entity_type": "disease",
#             "confidence": 0.95,
#             "context": "In Alzheimer's disease, tau pathology...",
#             "sentence_index": 12,
#             "approved": True,
#             "mapped_to_existing": "Alzheimer's Disease"
#         }
#     ],
#     "pathways": [
#         {
#             "name": "Tauopathy",
#             "entity_type": "pathway",
#             "confidence": 0.85,
#             "context": "The tauopathy cascade involves...",
#             "sentence_index": 8,
#             "approved": False,
#             "mapped_to_existing": "Tauopathy"
#         }
#     ]
# }


# ========================
# PAPER EDGES
# ========================
# Relationships discovered in papers
paper_edges = []

# Example structure:
# [
#     {
#         "paper_id": "paper_uuid_1",
#         "source": "MAPT",
#         "target": "Tau Protein",
#         "relation": "encodes",
#         "confidence": 0.87,
#         "evidence_text": "MAPT gene encodes the tau protein...",
#         "sentence_index": 3,
#         "approved": False,
#         "source_type": "paper",  # paper | curated
#         "extraction_method": "gpt4"
#     }
# ]


# ========================
# PAPER METADATA & STATISTICS
# ========================
# Aggregated statistics and tracking
paper_metadata = {
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
        "total": 0
    },
    "genes_from_papers": [],  # List of unique genes found across papers
    "proteins_from_papers": [],
    "diseases_from_papers": [],
    "pathways_from_papers": [],
    "last_extraction_date": None,
    "last_review_date": None,
}


# ========================
# UTILITY FUNCTIONS FOR DATA MANAGEMENT
# ========================

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
    sections=None
):
    """
    Add a new paper to papers_data.
    
    Args:
        paper_id (str): Unique identifier (UUID)
        title (str): Paper title
        authors (list): List of author names
        pmid (str): PubMed ID
        doi (str): Digital Object Identifier
        abstract (str): Paper abstract
        pdf_path (str): Path to stored PDF or empty string for link-based imports
        publication_date (str): Publication date (YYYY-MM-DD format)
        source (str): Origin of the paper data
        source_url (str): Original URL when imported from a PMC/PubMed link
    
    Returns:
        dict: The added paper record
    """
    import datetime
    
    paper = {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "pmid": pmid,
        "doi": doi,
        "abstract": abstract,
        "pdf_path": pdf_path,
        "publication_date": publication_date,
        "source": source,
        "source_url": source_url,
        "sections": sections or [],
        "upload_date": datetime.datetime.now().isoformat(),
        "extraction_status": "pending",
        "notes": ""
    }
    papers_data.append(paper)
    paper_metadata["total_papers"] += 1
    paper_metadata["papers_pending_extraction"] += 1
    return paper


def add_entities(paper_id, entity_type, entities):
    """
    Add extracted entities for a paper.
    
    Args:
        paper_id (str): Paper ID
        entity_type (str): Type of entity (genes, proteins, diseases, pathways)
        entities (list): List of entity dicts with name, confidence, context, etc.
    """
    if paper_id not in paper_entities:
        paper_entities[paper_id] = {
            "genes": [],
            "proteins": [],
            "diseases": [],
            "pathways": []
        }
    
    paper_entities[paper_id][entity_type] = entities
    
    # Update metadata
    paper_metadata["total_entities_extracted"] += len(entities)
    paper_metadata["entity_coverage"][entity_type] += len(entities)
    paper_metadata["entity_coverage"]["total"] = sum(
        paper_metadata["entity_coverage"][k] for k in ["genes", "proteins", "diseases", "pathways"]
    )


def add_edges(paper_id, edges):
    """
    Add discovered relationships for a paper.
    
    Args:
        paper_id (str): Paper ID
        edges (list): List of edge dicts (source, target, relation, confidence, etc.)
    """
    for edge in edges:
        edge["paper_id"] = paper_id
        if "source_type" not in edge:
            edge["source_type"] = "paper"
        if "extraction_method" not in edge:
            edge["extraction_method"] = "gpt4"
        paper_edges.append(edge)


def update_paper_status(paper_id, status):
    """
    Update extraction status of a paper.
    
    Args:
        paper_id (str): Paper ID
        status (str): New status (pending, extracted, reviewed, approved)
    """
    for paper in papers_data:
        if paper["paper_id"] == paper_id:
            previous_status = paper["extraction_status"]
            paper["extraction_status"] = status
            
            # Update counts
            if previous_status == "pending" and status != "pending":
                paper_metadata["papers_pending_extraction"] -= 1
            
            if status == "reviewed":
                paper_metadata["papers_pending_review"] += 1
            elif previous_status == "reviewed" and status != "reviewed":
                paper_metadata["papers_pending_review"] -= 1
            
            if status == "approved":
                paper_metadata["papers_approved"] += 1
            elif previous_status == "approved" and status != "approved":
                paper_metadata["papers_approved"] -= 1
            
            return paper
    return None


def get_paper_by_id(paper_id):
    """Get paper details by ID."""
    for paper in papers_data:
        if paper["paper_id"] == paper_id:
            return paper
    return None


def get_paper_entities(paper_id):
    """Get extracted entities for a paper."""
    return paper_entities.get(paper_id, None)


def get_papers_for_entity(entity_name, entity_type=None):
    """
    Find all papers mentioning a specific entity.
    
    Args:
        entity_name (str): Entity name to search
        entity_type (str): Optional type filter (gene, protein, disease, pathway)
    
    Returns:
        list: List of (paper_id, entity_dict) tuples
    """
    results = []
    
    for paper_id, entities in paper_entities.items():
        if entity_type:
            types_to_search = [entity_type]
        else:
            types_to_search = ["genes", "proteins", "diseases", "pathways"]
        
        for etype in types_to_search:
            for entity in entities.get(etype, []):
                if entity["name"].lower() == entity_name.lower():
                    results.append((paper_id, entity))
    
    return results


def get_approved_entities():
    """Get all approved entities across all papers."""
    approved = {
        "genes": [],
        "proteins": [],
        "diseases": [],
        "pathways": []
    }
    
    for paper_id, entities in paper_entities.items():
        for etype in ["genes", "proteins", "diseases", "pathways"]:
            for entity in entities.get(etype, []):
                if entity.get("approved", False):
                    approved[etype].append({
                        "paper_id": paper_id,
                        **entity
                    })
    
    return approved


def get_approved_edges():
    """Get all approved edges across all papers."""
    return [edge for edge in paper_edges if edge.get("approved", False)]
