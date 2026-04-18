# Vector Database Storage Analysis - TAU-KG

## Current State: What's Being Stored After Entity Extraction

### 1. **Main Collection: `gene_proteins`** 
This is the primary vector DB collection loaded from `nodes_main.csv`:

#### Stored Data Structure:
```
{
  "document": "Gene/Protein: MYC (ID: 12345) - Type: gene - Source: CURATED_GRAPH",
  "metadata": {
    "node_index": 12345,
    "node_id": 12345 (or string),
    "node_type": "gene|protein|disease|pathway|entity",
    "node_name": "MYC",
    "node_source": "CURATED_GRAPH"
  },
  "id": "gene_12345",
  "vector_embedding": [embedding array]
}
```

#### What It Contains:
- ✅ **Genes** - node_name, node_id, relationships from curated graph
- ✅ **Proteins** - same as genes
- ✅ **Diseases** - node_name, node_id
- ✅ **Pathways** - node_name, node_id
- ✅ **Relationships/Edges** - NOT directly stored in vector DB, but referenced via the graph structure in `deb_data.py` (edges_data)
- ❌ **Paper-derived entities** - Only synced if explicitly called

**Total Documents**: Currently 272 documents from CSV

---

### 2. **Paper Entities** (In-Memory Data Structure)
Stored in `deb_data_papers.py` and on disk at `data/paper_store.json`:

#### Structure:
```python
paper_entities: Dict[str, Dict[str, List[Dict[str, Any]]]]
# Example:
{
  "paper_123": {
    "genes": [
      {
        "name": "MAPT",
        "context": "MAPT mutations cause tauopathy...",
        "confidence": 0.95,
        "approved": False,
        "extraction_method": "gpt4",
        "extracted_date": "2024-01-15"
      },
      ...
    ],
    "proteins": [
      {
        "name": "Tau Protein",
        "context": "...",
        "confidence": 0.90,
        "approved": False
      },
      ...
    ],
    "diseases": [
      {
        "name": "Alzheimer's Disease",
        "context": "...",
        "confidence": 0.92,
        "approved": False
      },
      ...
    ],
    "pathways": [
      {
        "name": "Autophagy",
        "context": "...",
        "confidence": 0.88,
        "approved": False
      },
      ...
    ]
  }
}
```

**What It Contains**: 
- ✅ **Genes extracted from papers**
- ✅ **Proteins extracted from papers**
- ✅ **Diseases extracted from papers**
- ✅ **Pathways extracted from papers**
- ❌ **Edge data** - stored separately (see below)
- ❌ **Relationships** - stored separately (see below)

---

### 3. **Paper Edges/Relationships**
Stored in `deb_data_papers.py` and on disk at `data/paper_store.json`:

#### Structure:
```python
paper_edges: List[Dict[str, Any]]
# Example:
[
  {
    "source": "MAPT",
    "target": "Tau Protein",
    "relation": "encodes",
    "evidence": "MAPT gene encodes tau protein",
    "confidence": 0.92,
    "paper_id": "paper_123",
    "source_type": "paper",
    "extraction_method": "gpt4",
    "approved": False,
    "extracted_date": "2024-01-15"
  },
  {
    "source": "Tau Protein",
    "target": "Alzheimer's Disease",
    "relation": "involved_in",
    "evidence": "Tau hyperphosphorylation is central to...",
    "confidence": 0.95,
    "paper_id": "paper_123",
    "source_type": "paper",
    "extraction_method": "gpt4",
    "approved": False
  },
  ...
]
```

**What It Contains**:
- ✅ **Source entity** (gene/protein/disease/pathway)
- ✅ **Target entity** (gene/protein/disease/pathway)
- ✅ **Relationship type** (encodes, regulates, involves_in, etc.)
- ✅ **Evidence** (text snippet from paper)
- ✅ **Confidence score**
- ✅ **Paper ID** (reference to source paper)
- ✅ **Approval status** (manual review flag)

---

### 4. **Paper Metadata**
Stored alongside entities for tracking and statistics:

```python
{
  "total_papers": 5,
  "papers_pending_extraction": 0,
  "papers_pending_review": 2,
  "papers_approved": 3,
  "total_entities_extracted": 127,
  "total_entities_approved": 89,
  "entity_coverage": {
    "genes": 34,
    "proteins": 28,
    "diseases": 31,
    "pathways": 14,
    "total": 107
  },
  "genes_from_papers": ["MAPT", "APP", "PSEN1", ...],
  "proteins_from_papers": ["Tau Protein", "Amyloid-beta", ...],
  "diseases_from_papers": ["Alzheimer's Disease", ...],
  "pathways_from_papers": ["Autophagy", "Neuroinflammation", ...],
  "last_extraction_date": "2024-01-20",
  "last_review_date": "2024-01-21"
}
```

---

## What's NOT Currently Stored in Vector DB

| Feature | Status | Location |
|---------|--------|----------|
| Paper-derived genes | ❌ Not in vector DB | Only in `deb_data_papers.py` |
| Paper-derived proteins | ❌ Not in vector DB | Only in `deb_data_papers.py` |
| Paper-derived diseases | ❌ Not in vector DB | Only in `deb_data_papers.py` |
| Paper-derived pathways | ❌ Not in vector DB | Only in `deb_data_papers.py` |
| Paper-extracted relationships (edges) | ❌ Not in vector DB | Only in `deb_data_papers.py` |
| Curated graph edges | ✅ In memory only | `deb_data.py` edges_data |
| Paper context/snippets | ⚠️ Partial | Only when syncing to vector DB |

---

## Current Sync Function

### `sync_paper_entities_to_main_collection()`

This function attempts to add paper-derived entities to the main vector DB collection:

```python
# Only adds ENTITIES, not edges
# Example: when approved=True, syncs genes/proteins/diseases/pathways
sync_paper_entities_to_main_collection(
    paper_entities=deb_data_papers.paper_entities,
    source_name="PAPER_INGEST",
    only_approved=True
)
```

**Creates documents like:**
```
document: "Gene/Protein: MAPT | Type: gene | Source: PAPER_INGEST | Paper: Alzheimer's Research Study | Context: MAPT mutations cause..."

metadata: {
  "node_index": 123456,
  "node_id": "MAPT",
  "node_type": "gene",
  "node_name": "MAPT",
  "node_source": "PAPER_INGEST",
  "source_paper_id": "paper_123"
}
```

---

## What YOU Need to Store (Your Requirements)

### ✅ Genes
- **Current**: Stored in CSV (curated) + in-memory (paper-derived)
- **Vector DB**: Only curated genes vectorized
- **Action**: Need to sync paper genes to vector DB ✓

### ✅ Proteins
- **Current**: Stored in CSV (curated) + in-memory (paper-derived)
- **Vector DB**: Only curated proteins vectorized
- **Action**: Need to sync paper proteins to vector DB ✓

### ✅ Diseases
- **Current**: Stored in CSV (curated) + in-memory (paper-derived)
- **Vector DB**: Only curated diseases vectorized
- **Action**: Need to sync paper diseases to vector DB ✓

### ✅ Pathways
- **Current**: Stored in CSV (curated) + in-memory (paper-derived)
- **Vector DB**: Only curated pathways vectorized
- **Action**: Need to sync paper pathways to vector DB ✓

### ❌ Relationships/Edges from Papers
- **Current**: Stored in-memory only (`paper_edges` list)
- **Vector DB**: NOT stored anywhere in vector DB
- **Action**: CREATE new collection for paper edges ⚠️

---

## Recommended Changes

### 1. **Auto-Sync Paper Entities** (Easy)
Call `sync_paper_entities_to_main_collection()` after extraction:
```python
# In chat_app.py or paper upload workflow
result = db_manager.sync_paper_entities_to_main_collection(
    paper_entities=papers_db.paper_entities,
    source_name="PAPER_INGEST",
    only_approved=True  # or False to include pending
)
print(f"Synced {result['added']} entities to vector DB")
```

### 2. **Create Paper Edges Collection** (Medium)
Add a new Chroma collection for storing paper-extracted relationships:
```python
# New collection: "paper_edges"
# Store each edge as searchable document

edges_collection = client.create_collection(name="paper_edges")

# For each paper edge:
document = f"Relationship: {source} --[{relation}]-> {target} (Evidence: {evidence})"
metadata = {
  "source": source,
  "target": target,
  "relation": relation,
  "evidence": evidence,
  "confidence": confidence,
  "paper_id": paper_id,
  "source_type": "paper"
}
```

### 3. **Store Paper Metadata** (Optional)
Create separate collection for paper-entity metadata:
```python
# New collection: "paper_metadata"
# Store statistics, coverage, and lineage info
```

---

## Summary Table

### Current Vector DB Storage after Entity Extraction:

```
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR DB Collections                        │
├─────────────────────────────────────────────────────────────────┤
│ Collection Name  │ Documents │ Content                          │
├──────────────────┼───────────┼──────────────────────────────────┤
│ gene_proteins    │ 272       │ Curated genes/proteins/diseases  │
│                  │           │ from nodes_main.csv              │
├──────────────────┼───────────┼──────────────────────────────────┤
│ papers           │ 0         │ (Optional collection, empty)     │
├──────────────────┼───────────┼──────────────────────────────────┤
│ paper_entities   │ 0         │ (Optional collection, empty)     │
├──────────────────┼───────────┼──────────────────────────────────┤
│ paper_edges      │ N/A       │ (DOES NOT EXIST - needs creation)│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              Non-Vectorized Data (In-Memory)                    │
├─────────────────────────────────────────────────────────────────┤
│ Data Structure          │ Location              │ Contains      │
├─────────────────────────┼───────────────────────┼───────────────┤
│ paper_entities          │ deb_data_papers.py    │ All paper     │
│                         │ data/paper_store.json │ entities      │
│                         │                       │ (genes,       │
│                         │                       │ proteins,     │
│                         │                       │ diseases,     │
│                         │                       │ pathways)     │
├─────────────────────────┼───────────────────────┼───────────────┤
│ paper_edges             │ deb_data_papers.py    │ ALL paper     │
│                         │ data/paper_store.json │ relationships │
│                         │                       │ & edges       │
├─────────────────────────┼───────────────────────┼───────────────┤
│ edges_data (curated)    │ deb_data.py           │ Curated graph │
│                         │ edges_main.csv        │ relationships │
└─────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. ✅ **Call sync function** to vectorize paper entities ← START HERE
2. ⚠️ **Create paper_edges collection** to vectorize relationships
3. 🔗 **Update search functions** to query paper edges
4. 🎯 **Update UI** to display paper-extracted relationships
