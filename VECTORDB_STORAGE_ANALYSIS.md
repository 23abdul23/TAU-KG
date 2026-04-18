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

## 🆕 Schema Alignment Analysis

### Your Target Schema (CSV Format)
```csv
source_id,target_id,edge_type,source_type,target_type,source_name,target_name,edge_weight,source_chromosome
mapk_protein,breast,INTERACTS,Protein,Tissue,mapk_protein,Breast,0.81,
vemurafenib,sunitinib,EXPRESSES,Drug,Drug,vemurafenib,Sunitinib,0.82,
colon_cancer,PI3K_AKT_pathway,TREATS,Disease,Pathway,colon_cancer,Pi3K Akt Pathway,0.95,
lung_cancer,TP53,INTERACTS,Disease,Gene,lung_cancer,Tp53,0.81,
pi3k_protein,EGFR,EXPRESSES,Protein,Gene,pi3k_protein,Egfr,0.96,
BRCA2,pancreas,ACTIVATES,Gene,Tissue,BRCA2,Pancreas,0.75,18
PIK3CA,lung_cancer,PARTICIPATES,Gene,Disease,PIK3CA,Lung Cancer,0.79,14
sorafenib,imatinib,PARTICIPATES,Drug,Drug,sorafenib,Imatinib,0.73,
```

### Current Extracted Edge Format (from GPT-4 extraction)
```json
{
  "source": "MAPT",
  "target": "Tau Protein",
  "relation": "encodes",
  "confidence": 0.87,
  "evidence": "...",
  "paper_id": "paper_123",
  "source_type": "paper",
  "extraction_method": "gpt4",
  "approved": false,
  "extracted_date": "2024-01-15"
}
```

---

## 🔄 Gap Analysis: Current vs Target Schema

### ✅ Fields That Match or Are Close:
| Target Schema | Current Schema | Status | Gap |
|---|---|---|---|
| `source_name` | `source` | ✅ Match | None |
| `target_name` | `target` | ✅ Match | None |
| `edge_weight` | `confidence` | ✅ Match | None - both 0-1 scores |
| `source_type` | `source_type` | ⚠️ Partial | Currently: "paper", need entity type |
| Evidence in docs | `evidence` | ✅ Match | None |

### ❌ Fields That Are Missing or Different:

| Target Schema | Current Status | Impact | Solution |
|---|---|---|---|
| `source_id` | ❌ Missing | Need unique identifiers | Generate from normalized names |
| `target_id` | ❌ Missing | Need unique identifiers | Generate from normalized names |
| `edge_type` | ❌ Different naming | "relation" vs "edge_type" | Standardize relationship types |
| `target_type` | ❌ Missing | No target entity type | Extract from entity lookup |
| `source_chromosome` | ❌ Missing | Optional but useful | Extract if available (genes only) |

---

## 🎯 How to Align Current → Target Schema

### Step 1: Update Extraction Prompt
Update `src/paper_entity_extractor.py` to capture types:

**CURRENT:**
```json
{
  "source": "MAPT",
  "target": "Tau Protein",
  "relation": "encodes",
  "confidence": 0.87
}
```

**TARGET:**
```json
{
  "source_id": "MAPT",
  "source_name": "MAPT",
  "source_type": "Gene",
  "target_id": "tau_protein",
  "target_name": "Tau Protein",
  "target_type": "Protein",
  "edge_type": "encodes",
  "edge_weight": 0.87,
  "evidence": "...",
  "source_chromosome": "17"
}
```

### Step 2: Standardize Edge Types
Replace "relation" with "edge_type" using standardized values:

**Current (unclear):**
- encodes, regulates, involved_in, phosphorylates, inhibits, etc.

**Target (standardized):**
- ✅ EXPRESSES (gene/protein → protein)
- ✅ ENCODES (gene → protein)
- ✅ INTERACTS (any ↔ any)
- ✅ TREATS (drug → disease)
- ✅ PARTICIPATES (any → pathway)
- ✅ ACTIVATES (any → any)
- ✅ INHIBITS (any → any)
- ✅ REGULATES (any → any)
- ✅ ASSOCIATES (any ↔ any)

### Step 3: Add Entity Type Information

**Current Process:**
- Store only entity names
- Lose type information

**Target Process:**
- Extract source_type from entity extraction (gene/protein/disease/pathway)
- Extract target_type from entity extraction
- Map to extended types: Gene, Protein, Disease, Pathway, Drug, Tissue, etc.

### Step 4: Create Unique IDs

**Option A: Hash-based (Deterministic)**
```python
def generate_entity_id(name: str, entity_type: str) -> str:
    normalized = f"{entity_type.lower()}_{name.lower().replace(' ', '_')}"
    return hashlib.md5(normalized.encode()).hexdigest()[:8]
```

**Option B: Slug-based (Human-readable)**
```python
def generate_entity_id(name: str, entity_type: str) -> str:
    normalized = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    return f"{entity_type.lower()}_{normalized}"
```

### Step 5: Add Chromosome Information (Optional for Genes)

For genes extracted from papers, try to add chromosome info:

```python
# In paper entity extraction, add chromosome if mentioned
# Examples: "MAPT is located on chromosome 17"
{
  "source_id": "gene_mapt",
  "source_name": "MAPT",
  "source_type": "Gene",
  "source_chromosome": "17",  # ← NEW
  "target_id": "protein_tau",
  "target_name": "Tau Protein",
  "target_type": "Protein",
  "edge_type": "ENCODES",
  "edge_weight": 0.87
}
```

---

## 📊 Implementation Roadmap

### Phase 1: Update Extraction (HIGH PRIORITY)
**Goal:** Add missing fields to extraction prompt

1. **Update `src/paper_entity_extractor.py`**
   - Add entity_type to each extracted entity
   - Add `source_id` and `target_id` generation
   - Standardize edge_type values
   - Rename "relation" → "edge_type"
   - Rename "confidence" → "edge_weight"

2. **Update JSON schema in extraction prompt:**
   ```python
   {
     "genes": [
       {
         "id": "gene_mapt",
         "name": "MAPT",
         "context": "...",
         "confidence": 0.95,
         "chromosome": "17"  # NEW - if mentioned
       }
     ],
     "relationships": [
       {
         "source_id": "gene_mapt",
         "source_name": "MAPT",
         "source_type": "Gene",
         "target_id": "protein_tau",
         "target_name": "Tau Protein",
         "target_type": "Protein",
         "edge_type": "ENCODES",
         "edge_weight": 0.87,
         "evidence": "..."
       }
     ]
   }
   ```

### Phase 2: Store in Vector DB (HIGH PRIORITY)
**Goal:** Make edges searchable

1. **Create `paper_edges` Chroma collection**
2. **Vectorize each edge as searchable document:**
   ```
   document: "Relationship: MAPT [ENCODES] Tau Protein (confidence: 0.87)"
   metadata: {
     "source_id": "gene_mapt",
     "source_name": "MAPT",
     "source_type": "Gene",
     "target_id": "protein_tau",
     "target_name": "Tau Protein",
     "target_type": "Protein",
     "edge_type": "ENCODES",
     "edge_weight": 0.87,
     "evidence": "...",
     "paper_id": "paper_123",
     "source_chromosome": "17"
   }
   ```

### Phase 3: Enhance VectorDBManager (MEDIUM PRIORITY)
**Goal:** Support querying edges

1. Add `search_relationships()` method
2. Add `search_entities_by_type()` method
3. Add relationship filtering capabilities

### Phase 4: Update UI & Queries (MEDIUM PRIORITY)
**Goal:** Display enriched relationship data

1. Update `chat_app.py` to show edge types
2. Display entity types in network visualization
3. Filter/search by edge_type and entity types

---

## 🔗 Data Model Comparison Table

```
┌──────────────────────────┬──────────────────────┬──────────────────────┐
│ Your Target Schema       │ Current Schema       │ Recommendation       │
├──────────────────────────┼──────────────────────┼──────────────────────┤
│ source_id                │ (missing)            │ Add to extraction    │
│ target_id                │ (missing)            │ Add to extraction    │
│ source_name              │ source               │ Rename to source_name│
│ target_name              │ target               │ Rename to target_name│
│ source_type              │ source_type: "paper" │ Use entity type      │
│ target_type              │ (missing)            │ Add to extraction    │
│ edge_type                │ relation             │ Standardize types    │
│ edge_weight              │ confidence           │ Rename (0-1 scale)   │
│ source_chromosome        │ (missing)            │ Optional, add if LLM │
│ evidence                 │ evidence             │ Keep (vectorize in  │
│                          │                      │ document text)       │
│ paper_id                 │ paper_id             │ Keep (metadata)      │
│ extraction_method        │ extraction_method    │ Keep (track source)  │
│ approved                 │ approved             │ Keep (QA flag)       │
└──────────────────────────┴──────────────────────┴──────────────────────┘
```

---

## Next Steps

1. ✅ **Phase 1:** Update extraction to add `source_id`, `target_id`, `source_type`, `target_type`
2. ✅ **Phase 1:** Standardize edge_type names (ENCODES, INTERACTS, TREATS, PARTICIPATES, ACTIVATES, INHIBITS, REGULATES, ASSOCIATES)
3. ⚠️ **Phase 2:** Create `paper_edges` vector collection
4. ⚠️ **Phase 2:** Implement `sync_paper_edges_to_vectordb()` function
5. 🔗 **Phase 3:** Add entity type filtering to search functions
6. 🎯 **Phase 4:** Update UI to display standardized schema data
