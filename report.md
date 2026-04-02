# TAU-KG Project Context Report

## 1. Purpose and Scope
TAU-KG is a Python + Streamlit biomedical knowledge application that combines:
- Gene/protein semantic search over a local vector database.
- Chat responses enhanced by OpenAI and PubMed citations.
- A paper-ingestion workflow (PDF and PMC/NLM links) that extracts entities/relations and stages them for review.
- Graph analytics and visualization utilities for biological network exploration.

This report is intended to provide complete, practical project context for an LLM collaborator.

## 2. Tech Stack
- UI: Streamlit (multi-page app)
- Search store: ChromaDB (persistent local directory)
- Embeddings: sentence-transformers (`all-MiniLM-L6-v2`)
- AI: OpenAI Chat Completions (GPT-4 configured in code)
- Literature: NCBI E-utilities (PubMed `esearch` + `efetch`)
- Data processing: pandas, networkx, plotly, pyvis
- PDF extraction: pdfplumber with PyPDF2 fallback

Key dependency declarations: `requirements.txt`

## 3. Runtime Entry Points
### Primary app
- `chat_app.py`
- Includes two tabs: chat and gene network.

### Additional Streamlit app (legacy/parallel visualization)
- `streamlit_app.py`
- Dedicated network visualization and analytics interface.

### Paper workflow pages
- `pages/paper_upload.py`
- `pages/paper_review.py`
- `pages/paper_browser.py`

### Batch processing CLI
- `scripts/batch_upload_papers.py`

## 4. High-Level Architecture
### UI layer
- Chat + network tab in `chat_app.py`
- Paper Upload / Review / Browser pages under `pages/`
- Separate visualization-heavy app in `streamlit_app.py`

### Service layer
- `vector_db_manager.py`: vector DB lifecycle, search, optional AI-enhanced response generation, paper collection helpers
- `citations.py`: PubMed query generation/fetch/parsing + optional LLM-assisted entity extraction for query optimization
- `src/pdf_processor.py`: PDF text extraction and PMC URL ingestion/cleaning
- `src/paper_entity_extractor.py`: GPT-based entity + relationship extraction from paper text
- `src/paper_db_manager.py`: merge approved paper entities/edges into main graph data

### Data layer
- ChromaDB at `./chroma_db/`
- Main graph in code (`deb_data.py`: nodes and edges lists)
- Paper staging store in code (`deb_data_papers.py`: papers, entities, edges, metadata)
- CSV seed data (`nodes_main.csv`)
- Uploaded PDFs (`./uploaded_papers/`)

## 5. Core Workflows
## 5.1 Chat query workflow
1. User initializes DB from sidebar (`chat_app.py`).
2. `VectorDBManager` loads `nodes_main.csv` into Chroma if empty.
3. User submits chat prompt.
4. Semantic retrieval: `search_similar(query, n_results=5)`.
5. Response path:
- Preferred: enhanced response via OpenAI + citations via PubMed.
- Fallback: basic local response formatting if enhancement fails.
6. Optional paper retrieval: `search_papers_for_query` uses paper collection search when available.
7. UI renders response, citations, related papers, and expandable retrieval diagnostics.

## 5.2 Citation workflow
1. Query enters `citations.py` pipeline.
2. Entities are extracted (LLM or regex fallback, depending on config/availability).
3. PubMed query is optimized and sent to NCBI `esearch`.
4. PMIDs are fetched via `efetch`.
5. XML is parsed into citation objects shown in chat.

## 5.3 Paper upload workflow (PDF)
1. User uploads one or more PDFs in `pages/paper_upload.py`.
2. File saved to `uploaded_papers/` with UUID filename.
3. `process_pdf` extracts text and metadata.
4. User reviews editable metadata fields.
5. User triggers AI extraction for genes/proteins/diseases/pathways + relationships.
6. User saves to paper store (`deb_data_papers`), status moved to `extracted`.

## 5.4 Paper upload workflow (PMC/NLM link)
1. User selects link mode and pastes one or more PMC URLs.
2. `process_pmc_url` fetches HTML and converts to cleaned text.
3. Boilerplate removal in `_normalize_pmc_text` strips many PMC/NCBI navigation/footer patterns and truncates after reference-like section markers.
4. Metadata is parsed from HTML + extracted text.
5. Extraction/save path is the same as PDF path.

Important current behavior:
- Multi-link mode is supported.
- Streamlit widget keys are namespaced per paper source key to avoid duplicate text area/input IDs in batch link runs.

## 5.5 Review + merge workflow
1. `pages/paper_review.py` lists papers with `extracted` or `reviewed` status.
2. Reviewer edits entity names, mappings, confidence, approval flags.
3. Reviewer saves updates back to `papers_db.paper_entities`.
4. "Merge to Graph" button currently marks paper as approved and indicates merge intent.
5. Actual merge logic is implemented in `src/paper_db_manager.py` (`merge_paper_entities_to_graph`), which adds approved entities/edges to `deb_data` if not already present.

## 5.6 Paper browsing workflow
- `pages/paper_browser.py` supports title/author/PMID search, entity filtering, metadata and extracted-entity inspection, and status visibility.

## 6. UI/UX Map
### `chat_app.py`
- Sidebar:
- DB initialization
- DB stats (sources/types)
- Sample queries
- Clear chat
- Main area:
- Tab 1: chat conversation, citations, related papers, detailed retrieved-node drilldown
- Tab 2: gene-network visualization and query-level graph exploration

### `pages/paper_upload.py`
- Source selector:
- Upload PDF files
- Paste PMC/NLM links
- Extraction/review form:
- editable title/authors/PMID/DOI/date
- abstract and text preview expanders
- AI extraction button
- save button
- Sidebar metrics: papers count/review status/entities total

### `pages/paper_review.py`
- Sidebar paper picker
- Per-entity editable rows with:
- confidence indicator
- mapping select box
- confidence slider
- approval checkbox
- relationship approval
- Actions:
- approve all visible
- save changes
- mark for merge
- skip paper

### `pages/paper_browser.py`
- Search and filter papers
- Paper cards with extraction status and entity counts
- Detailed paper view with metadata + tabbed extracted entities/relationships

## 7. Data Models
### 7.1 Main graph (`deb_data.py`)
- `nodes_data`: list of graph nodes with fields such as `id`, `type`, `cluster`.
- `edges_data`: list of edges with `source`, `target`, `relation`, `score`.

### 7.2 Paper staging store (`deb_data_papers.py`)
- `papers_data`: per-paper metadata and extraction state.
- `paper_entities`: dict keyed by paper_id with arrays for entity categories.
- `paper_edges`: extracted relationships with confidence/evidence and approval flags.
- `paper_metadata`: aggregate counters.

Paper source tracking fields include:
- `source`: `user_uploaded`, `pmc_link`, or `pmid_fetched`
- `source_url`: original link (for link-based ingestion)

## 8. Configuration and Environment
### Config file
`config.json` controls:
- PubMed retry/timeouts/citation limits
- Network/proxy usage
- Citation extraction model options
- Paper upload thresholds and limits

### Environment variables
Expected in `.env`:
- `OPENAI_API_KEY`
- `NCBI_API_KEY` (optional but recommended)
- Proxy env vars if needed

### Logging
- Core logger utility: `src/logger_config.py` (`get_logger`)
- Compatibility shim: root `logger_config.py` exposes `setup_logger` for legacy imports.

## 9. Deployment and Run Commands
### Local
- Install: `pip install -r requirements.txt`
- Chat app: `streamlit run chat_app.py`
- Network-only app: `streamlit run streamlit_app.py`

### Docker
- `Dockerfile` runs `streamlit_app.py` by default.
- `compose.yml` maps port `8501` and builds from local Dockerfile.

Operational note:
- Container default entrypoint is currently the network visualization app, not `chat_app.py`.

## 10. Current Caveats and Technical Debt
1. Paper and graph stores are in-memory Python modules (`deb_data.py`, `deb_data_papers.py`), not a transactional database.
2. Merge actions rely on explicit function calls; UI and merge orchestration are partially decoupled.
3. `citations.py` contains duplicate import/setup blocks, increasing maintenance risk.
4. Batch script imports `validate_extracted_entities` from `src/paper_entity_extractor.py`; verify this symbol exists/aligns with runtime before relying on CLI batch mode.
5. Docker default app (`streamlit_app.py`) differs from main chat app (`chat_app.py`).
6. PDF extraction has no OCR path for scanned-image PDFs.
7. PMC text cleanup is heuristic (pattern-based) and may require updates if source page structure shifts.

## 11. Suggested Mental Model for LLM Contributors
When making changes, treat the system as two coupled planes:
- Retrieval/chat plane: `chat_app.py` + `vector_db_manager.py` + `citations.py` + ChromaDB.
- Curation/ingestion plane: `pages/paper_*` + `src/pdf_processor.py` + `src/paper_entity_extractor.py` + `deb_data_papers.py` + `src/paper_db_manager.py`.

Safe implementation strategy:
1. Keep paper extraction output schema stable (`genes`, `proteins`, `diseases`, `pathways`, `relationships`).
2. Preserve `paper_id` as the join key across review, browser, and merge.
3. Maintain unique Streamlit widget keys for repeated multi-paper components.
4. Prefer additive changes to `deb_data_papers` helpers over direct list/dict mutation in many pages.
5. Validate both UI flow and fallback behavior when OpenAI or PubMed is unavailable.

## 12. Quick File Index for LLMs
- Main chat app: `chat_app.py`
- Vector DB service: `vector_db_manager.py`
- Citations + PubMed: `citations.py`
- Paper upload UI: `pages/paper_upload.py`
- Paper review UI: `pages/paper_review.py`
- Paper browser UI: `pages/paper_browser.py`
- Paper text/metadata extraction: `src/pdf_processor.py`
- Paper entity extraction: `src/paper_entity_extractor.py`
- Merge service: `src/paper_db_manager.py`
- Main graph data: `deb_data.py`
- Paper staging data: `deb_data_papers.py`
- Config: `config.json`
- Dependencies: `requirements.txt`
- Containerization: `Dockerfile`, `compose.yml`

## 13. Recommended Prompt Snippet for Future LLM Sessions
Use this project context:
- TAU-KG has a Streamlit chat app with ChromaDB semantic retrieval and optional OpenAI/PubMed enrichment.
- It also has a multi-step paper ingestion pipeline (PDF + PMC URLs), review page, and browser page.
- Paper data lives in `deb_data_papers.py`; main graph data lives in `deb_data.py`.
- Keep extraction schema stable and preserve multi-paper Streamlit key uniqueness.
- Prefer minimal, local changes and verify impacted flow end-to-end.
