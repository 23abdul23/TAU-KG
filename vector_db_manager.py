import pandas as pd
import chromadb
from chromadb.config import Settings
import os
import hashlib
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import json
from tqdm import tqdm
import time
from src.llm_provider import LLMClient
from src.paper_schema import normalize_entities_for_paper, normalize_relationships_for_paper, slugify_identifier

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Enhanced imports for LLM and PubMed integration

try:
    from citations import fetch_pubmed_citations, Citation
    CITATIONS_AVAILABLE = True
except ImportError:
    CITATIONS_AVAILABLE = False
    print("Warning: Citations module not available. Literature search disabled.")

try:
    import deb_data_papers as papers_db
    PAPERS_DB_AVAILABLE = True
except ImportError:
    PAPERS_DB_AVAILABLE = False
    papers_db = None

class VectorDBManager:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "gene_proteins"):
        """
        Initialize the Vector Database Manager
        
        Args:
            db_path: Path to store the ChromaDB database
            collection_name: Name of the collection to store documents
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize sentence transformer for embeddings
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token=hf_token)
            except TypeError:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=hf_token)
        else:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize active LLM client based on LLM_PROVIDER.
        self.llm_client = LLMClient()
        if self.llm_client.is_available():
            print(f"{self.llm_client.get_provider_label()} client initialized for enhanced responses.")
        else:
            print(f"Warning: LLM client unavailable ({self.llm_client.unavailable_reason}). Enhanced responses disabled.")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")

    @staticmethod
    def _distance_to_similarity(distance: Any) -> float:
        """Convert an arbitrary non-negative vector distance into a bounded similarity percentage."""
        try:
            numeric_distance = max(float(distance), 0.0)
        except (TypeError, ValueError):
            return 0.0
        return 100.0 / (1.0 + numeric_distance)

    @staticmethod
    def _stable_numeric_id(seed: str) -> int:
        """Generate a deterministic positive integer from a string seed."""
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return int(digest, 16)

    def _normalize_node_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Repair missing or placeholder metadata for paper-derived entities."""
        normalized = dict(metadata or {})

        node_index = normalized.get("node_index")
        try:
            node_index = int(node_index)
        except (TypeError, ValueError):
            node_index = -1

        if node_index < 0:
            seed = "|".join([
                str(normalized.get("node_source", "")),
                str(normalized.get("node_type", "")),
                str(normalized.get("node_id", "")),
                str(normalized.get("source_paper_id", "")),
            ])
            normalized["node_index"] = self._stable_numeric_id(seed)
        else:
            normalized["node_index"] = node_index

        return normalized

    def _hydrate_paper_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten paper search metadata and enrich it from the in-memory papers DB when available."""
        hydrated = dict(metadata or {})
        authors_raw = hydrated.get("authors", "")
        if isinstance(authors_raw, str):
            hydrated["authors"] = [author for author in authors_raw.split("|") if author]
        elif not isinstance(authors_raw, list):
            hydrated["authors"] = []

        paper_id = str(hydrated.get("paper_id", "")).strip()
        if PAPERS_DB_AVAILABLE and paper_id:
            paper_record = papers_db.get_paper_by_id(paper_id)
            if paper_record:
                hydrated["title"] = paper_record.get("title") or hydrated.get("title", "")
                hydrated["authors"] = paper_record.get("authors") or hydrated.get("authors", [])
                hydrated["abstract"] = paper_record.get("abstract", "")
                hydrated["doi"] = paper_record.get("doi") or hydrated.get("doi", "")
                hydrated["pmid"] = paper_record.get("pmid") or hydrated.get("pmid", "")
                hydrated["publication_date"] = paper_record.get("publication_date") or hydrated.get("publication_date", "")
                hydrated["source_url"] = paper_record.get("source_url", "")

        publication_date = str(hydrated.get("publication_date", "") or "")
        hydrated["publication_year"] = publication_date[:4] if len(publication_date) >= 4 else ""
        return hydrated

    def _build_paper_entity_document_text(
        self,
        paper_id: str,
        entity_type: str,
        entity: Dict[str, Any],
        source_name: str,
    ) -> str:
        """Build a richer entity document so vector search has entity and paper context."""
        node_type = {
            "genes": "gene",
            "proteins": "protein",
            "diseases": "disease",
            "pathways": "pathway",
        }.get(entity_type, "entity")

        entity_name = str(entity.get("name", "")).strip()
        context = str(entity.get("context", "")).strip()
        paper_title = ""
        paper_abstract = ""

        if PAPERS_DB_AVAILABLE and paper_id:
            paper_record = papers_db.get_paper_by_id(paper_id)
            if paper_record:
                paper_title = str(paper_record.get("title", "")).strip()
                paper_abstract = str(paper_record.get("abstract", "")).strip()

        parts = [entity_name, f"Type: {node_type}", f"Source: {source_name}"]
        if paper_title:
            parts.append(f"Paper: {paper_title}")
        if context:
            parts.append(f"Context: {context[:250]}")
        elif paper_abstract:
            parts.append(f"Paper abstract: {paper_abstract[:250]}")

        return " | ".join(part for part in parts if part)

    def _build_paper_edge_document_text(self, edge: Dict[str, Any]) -> str:
        """Build searchable relationship text for a paper-derived edge."""
        paper_title = ""
        paper_id = str(edge.get("paper_id", "")).strip()
        if PAPERS_DB_AVAILABLE and paper_id:
            paper_record = papers_db.get_paper_by_id(paper_id)
            if paper_record:
                paper_title = str(paper_record.get("title", "")).strip()

        parts = [
            f"Relationship: {edge.get('source_name', '')} [{edge.get('edge_type', 'ASSOCIATES')}] {edge.get('target_name', '')}",
            f"Source type: {edge.get('source_type', 'Entity')}",
            f"Target type: {edge.get('target_type', 'Entity')}",
        ]
        if edge.get("evidence"):
            parts.append(f"Evidence: {str(edge.get('evidence', ''))[:400]}")
        if paper_title:
            parts.append(f"Paper: {paper_title}")
        return " | ".join(part for part in parts if part)

    @staticmethod
    def _normalize_label_filter(values: Optional[List[str]]) -> Optional[set]:
        if not values:
            return None
        return {str(value).strip().lower() for value in values if str(value).strip()}
    
    def load_csv_to_vectordb(self, csv_path: str, batch_size: int = 100):
        """
        Load CSV data into the vector database with optimized batch processing
        
        Args:
            csv_path: Path to the CSV file
            batch_size: Number of documents to process in each batch
        """
        # Check if collection is already populated
        current_count = self.collection.count()
        if current_count > 0:
            print(f"Collection already contains {current_count} documents. Skipping load.")
            return
        
        print(f"Reading CSV file: {csv_path}")
        # Read CSV file
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        print(f"Found {total_rows} rows to process")
        
        # Process in batches to avoid memory issues and provide progress feedback
        processed_count = 0
        
        # Create progress bar
        print("Processing data in batches...")
        with tqdm(total=total_rows, desc="Loading data", unit="rows") as pbar:
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                documents = []
                metadatas = []
                ids = []
                
                for index, row in batch_df.iterrows():
                    # Create a text representation of each row
                    doc_text = f"Gene/Protein: {row['node_name']} (ID: {row['node_id']}) - Type: {row['node_type']} - Source: {row['node_source']}"
                    
                    # Create metadata with safe type conversion
                    try:
                        node_id = int(row['node_id']) if pd.notna(row['node_id']) else 0
                    except (ValueError, TypeError):
                        # Handle non-numeric node_id values
                        node_id = str(row['node_id'])
                    
                    metadata = {
                        "node_index": int(row['node_index']),
                        "node_id": node_id,
                        "node_type": str(row['node_type']),
                        "node_name": str(row['node_name']),
                        "node_source": str(row['node_source'])
                    }
                    
                    documents.append(doc_text)
                    metadatas.append(metadata)
                    ids.append(f"gene_{row['node_index']}")
                
                # Add batch to collection
                try:
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    processed_count += len(documents)
                    pbar.update(len(documents))
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"Error processing batch {start_idx}-{end_idx}: {e}")
                    continue
        
        print(f"Successfully loaded {processed_count} documents into the vector database.")
    
    def search_similar(self, query: str, n_results: int = 15) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector database
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i] if results['distances'] else None
                result = {
                    'document': results['documents'][0][i],
                    'metadata': self._normalize_node_metadata(results['metadatas'][0][i]),
                    'distance': distance,
                    'similarity_score': self._distance_to_similarity(distance),
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_all_gene_names(self) -> List[str]:
        """
        Get all gene names from the database
        
        Returns:
            List of all gene names
        """
        all_results = self.collection.get()
        gene_names = [metadata['node_name'] for metadata in all_results['metadatas']]
        return sorted(list(set(gene_names)))
    
    def get_gene_info(self, gene_name: str) -> Dict[str, Any]:
        """
        Get specific information about a gene
        
        Args:
            gene_name: Name of the gene to search for
            
        Returns:
            Gene information dictionary
        """
        results = self.collection.query(
            query_texts=[gene_name],
            n_results=1,
            where={"node_name": gene_name}
        )
        
        if results['documents'] and results['documents'][0]:
            return {
                'document': results['documents'][0][0],
                'metadata': self._normalize_node_metadata(results['metadatas'][0][0])
            }
        return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database
        
        Returns:
            Database statistics
        """
        all_results = self.collection.get()
        total_docs = len(all_results['documents'])
        
        # Count by source
        sources = {}
        node_types = {}
        
        for metadata in all_results['metadatas']:
            source = metadata['node_source']
            node_type = metadata['node_type']
            
            sources[source] = sources.get(source, 0) + 1
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        stats = {
            'total_documents': total_docs,
            'sources': sources,
            'node_types': node_types
        }

        # Optional additional collections for papers and extracted entities
        try:
            papers_collection = self.client.get_collection(name="papers")
            stats['papers_documents'] = papers_collection.count()
        except Exception:
            stats['papers_documents'] = 0

        try:
            paper_entities_collection = self.client.get_collection(name="paper_entities")
            stats['paper_entities_documents'] = paper_entities_collection.count()
        except Exception:
            stats['paper_entities_documents'] = 0

        try:
            paper_edges_collection = self.client.get_collection(name="paper_edges")
            stats['paper_edges_documents'] = paper_edges_collection.count()
        except Exception:
            stats['paper_edges_documents'] = 0

        stats['total_documents_all_collections'] = (
            stats['total_documents'] +
            stats['papers_documents'] +
            stats['paper_entities_documents'] +
            stats['paper_edges_documents']
        )

        return stats

    def sync_paper_entities_to_main_collection(
        self,
        paper_entities: Dict[str, Any],
        source_name: str = "PAPER_INGEST",
        only_approved: bool = False
    ) -> Dict[str, int]:
        """
        Add new unique paper-derived entities into the main gene/protein collection.

        Returns summary stats for added/skipped/total entities.
        """
        entity_types = ["genes", "proteins", "diseases", "pathways"]
        candidates = []

        for paper_id, entities in paper_entities.items():
            for entity_type in entity_types:
                for entity in entities.get(entity_type, []):
                    if only_approved and not entity.get("approved", False):
                        continue
                    name = str(entity.get("name", "")).strip()
                    if not name:
                        continue
                    candidates.append((paper_id, entity_type, name))

        if not candidates:
            return {"total_candidates": 0, "added": 0, "skipped": 0}

        # Build a fast lookup of existing IDs in the main collection.
        existing_ids = set(self.collection.get().get("ids", []))

        documents = []
        metadatas = []
        ids = []
        added = 0
        skipped = 0
        added_by_type = {entity_type: 0 for entity_type in entity_types}
        candidates_by_type = {entity_type: 0 for entity_type in entity_types}

        # Use stable deterministic IDs to prevent duplicate insertions.
        for paper_id, entity_type, name in candidates:
            candidates_by_type[entity_type] += 1
            normalized = ''.join(ch if ch.isalnum() else '_' for ch in name.lower()).strip('_')
            doc_id = f"paper_node_{entity_type}_{normalized}"
            if doc_id in existing_ids:
                skipped += 1
                continue

            node_type = {
                "genes": "gene",
                "proteins": "protein",
                "diseases": "disease",
                "pathways": "pathway",
            }.get(entity_type, "entity")

            doc_text = self._build_paper_entity_document_text(paper_id, entity_type, entity, source_name)
            metadata = {
                "node_index": self._stable_numeric_id(doc_id),
                "node_id": name,
                "node_type": node_type,
                "node_name": name,
                "node_source": source_name,
                "source_paper_id": str(paper_id),
            }

            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(doc_id)
            existing_ids.add(doc_id)
            added += 1
            added_by_type[entity_type] += 1

        if ids:
            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

        return {
            "total_candidates": len(candidates),
            "added": added,
            "skipped": skipped,
            "added_by_type": added_by_type,
            "candidates_by_type": candidates_by_type,
        }

    def search_entities_by_type(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        node_sources: Optional[List[str]] = None,
        n_results: int = 15,
    ) -> List[Dict[str, Any]]:
        """Search the main entity collection and filter by type/source metadata."""
        results = self.search_similar(query, n_results=max(n_results * 3, n_results))
        type_filter = self._normalize_label_filter(entity_types)
        source_filter = self._normalize_label_filter(node_sources)

        filtered: List[Dict[str, Any]] = []
        for result in results:
            metadata = result.get("metadata", {})
            node_type = str(metadata.get("node_type", "")).strip().lower()
            node_source = str(metadata.get("node_source", "")).strip().lower()
            if type_filter and node_type not in type_filter:
                continue
            if source_filter and node_source not in source_filter:
                continue
            filtered.append(result)
            if len(filtered) >= n_results:
                break
        return filtered

    def sync_paper_edges_to_vectordb(
        self,
        paper_edges: List[Dict[str, Any]],
        only_approved: bool = True,
        collection_name: str = "paper_edges",
    ) -> Dict[str, int]:
        """Upsert paper relationships into the searchable paper edge collection."""
        collection = self._get_or_create_collection(collection_name)
        self.paper_edges_collection = collection

        grouped_edges: Dict[str, List[Dict[str, Any]]] = {}
        grouped_entities: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        if PAPERS_DB_AVAILABLE:
            grouped_entities = {
                str(paper_id): normalize_entities_for_paper(str(paper_id), entities)
                for paper_id, entities in getattr(papers_db, "paper_entities", {}).items()
            }

        for edge in paper_edges or []:
            if not isinstance(edge, dict):
                continue
            paper_id = str(edge.get("paper_id", "")).strip()
            grouped_edges.setdefault(paper_id, []).append(edge)

        total_candidates = 0
        total_upserted = 0
        for paper_id, edges in grouped_edges.items():
            normalized_edges = normalize_relationships_for_paper(
                paper_id,
                edges,
                grouped_entities.get(paper_id, {}),
            )
            if only_approved:
                normalized_edges = [edge for edge in normalized_edges if edge.get("approved", False)]

            total_candidates += len(edges)
            if not normalized_edges:
                continue

            try:
                collection.delete(where={"paper_id": paper_id})
            except Exception:
                pass

            documents = [self._build_paper_edge_document_text(edge) for edge in normalized_edges]
            metadatas = normalized_edges
            ids = [f"paper_edge_{paper_id}_{index}" for index, _ in enumerate(normalized_edges)]
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            total_upserted += len(ids)

        return {
            "total_candidates": total_candidates,
            "upserted": total_upserted,
        }
    
    def _format_context_for_gpt(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Format search results into context for GPT-4
        
        Args:
            search_results: List of search results from vector database
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant gene/protein data found in the database."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            relevance = result.get('similarity_score', self._distance_to_similarity(result.get('distance')))
            
            context_part = f"""
Gene/Protein {i}:
- Name: {metadata['node_name']}
- Node ID: {metadata['node_id']}
- Type: {metadata['node_type']}
- Source: {metadata['node_source']}
- Relevance: {relevance:.1f}%
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _get_gpt_response(self, query: str, context: str, citations_context: str = "", max_tokens: int = 1024) -> str:
        """
        Get GPT-4 response based on query and context
        
        Args:
            query: User query
            context: Formatted context from database
            citations_context: Formatted citations context
            max_tokens: Maximum tokens for response
            
        Returns:
            GPT-4 generated response
        """
        if not self.llm_client.is_available():
            return "Enhanced responses not available (LLM client not initialized)."
        
        system_prompt = f"""You are a concise biomedical AI assistant. Provide BRIEF, focused responses about genes/proteins.

RULES:
1. **BE CONCISE**: Keep responses short and to the point
2. **Use Database**: Reference gene/protein data with node_id when available
3. **Structure**: Use bullet points and clear sections
4. **Cite Literature**: Reference provided citations when relevant

DATABASE CONTEXT:
{context}

LITERATURE CONTEXT:
{citations_context}

USER QUERY: {query}

Provide a CONCISE response covering:
• **Function**: What it does (1-2 sentences)
• **Clinical Relevance**: Disease/therapeutic connections (1-2 sentences)
• **Key Points**: Notable characteristics (bullet points)

Keep it brief but informative."""

        try:
            response = self.llm_client.generate_text(
                system_prompt="You are a concise biomedical AI assistant.",
                user_prompt=system_prompt,
                model=os.getenv("CHAT_LLM_MODEL", ""),
                max_tokens=max_tokens,
                temperature=0.1,
            )

            return response.text.strip()
            
        except Exception as e:
            return f"Error generating enhanced response: {str(e)}"
    
    def _get_pubmed_citations(self, query: str, max_citations: int = 5) -> List[Citation]:
        """
        Get PubMed citations for the query
        
        Args:
            query: Search query
            max_citations: Maximum number of citations to return
            
        Returns:
            List of Citation objects
        """
        if not CITATIONS_AVAILABLE:
            return []
        
        try:
            citations = fetch_pubmed_citations(
                query=query,
                max_citations=max_citations,
                prioritize_reviews=True,
                max_age_years=5
            )
            return citations
        except Exception as e:
            print(f"Error fetching citations: {str(e)}")
            return []
    
    def _format_citations_for_gpt(self, citations: List[Citation]) -> str:
        """
        Format citations into context for GPT-4
        
        Args:
            citations: List of Citation objects
            
        Returns:
            Formatted citations string
        """
        if not citations:
            return "No recent literature found."
        
        citations_parts = []
        for i, citation in enumerate(citations, 1):
            citation_text = f"""
Citation {i}:
- Title: {citation.title}
- Authors: {citation.authors if citation.authors else 'N/A'}
- Journal: {citation.journal} ({citation.year if citation.year else 'N/A'})
- PMID: {citation.pmid if citation.pmid else 'N/A'}
"""
            citations_parts.append(citation_text)
        
        return "\n".join(citations_parts)
    
    def generate_enhanced_response(self, query: str, search_results: List[Dict[str, Any]], max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Generate enhanced response using GPT-4 and PubMed citations
        
        Args:
            query: User query
            search_results: Vector database search results
            max_tokens: Maximum tokens for GPT response
            
        Returns:
            Dictionary containing GPT response, citations, and metadata
        """
        # Format context for GPT-4
        context = self._format_context_for_gpt(search_results)
        
        # Get PubMed citations first
        citations = self._get_pubmed_citations(query, max_citations=5)
        
        # Format citations for GPT-4
        citations_context = self._format_citations_for_gpt(citations)
        
        # Get GPT-4 response with both database and citations context
        gpt_response = self._get_gpt_response(query, context, citations_context, max_tokens)
        
        return {
            'gpt_response': gpt_response,
            'citations': citations,
            'search_results': search_results,
            'context_used': context,
            'citations_context_used': citations_context,
            'has_openai': self.llm_client.is_available(),
            'has_llm': self.llm_client.is_available(),
            'llm_provider': self.llm_client.provider,
            'has_citations': CITATIONS_AVAILABLE
        }
    
    def create_papers_collection(self, collection_name: str = "papers") -> None:
        """
        Create a separate collection for papers if it doesn't exist.
        
        Args:
            collection_name: Name of the papers collection
        """
        try:
            self.papers_collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing papers collection: {collection_name}")
        except:
            self.papers_collection = self.client.create_collection(name=collection_name)
            print(f"Created new papers collection: {collection_name}")

    def create_paper_edges_collection(self, collection_name: str = "paper_edges") -> None:
        """Create a separate collection for paper-derived relationships if needed."""
        try:
            self.paper_edges_collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing paper edges collection: {collection_name}")
        except Exception:
            self.paper_edges_collection = self.client.create_collection(name=collection_name)
            print(f"Created new paper edges collection: {collection_name}")

    def _get_or_create_collection(self, collection_name: str):
        """Return a collection, creating it on demand."""
        try:
            return self.client.get_collection(name=collection_name)
        except Exception:
            return self.client.create_collection(name=collection_name)

    def upsert_paper_edge_to_vectordb(
        self,
        edge: Dict[str, Any],
        collection_name: str = "paper_edges",
    ) -> Dict[str, int]:
        """Upsert a single normalized paper relationship into the edge collection."""
        if not edge:
            return {"upserted": 0}

        collection = self._get_or_create_collection(collection_name)
        self.paper_edges_collection = collection
        paper_id = str(edge.get("paper_id", "")).strip() or "paper"
        normalized_edges = normalize_relationships_for_paper(
            paper_id,
            [edge],
            normalize_entities_for_paper(paper_id, getattr(papers_db, "paper_entities", {}).get(paper_id, {})) if PAPERS_DB_AVAILABLE else {},
        )
        if not normalized_edges:
            return {"upserted": 0}

        normalized_edge = normalized_edges[0]
        collection.upsert(
            documents=[self._build_paper_edge_document_text(normalized_edge)],
            metadatas=[normalized_edge],
            ids=[f"paper_edge_{paper_id}_{slugify_identifier(normalized_edge.get('id', '0'))}"],
        )
        return {"upserted": 1}

    def upsert_paper_to_vectordb(self, paper: Dict[str, Any], collection_name: str = "papers") -> Dict[str, int]:
        """Upsert a single paper document into the papers collection."""
        if not paper:
            return {"upserted": 0}

        collection = self._get_or_create_collection(collection_name)
        self.papers_collection = collection

        paper_id = str(paper.get("paper_id", "")).strip()
        if not paper_id:
            return {"upserted": 0}

        document = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        metadata = {
            "paper_id": paper_id,
            "title": str(paper.get("title", "")),
            "authors": "|".join(paper.get("authors", []))[:500],
            "pmid": str(paper.get("pmid", "")),
            "doi": str(paper.get("doi", "")),
            "publication_date": str(paper.get("publication_date", "")),
            "upload_date": str(paper.get("upload_date", "")),
            "extraction_status": str(paper.get("extraction_status", "pending")),
        }
        collection.upsert(
            documents=[document],
            metadatas=[metadata],
            ids=[f"paper_{paper_id}"],
        )
        return {"upserted": 1}
    
    def load_papers_to_vectordb(self, papers_data: List[Dict[str, Any]], batch_size: int = 50) -> None:
        """
        Load paper documents into the vector database. Each paper is indexed with its metadata.
        
        Args:
            papers_data: List of paper dictionaries with title, abstract, pdf_path, etc.
            batch_size: Number of papers to process in each batch
        """
        if not hasattr(self, 'papers_collection'):
            self.create_papers_collection()
        
        total_papers = len(papers_data)
        if total_papers == 0:
            print("No papers to load.")
            return
        
        print(f"Loading {total_papers} papers into vector database...")
        processed_count = 0
        
        with tqdm(total=total_papers, desc="Loading papers", unit="papers") as pbar:
            for start_idx in range(0, total_papers, batch_size):
                end_idx = min(start_idx + batch_size, total_papers)
                batch = papers_data[start_idx:end_idx]
                
                documents = []
                metadatas = []
                ids = []
                
                for paper in batch:
                    # Create combined searchable text (title + abstract)
                    doc_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                    
                    # Create metadata
                    metadata = {
                        "paper_id": str(paper.get('paper_id', '')),
                        "title": str(paper.get('title', '')),
                        "authors": '|'.join(paper.get('authors', []))[:500],  # Limit string length
                        "pmid": str(paper.get('pmid', '')),
                        "doi": str(paper.get('doi', '')),
                        "publication_date": str(paper.get('publication_date', '')),
                        "upload_date": str(paper.get('upload_date', '')),
                        "extraction_status": str(paper.get('extraction_status', 'pending'))
                    }
                    
                    documents.append(doc_text)
                    metadatas.append(metadata)
                    ids.append(f"paper_{paper.get('paper_id', start_idx + len(documents))}")
                
                try:
                    self.papers_collection.upsert(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    processed_count += len(documents)
                    pbar.update(len(documents))
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Error loading papers batch {start_idx}-{end_idx}: {e}")
                    continue
        
        print(f"Successfully loaded {processed_count} papers into the vector database.")
    
    def search_papers(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for papers similar to a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of paper results with metadata
        """
        if not hasattr(self, 'papers_collection'):
            return []
        
        try:
            results = self.papers_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if results['distances'] else None
                    metadata = self._hydrate_paper_metadata(results['metadatas'][0][i])
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': metadata,
                        'distance': distance,
                        'similarity_score': self._distance_to_similarity(distance),
                        'paper_id': metadata.get('paper_id', ''),
                        'title': metadata.get('title', ''),
                        'authors': metadata.get('authors', []),
                        'pmid': metadata.get('pmid', ''),
                        'doi': metadata.get('doi', ''),
                        'publication_date': metadata.get('publication_date', ''),
                        'publication_year': metadata.get('publication_year', ''),
                        'abstract': metadata.get('abstract', ''),
                        'source_url': metadata.get('source_url', ''),
                    }
                    formatted_results.append(result)
            
            return formatted_results
        except Exception as e:
            print(f"Error searching papers: {e}")
            return []

    def upsert_paper_entities_to_vectordb(
        self,
        paper_id: str,
        entities: Dict[str, Any],
        collection_name: str = "paper_entities",
    ) -> Dict[str, int]:
        """Replace a single paper's entity documents in the entity collection."""
        collection = self._get_or_create_collection(collection_name)

        normalized_paper_id = str(paper_id).strip()
        if not normalized_paper_id:
            return {"upserted": 0}

        try:
            collection.delete(where={"paper_id": normalized_paper_id})
        except Exception:
            pass

        documents = []
        metadatas = []
        ids = []

        normalized_entities = normalize_entities_for_paper(normalized_paper_id, entities)

        for entity_type in ["genes", "proteins", "diseases", "pathways"]:
            for entity_index, entity in enumerate(normalized_entities.get(entity_type, [])):
                documents.append(self._build_paper_entity_document_text(normalized_paper_id, entity_type, entity, "PAPER_INGEST"))
                metadatas.append({
                    "paper_id": normalized_paper_id,
                    "entity_id": str(entity.get("id", "")),
                    "entity_name": str(entity.get("name", "")),
                    "entity_type": str(entity.get("entity_type", entity_type)),
                    "confidence": float(entity.get("confidence", 0.0)),
                    "approved": bool(entity.get("approved", False)),
                    "mapped_to_existing": str(entity.get("mapped_to_existing", "")),
                    "chromosome": str(entity.get("chromosome", "")),
                })
                ids.append(f"entity_{normalized_paper_id}_{entity_type}_{entity_index}")

        if ids:
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

        return {"upserted": len(ids)}
    
    def load_paper_entities_to_vectordb(self, paper_entities: Dict[str, Any], collection_name: str = "paper_entities") -> None:
        """
        Load extracted entities from papers into a separate collection for entity-specific search.
        
        Args:
            paper_entities: Dictionary mapping paper_id to extracted entities
            collection_name: Name of the entities collection
        """
        entities_collection = self._get_or_create_collection(collection_name)
        
        total_entities = sum(
            len(entities.get(etype, []))
            for entities in paper_entities.values()
            for etype in ['genes', 'proteins', 'diseases', 'pathways']
        )
        
        if total_entities == 0:
            print("No entities to load.")
            return
        
        print(f"Loading {total_entities} entities from papers...")
        processed_count = 0
        batch_size = 100
        documents = []
        metadatas = []
        ids = []
        
        with tqdm(total=total_entities, desc="Loading entities", unit="entities") as pbar:
            for paper_id, entities in paper_entities.items():
                normalized_entities = normalize_entities_for_paper(str(paper_id), entities)
                for entity_type in ['genes', 'proteins', 'diseases', 'pathways']:
                    for entity_index, entity in enumerate(normalized_entities.get(entity_type, [])):
                        doc_text = self._build_paper_entity_document_text(str(paper_id), entity_type, entity, "PAPER_INGEST")
                        
                        metadata = {
                            "paper_id": str(paper_id),
                            "entity_id": str(entity.get('id', '')),
                            "entity_name": str(entity.get('name', '')),
                            "entity_type": str(entity.get('entity_type', entity_type)),
                            "confidence": float(entity.get('confidence', 0.0)),
                            "approved": bool(entity.get('approved', False)),
                            "mapped_to_existing": str(entity.get('mapped_to_existing', '')),
                            "chromosome": str(entity.get('chromosome', '')),
                        }
                        
                        documents.append(doc_text)
                        metadatas.append(metadata)
                        ids.append(f"entity_{paper_id}_{entity_type}_{entity_index}")
                        
                        if len(documents) >= batch_size:
                            try:
                                entities_collection.upsert(
                                    documents=documents,
                                    metadatas=metadatas,
                                    ids=ids
                                )
                                processed_count += len(documents)
                                pbar.update(len(documents))
                                documents, metadatas, ids = [], [], []
                                time.sleep(0.01)
                            except Exception as e:
                                print(f"Error loading entity batch: {e}")
        
        # Load remaining documents
        if documents:
            try:
                entities_collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                processed_count += len(documents)
                pbar.update(len(documents))
            except Exception as e:
                print(f"Error loading final entity batch: {e}")
        
        print(f"Successfully loaded {processed_count} entities into the vector database.")

    def load_paper_edges_to_vectordb(
        self,
        paper_edges: List[Dict[str, Any]],
        collection_name: str = "paper_edges",
        only_approved: bool = True,
    ) -> None:
        """Load paper-derived relationships into the edge collection."""
        self.create_paper_edges_collection(collection_name)
        result = self.sync_paper_edges_to_vectordb(
            paper_edges=paper_edges,
            only_approved=only_approved,
            collection_name=collection_name,
        )
        print(f"Successfully loaded {result.get('upserted', 0)} paper relationships into the vector database.")

    def search_relationships(
        self,
        query: str,
        n_results: int = 10,
        edge_types: Optional[List[str]] = None,
        source_types: Optional[List[str]] = None,
        target_types: Optional[List[str]] = None,
        approved_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search indexed paper-derived relationships with optional metadata filters."""
        if not hasattr(self, "paper_edges_collection"):
            try:
                self.paper_edges_collection = self.client.get_collection(name="paper_edges")
            except Exception:
                return []

        try:
            results = self.paper_edges_collection.query(
                query_texts=[query],
                n_results=max(n_results * 3, n_results),
            )
        except Exception as e:
            print(f"Error searching paper relationships: {e}")
            return []

        edge_type_filter = self._normalize_label_filter(edge_types)
        source_type_filter = self._normalize_label_filter(source_types)
        target_type_filter = self._normalize_label_filter(target_types)

        formatted_results: List[Dict[str, Any]] = []
        if results.get("documents") and results["documents"][0]:
            for index in range(len(results["documents"][0])):
                metadata = dict(results["metadatas"][0][index])
                if approved_only and not bool(metadata.get("approved", False)):
                    continue

                edge_type = str(metadata.get("edge_type", "")).strip().lower()
                source_type = str(metadata.get("source_type", "")).strip().lower()
                target_type = str(metadata.get("target_type", "")).strip().lower()
                if edge_type_filter and edge_type not in edge_type_filter:
                    continue
                if source_type_filter and source_type not in source_type_filter:
                    continue
                if target_type_filter and target_type not in target_type_filter:
                    continue

                distance = results["distances"][0][index] if results.get("distances") else None
                formatted_results.append({
                    "document": results["documents"][0][index],
                    "metadata": metadata,
                    "distance": distance,
                    "similarity_score": self._distance_to_similarity(distance),
                    "source_name": metadata.get("source_name", ""),
                    "source_type": metadata.get("source_type", ""),
                    "target_name": metadata.get("target_name", ""),
                    "target_type": metadata.get("target_type", ""),
                    "edge_type": metadata.get("edge_type", ""),
                    "edge_weight": metadata.get("edge_weight", 0.0),
                    "evidence": metadata.get("evidence", ""),
                    "paper_id": metadata.get("paper_id", ""),
                })
                if len(formatted_results) >= n_results:
                    break
        return formatted_results

# Example usage
if __name__ == "__main__":
    # Initialize the vector database manager
    db_manager = VectorDBManager()
    
    # Load CSV data
    db_manager.load_csv_to_vectordb("nodes_main.csv")
    
    # Test search
    results = db_manager.search_similar("protein kinase", n_results=3)
    print("\nSearch results for 'protein kinase':")
    for result in results:
        print(f"- {result['document']}")
        print(f"  Distance: {result['distance']:.4f}")
    
    # Get database stats
    stats = db_manager.get_database_stats()
    print(f"\nDatabase Statistics:")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Sources: {stats['sources']}")
    print(f"Node types: {stats['node_types']}")
