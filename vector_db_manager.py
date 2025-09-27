import pandas as pd
import chromadb
from chromadb.config import Settings
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json
from tqdm import tqdm
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Enhanced imports for GPT-4 and PubMed integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Enhanced responses disabled.")

try:
    from citations import fetch_pubmed_citations, Citation
    CITATIONS_AVAILABLE = True
except ImportError:
    CITATIONS_AVAILABLE = False
    print("Warning: Citations module not available. Literature search disabled.")

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
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if OPENAI_AVAILABLE:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                print("OpenAI client initialized for enhanced responses.")
            else:
                print("Warning: OPENAI_API_KEY not found. Enhanced responses disabled.")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
    
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
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
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
                result = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
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
                'metadata': results['metadatas'][0][0]
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
        
        return {
            'total_documents': total_docs,
            'sources': sources,
            'node_types': node_types
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
            relevance = (1 - result['distance']) * 100 if result['distance'] else 100
            
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
        if not self.openai_client:
            return "Enhanced responses not available (OpenAI client not initialized)."
        
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
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
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
            'has_openai': self.openai_client is not None,
            'has_citations': CITATIONS_AVAILABLE
        }

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
