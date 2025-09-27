import streamlit as st
import pandas as pd
from vector_db_manager import VectorDBManager
from typing import List, Dict, Any
import time
import os
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from collections import defaultdict
import random
import tempfile
import os
os.environ["GENSIM_NO_CYTHON"] = "1"


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Knowledge Chat",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_citations' not in st.session_state:
    st.session_state.chat_citations = {}  # Store citations for each message
if 'db_loaded' not in st.session_state:
    st.session_state.db_loaded = False
if 'gene_network' not in st.session_state:
    st.session_state.gene_network = defaultdict(set)
if 'query_genes' not in st.session_state:
    st.session_state.query_genes = []

def initialize_database():
    """Initialize the vector database"""
    try:
        with st.spinner("Initializing vector database..."):
            st.session_state.db_manager = VectorDBManager()
            st.session_state.db_manager.load_csv_to_vectordb("nodes_main.csv")
            st.session_state.db_loaded = True
        st.success("Vector database initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        return False

def generate_enhanced_response(query: str, search_results: List[Dict[str, Any]], db_manager: VectorDBManager) -> Dict[str, Any]:
    """Generate enhanced response using GPT-4 and PubMed citations"""
    if not search_results:
        return {
            'response': "I couldn't find any relevant information in the gene/protein database for your query.",
            'citations': [],
            'has_enhanced': False
        }
    
    # Try to get enhanced response with GPT-4 and citations
    try:
        enhanced_result = db_manager.generate_enhanced_response(query, search_results, max_tokens=1024)
        
        
        return {
            'response': enhanced_result['gpt_response'],
            'citations': enhanced_result['citations'],
            'has_enhanced': enhanced_result['has_openai'],
            'has_citations': enhanced_result['has_citations'],
            'search_results': search_results
        }
    except Exception as e:
        # Fallback to basic response if enhanced fails
        st.error(f"Enhanced response failed, using basic mode: {str(e)}")
        print(f"DEBUG: Exception in generate_enhanced_response: {str(e)}")
        return generate_basic_response(query, search_results)

def generate_basic_response(query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate basic response (fallback when GPT-4 is not available)"""
    if not search_results:
        return {
            'response': "I couldn't find any relevant information in the gene/protein database for your query.",
            'citations': [],
            'has_enhanced': False
        }
    
    # Create a comprehensive response
    response = f"Based on your query about '{query}', I found the following relevant gene/protein information:\n\n"
    
    for i, result in enumerate(search_results, 1):
        metadata = result['metadata']
        distance = result.get('distance', 0)
        
        response += f"**{i}. {metadata['node_name']}**\n"
        response += f"   - **Gene/Protein ID**: {metadata['node_id']}\n"
        response += f"   - **Type**: {metadata['node_type']}\n"
        response += f"   - **Source**: {metadata['node_source']}\n"
        response += f"   - **Relevance Score**: {(1-distance)*100:.1f}%\n\n"
    
    # Add contextual information
    if len(search_results) > 1:
        response += f"I found {len(search_results)} relevant matches. "
    
    # Add suggestions for further queries
    gene_names = [result['metadata']['node_name'] for result in search_results]
    response += f"\n**Related genes/proteins you might want to ask about**: {', '.join(gene_names[:3])}"
    
    return {
        'response': response,
        'citations': [],
        'has_enhanced': False,
        'has_citations': False,
        'search_results': search_results
    }

def display_citations(citations):
    """Display PubMed citations in a formatted way"""
    if not citations:
        return
    
    st.markdown("### 📚 Supporting Literature")
    
    for i, citation in enumerate(citations, 1):
        with st.container():
            # Title with PMID link
            if citation.pmid:
                pmid_link = f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/"
                st.markdown(f"**{i}. [{citation.title}]({pmid_link})**")
            else:
                st.markdown(f"**{i}. {citation.title}**")
            
            # Authors and journal info
            if citation.authors:
                st.write(f"👥 **Authors:** {citation.authors}")
            
            if citation.journal and citation.year:
                st.write(f"📖 **Journal:** {citation.journal} ({citation.year})")
            elif citation.journal:
                st.write(f"📖 **Journal:** {citation.journal}")
            elif citation.year:
                st.write(f"📅 **Year:** {citation.year}")
            
            if citation.pmid:
                st.write(f"🔗 **PMID:** {citation.pmid}")
            
            if citation.is_review:
                st.markdown("🔬 **Review Article**")
            
def create_gene_network_graph(genes_data: List[Dict], query: str = ""):
    """Create an interactive network graph of related genes using pyvis"""
    if not genes_data:
        return None
    
    # Create pyvis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.force_atlas_2based()
    
    # Color scheme for different node types
    color_scheme = {
        "gene": "#1f77b4",      # Blue
        "protein": "#2ca02c",   # Green
        "gene/protein": "#ff7f0e",  # Orange
        "default": "#9467bd"    # Purple
    }
    
    # Add nodes for each gene
    for gene in genes_data:
        metadata = gene['metadata']
        gene_name = metadata['node_name']
        gene_id = metadata['node_id']
        gene_type = metadata['node_type']
        gene_source = metadata['node_source']
        relevance = (1 - gene.get('distance', 0)) * 100
        
        # Determine color based on type
        color = color_scheme.get(gene_type, color_scheme["default"])
        
        # Size based on relevance
        size = max(15, relevance / 2)
        
        # Enhanced tooltip
        title = (f"Gene: {gene_name}<br>"
                f"ID: {gene_id}<br>"
                f"Type: {gene_type}<br>"
                f"Source: {gene_source}<br>"
                f"Relevance: {relevance:.1f}%<br>"
                f"Query: {query}")
        
        net.add_node(
            gene_name,
            label=gene_name,
            color=color,
            title=title,
            size=size,
            font={'size': 12}
        )
    
    # Add edges between genes (connect all genes from same query)
    gene_names = [gene['metadata']['node_name'] for gene in genes_data]
    for i, gene1 in enumerate(gene_names):
        for gene2 in gene_names[i+1:]:
            # Edge width based on combined relevance
            gene1_relevance = (1 - genes_data[i].get('distance', 0)) * 100
            gene2_idx = next(j for j, g in enumerate(genes_data) if g['metadata']['node_name'] == gene2)
            gene2_relevance = (1 - genes_data[gene2_idx].get('distance', 0)) * 100
            
            edge_width = max(1, (gene1_relevance + gene2_relevance) / 50)
            
            net.add_edge(
                gene1, 
                gene2, 
                title=f"Co-occurrence in query: {query}",
                width=edge_width,
                color="#666666"
            )
    
    # Configure physics
    net.set_options("""
    var options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08,
                "damping": 0.4
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": true,
                "iterations": 1000
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "multiselect": true
        }
    }
    """)
    
    return net

def create_compound_gene_network_graph(selected_queries: List[str], query_genes_data: List[Dict]) -> Network:
    """Create a compound network graph combining genes from multiple queries"""
    if not selected_queries or not query_genes_data:
        return None
    
    # Create pyvis network
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    net.force_atlas_2based()
    
    # Color scheme for different queries and node types
    query_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange  
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf"   # Cyan
    ]
    
    # Track all genes and their query associations
    all_genes = {}  # gene_name -> {queries: set, data: gene_data, relevance: max_relevance}
    query_gene_map = {}  # query -> [gene_names]
    
    # Collect genes from selected queries
    for i, query in enumerate(selected_queries):
        query_gene_map[query] = []
        
        # Find genes for this query
        for query_data in query_genes_data:
            if isinstance(query_data, dict) and query_data.get('query') == query:
                genes = query_data.get('genes', [])
                
                for gene in genes:
                    metadata = gene['metadata']
                    gene_name = metadata['node_name']
                    relevance = (1 - gene.get('distance', 0)) * 100
                    
                    query_gene_map[query].append(gene_name)
                    
                    if gene_name not in all_genes:
                        all_genes[gene_name] = {
                            'queries': set(),
                            'data': gene,
                            'relevance': relevance
                        }
                    else:
                        # Update with higher relevance if found
                        all_genes[gene_name]['relevance'] = max(
                            all_genes[gene_name]['relevance'], 
                            relevance
                        )
                    
                    all_genes[gene_name]['queries'].add(query)
                break
    
    # Add nodes to the network
    for gene_name, gene_info in all_genes.items():
        metadata = gene_info['data']['metadata']
        relevance = gene_info['relevance']
        associated_queries = gene_info['queries']
        
        # Determine node color based on query association
        if len(associated_queries) > 1:
            # Multi-query gene - use a special color (gold)
            color = "#FFD700"
            border_color = "#FFA500"
            size = max(20, relevance / 2 + 10)  # Larger for multi-query genes
        else:
            # Single query gene - use query-specific color
            query = list(associated_queries)[0]
            query_index = selected_queries.index(query) % len(query_colors)
            color = query_colors[query_index]
            border_color = color
            size = max(15, relevance / 2)
        
        # Enhanced tooltip
        queries_text = "<br>".join([f"• {q}" for q in associated_queries])
        title = (f"Gene: {gene_name}<br>"
                f"ID: {metadata['node_id']}<br>"
                f"Type: {metadata['node_type']}<br>"
                f"Source: {metadata['node_source']}<br>"
                f"Max Relevance: {relevance:.1f}%<br>"
                f"Found in queries:<br>{queries_text}")
        
        net.add_node(
            gene_name,
            label=gene_name,
            color=color,
            title=title,
            size=size,
            font={'size': 12},
            borderWidth=2,
            borderWidthSelected=4,
            chosen={'node': True}
        )
    
    # Add edges
    # 1. Intra-query edges (genes from same query)
    for query, gene_names in query_gene_map.items():
        query_index = selected_queries.index(query) % len(query_colors)
        edge_color = query_colors[query_index]
        
        for i, gene1 in enumerate(gene_names):
            for gene2 in gene_names[i+1:]:
                if gene1 in all_genes and gene2 in all_genes:
                    # Edge width based on combined relevance
                    relevance1 = all_genes[gene1]['relevance']
                    relevance2 = all_genes[gene2]['relevance']
                    edge_width = max(1, (relevance1 + relevance2) / 50)
                    
                    net.add_edge(
                        gene1, 
                        gene2, 
                        title=f"Co-occurrence in: {query}",
                        width=edge_width,
                        color=edge_color,
                        alpha=0.7
                    )
    
    # 2. Inter-query edges (genes that appear in multiple queries)
    multi_query_genes = [name for name, info in all_genes.items() if len(info['queries']) > 1]
    for i, gene1 in enumerate(multi_query_genes):
        for gene2 in multi_query_genes[i+1:]:
            # Connect multi-query genes with special edges
            shared_queries = all_genes[gene1]['queries'].intersection(all_genes[gene2]['queries'])
            if shared_queries:
                net.add_edge(
                    gene1,
                    gene2,
                    title=f"Shared in: {', '.join(shared_queries)}",
                    width=3,
                    color="#FFD700",  # Gold for multi-query connections
                    dashes=True,
                    alpha=0.8
                )
    
    # Configure physics for better layout with more nodes
    net.set_options("""
    var options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.005,
                "springLength": 120,
                "springConstant": 0.05,
                "damping": 0.6
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": true,
                "iterations": 1500
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "multiselect": true,
            "selectConnectedEdges": false
        }
    }
    """)
    
    return net

def display_gene_network_tab():
    """Display the gene network visualization tab with compound network option"""
    st.header("🕸️ Gene Network Visualization")
    
    if not st.session_state.query_genes:
        st.info("🔍 No gene networks to display yet. Start by asking questions in the Chat tab to build gene relationships!")
        return
    
    # Extract query strings from the stored data
    query_options = []
    for query_data in st.session_state.query_genes:
        if isinstance(query_data, dict) and 'query' in query_data:
            query_options.append(query_data['query'])
    
    if not query_options:
        return
    
    # Network visualization mode selection
    st.subheader("🎛️ Network Visualization Mode")
    
    viz_mode = st.radio(
        "Choose visualization mode:",
        ["Single Query Network", "Compound Network (Multiple Queries)"],
        key="viz_mode_selector"
    )
    
    if viz_mode == "Single Query Network":
        # Original single query functionality
        st.subheader("Available Gene Networks")
        
        selected_query = st.selectbox(
            "Select a query to visualize its gene network:",
            options=query_options,
            key="single_network_query_selector"
        )
        
        if selected_query:
            # Find the corresponding genes data
            genes_data = None
            for query_data in st.session_state.query_genes:
                if isinstance(query_data, dict) and query_data.get('query') == selected_query:
                    genes_data = query_data.get('genes', [])
                    break
            
            if genes_data:
                # Create and display the network graph
                net = create_gene_network_graph(genes_data, selected_query)
                if net:
                    # Save the network to a temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                        net.save_graph(tmp_file.name)
                        
                        # Read the HTML content
                        with open(tmp_file.name, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        # Display the network
                        components.html(html_content, height=650)
                        
                        # Clean up temporary file
                        os.unlink(tmp_file.name)
                    
                    # Show gene details below the graph
                    st.subheader("📋 Genes in Network")
                    
                    cols = st.columns(min(3, len(genes_data)))
                    for i, gene in enumerate(genes_data):
                        with cols[i % 3]:
                            metadata = gene['metadata']
                            relevance = (1 - gene.get('distance', 0)) * 100
                            
                            st.markdown(f"""
                            **{metadata['node_name']}**
                            - ID: {metadata['node_id']}
                            - Type: {metadata['node_type']}
                            - Source: {metadata['node_source']}
                            - Relevance: {relevance:.1f}%
                            """)
                else:
                    st.error("Failed to create network graph")
            else:
                st.warning("No gene data found for selected query")
    
    else:  # Compound Network mode
        st.subheader("🔗 Compound Network Builder")
        
        # Multi-select for queries
        selected_queries = st.multiselect(
            "Select multiple queries to combine into one network:",
            options=query_options,
            default=query_options[:min(3, len(query_options))],  # Default to first 3 queries
            key="compound_network_query_selector",
            help="Select 2 or more queries to see how their gene networks interconnect"
        )
        
        if len(selected_queries) < 2:
            st.warning("⚠️ Please select at least 2 queries to create a compound network.")
        else:
            st.info(f"🔬 Creating compound network from {len(selected_queries)} queries...")
            
            # Create compound network
            compound_net = create_compound_gene_network_graph(selected_queries, st.session_state.query_genes)
            
            if compound_net:
                # Save the network to a temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                    compound_net.save_graph(tmp_file.name)
                    
                    # Read the HTML content
                    with open(tmp_file.name, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Display the network
                    components.html(html_content, height=700)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
                
                # Show legend and analysis
                st.subheader("🎨 Network Legend")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Node Colors:**")
                    st.markdown("🟡 **Gold**: Genes found in multiple queries")
                    st.markdown("🔵 **Colored**: Query-specific genes")
                    st.markdown("**Node Size**: Based on relevance score")
                
                with col2:
                    st.markdown("**Edge Types:**")
                    st.markdown("**Solid lines**: Genes from same query")
                    st.markdown("**Dashed gold lines**: Multi-query connections")
                    st.markdown("**Edge Width**: Based on combined relevance")
                
                # Compound network analysis
                st.subheader("🔍 Compound Network Analysis")
                
                # Collect analysis data
                all_genes = set()
                multi_query_genes = set()
                query_gene_counts = {}
                
                for query in selected_queries:
                    query_gene_counts[query] = 0
                    for query_data in st.session_state.query_genes:
                        if isinstance(query_data, dict) and query_data.get('query') == query:
                            genes = query_data.get('genes', [])
                            query_genes = set()
                            for gene in genes:
                                gene_name = gene['metadata']['node_name']
                                all_genes.add(gene_name)
                                query_genes.add(gene_name)
                            query_gene_counts[query] = len(query_genes)
                            
                            # Check for multi-query genes
                            for other_query in selected_queries:
                                if other_query != query:
                                    for other_query_data in st.session_state.query_genes:
                                        if isinstance(other_query_data, dict) and other_query_data.get('query') == other_query:
                                            other_genes = set(g['metadata']['node_name'] for g in other_query_data.get('genes', []))
                                            multi_query_genes.update(query_genes.intersection(other_genes))
                            break
                
                # Display analysis metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Unique Genes", len(all_genes))
                
                with col2:
                    st.metric("Multi-Query Genes", len(multi_query_genes))
                
                with col3:
                    overlap_percentage = (len(multi_query_genes) / len(all_genes) * 100) if all_genes else 0
                    st.metric("Overlap %", f"{overlap_percentage:.1f}%")
                
                with col4:
                    st.metric("Selected Queries", len(selected_queries))
                
                # Query breakdown
                if st.expander("📊 Query Breakdown", expanded=False):
                    for query, count in query_gene_counts.items():
                        st.write(f"**{query}**: {count} genes")
                
            else:
                st.error("Failed to create compound network graph")
    
    # Network statistics (common for both modes)
    if st.session_state.query_genes:
        st.markdown("---")
        st.subheader("📊 Overall Network Statistics")
        
        total_queries = len(st.session_state.query_genes)
        total_unique_genes = set()
        
        for query_data in st.session_state.query_genes:
            if isinstance(query_data, dict):
                genes = query_data.get('genes', [])
                for gene in genes:
                    total_unique_genes.add(gene['metadata']['node_name'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("Unique Genes", len(total_unique_genes))
        with col3:
            avg_genes = sum(len(q.get('genes', [])) for q in st.session_state.query_genes if isinstance(q, dict)) / max(1, total_queries)
            st.metric("Avg Genes/Query", f"{avg_genes:.1f}")

def main():
    st.title("🧬 Gene/Protein Knowledge Chat")
    st.markdown("Ask questions about genes and proteins from your dataset!")
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "🕸️ Gene Network"])
    
    # Sidebar
    with st.sidebar:
        st.header("Database Information")
        
        if not st.session_state.db_loaded:
            if st.button("Initialize Database", type="primary"):
                initialize_database()
        else:
            st.success("✅ Database Ready")
            
            # Show database statistics
            if st.session_state.db_manager:
                try:
                    stats = st.session_state.db_manager.get_database_stats()
                    st.metric("Total Genes/Proteins", stats['total_documents'])
                    
                    st.subheader("Data Sources")
                    for source, count in stats['sources'].items():
                        st.write(f"• {source}: {count}")
                    
                    st.subheader("Node Types")
                    for node_type, count in stats['node_types'].items():
                        st.write(f"• {node_type}: {count}")
                        
                except Exception as e:
                    st.error(f"Error loading stats: {str(e)}")
        
        st.markdown("---")
        st.subheader("Sample Queries")
        st.markdown("""
        Try asking:
        - "Tell me about MYC protein"
        - "What genes are related to kinase?"
        - "Show me information about BRCA2"
        - "Find proteins involved in transcription"
        - "What is EGFR?"
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.chat_citations = {}  # Clear citations
            st.session_state.query_genes = []  # Also clear network data
            st.rerun()
    
    # Tab 1: Chat Interface
    with tab1:
        # Main chat interface
        if not st.session_state.db_loaded:
            st.info("👈 Please initialize the database using the sidebar to start chatting!")
        else:
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for i, (role, message) in enumerate(st.session_state.chat_history):
                    if role == "user":
                        st.chat_message("user").write(message)
                    else:
                        with st.chat_message("assistant"):
                            st.write(message)
                            # Display citations if they exist for this message
                            message_key = f"message_{i}"
                            if message_key in st.session_state.chat_citations:
                                display_citations(st.session_state.chat_citations[message_key])
            
            # Chat input
            if prompt := st.chat_input("Ask about genes, proteins, or any related information..."):
                # Add user message to chat history
                st.session_state.chat_history.append(("user", prompt))
                
                # Display user message
                st.chat_message("user").write(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Searching database and generating enhanced response..."):
                        try:
                            # Search the vector database
                            search_results = st.session_state.db_manager.search_similar(prompt, n_results=5)
                            
                            # Store query and genes for network visualization
                            if search_results:
                                st.session_state.query_genes.append({
                                    'query': prompt,
                                    'genes': search_results
                                })
                            
                            # Generate enhanced response with GPT-4 and citations
                            result = generate_enhanced_response(prompt, search_results, st.session_state.db_manager)
                            
                            # Display main response
                            st.write(result['response'])
                            
                            # Display citations if available
                            if result.get('citations'):
                                display_citations(result['citations'])
                            
                            # Add status indicators
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if result.get('has_enhanced'):
                                    st.success("🤖 GPT-4 Enhanced")
                                else:
                                    st.info("📝 Basic Mode")
                            
                            with col2:
                                if result.get('has_citations') and result.get('citations'):
                                    st.success(f"📚 {len(result['citations'])} Citations")
                                elif result.get('has_citations'):
                                    st.warning("📚 No Citations Found")
                                else:
                                    st.info("📚 Citations Disabled")
                            
                            with col3:
                                # Show API key status
                                openai_key = "✅" if os.getenv("OPENAI_API_KEY") else "❌"
                                ncbi_key = "✅" if os.getenv("NCBI_API_KEY") else "❌"
                                st.info(f"🔑 OpenAI: {openai_key} NCBI: {ncbi_key}")
                            
                            # Add to chat history (just the main response for cleaner history)
                            st.session_state.chat_history.append(("assistant", result['response']))
                            
                            # Store citations for this message
                            if result.get('citations'):
                                message_key = f"message_{len(st.session_state.chat_history) - 1}"
                                st.session_state.chat_citations[message_key] = result['citations']
                            
                            # Show retrieved nodes in a dropdown
                            if search_results:
                                with st.expander("🧬 Retrieved Genes/Proteins from Database", expanded=False):
                                    # Store search results in session state to persist across reruns
                                    current_query_key = f"search_results_{len(st.session_state.chat_history)}"
                                    if current_query_key not in st.session_state:
                                        st.session_state[current_query_key] = search_results
                                    
                                    # Create a selectbox for the retrieved nodes
                                    node_options = []
                                    stored_results = st.session_state[current_query_key]
                                    
                                    for i, result_item in enumerate(stored_results):
                                        metadata = result_item['metadata']
                                        relevance = (1 - result_item.get('distance', 0)) * 100
                                        option_text = f"{metadata['node_name']} (ID: {metadata['node_id']}) - {relevance:.1f}% relevance"
                                        node_options.append(option_text)
                                    
                                    if node_options:
                                        # Use a unique key that persists
                                        selector_key = f"node_selector_{current_query_key}"
                                        
                                        selected_node = st.selectbox(
                                            "Select a gene/protein to view details:",
                                            options=["Select a gene/protein..."] + node_options,
                                            key=selector_key
                                        )
                                        
                                        if selected_node != "Select a gene/protein...":
                                            # Find the selected result
                                            selected_index = node_options.index(selected_node)
                                            selected_result = stored_results[selected_index]
                                            metadata = selected_result['metadata']
                                            
                                            # Display detailed information in a container that won't cause rerun
                                            with st.container():
                                                st.markdown("### 📋 Gene/Protein Details")
                                                col1, col2 = st.columns(2)
                                                
                                                with col1:
                                                    st.write(f"**Name:** {metadata['node_name']}")
                                                    st.write(f"**Node ID:** {metadata['node_id']}")
                                                    st.write(f"**Type:** {metadata['node_type']}")
                                                
                                                with col2:
                                                    st.write(f"**Source:** {metadata['node_source']}")
                                                    relevance = (1 - selected_result.get('distance', 0)) * 100
                                                    st.write(f"**Relevance:** {relevance:.1f}%")
                                                
                                                st.write(f"**Full Document:** {selected_result['document']}")
                                                
                                                # Add a small note
                                                st.info("💡 This information is retrieved from your local gene/protein database.")
                                
                                # Also show detailed search results in a separate expander
                                with st.expander("🔍 Raw Search Results (Technical)", expanded=False):
                                    for i, result_item in enumerate(search_results, 1):
                                        st.write(f"**Result {i}:**")
                                        st.write(f"Document: {result_item['document']}")
                                        st.write(f"Similarity Score: {(1-result_item.get('distance', 0))*100:.2f}%")
                                        st.json(result_item['metadata'])
                                        st.markdown("---")
                        
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error while searching: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append(("assistant", error_msg))

            # Additional features
            st.markdown("---")
            
            # Quick actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🎲 Random Gene Info"):
                    if st.session_state.db_manager:
                        try:
                            gene_names = st.session_state.db_manager.get_all_gene_names()
                            import random
                            random_gene = random.choice(gene_names)
                            gene_info = st.session_state.db_manager.get_gene_info(random_gene)
                            if gene_info:
                                st.info(f"**Random Gene**: {gene_info['document']}")
                        except Exception as e:
                            st.error(f"Error getting random gene: {str(e)}")
            
            with col2:
                if st.button("📊 Database Stats"):
                    if st.session_state.db_manager:
                        try:
                            stats = st.session_state.db_manager.get_database_stats()
                            st.json(stats)
                        except Exception as e:
                            st.error(f"Error loading stats: {str(e)}")
            
            with col3:
                if st.button("📋 All Gene Names"):
                    if st.session_state.db_manager:
                        try:
                            gene_names = st.session_state.db_manager.get_all_gene_names()
                            st.write(f"**Total Genes**: {len(gene_names)}")
                            st.write(", ".join(gene_names[:20]) + ("..." if len(gene_names) > 20 else ""))
                        except Exception as e:
                            st.error(f"Error loading gene names: {str(e)}")
    
    # Tab 2: Gene Network Visualization
    with tab2:
        display_gene_network_tab()

if __name__ == "__main__":
    main()
