import streamlit as st
import json
import math
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import streamlit as st
from itertools import combinations
# import matplotlib

from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Network_stats import *
# Set page config must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Biomedical Knowledge Graph")
st.title("Biomedical Knowledge Graph Visualization")
# Load data from JSON file
def load_data(file_path="data_unique.json"):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data["nodes"], data["edges"], data["clusters"]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Load the data
nodes_data, edges_data, clusters_data = load_data()

# Color scheme with additional types from your data
color_scheme = {
    "gene": "#1f77b4",      # Blue
    "gene group": "#aec7e8", # Light Blue
    "protein": "#2ca02c",   # Green
    "protein complex": "#98df8a",  # Light Green
    "protein fragment": "#ff9896", # Light Red
    "disease": "#d62728",   # Red
    "pathway": "#9467bd",   # Purple
    "process": "#c5b0d5",   # Light Purple
    "cell line": "#ff7f0e", # Orange
    "cell line clone": "#e377c2",  # Pink
    "cell population": "#8c564b",  # Brown
    "reagent": "#17becf",   # Cyan
    "chemical": "#bcbd22",  # Yellow-green
    "method": "#7f7f7f",    # Gray
    "organelle": "#9edae5", # Light blue
    "biological process": "#c49c94",  # Brown-red
    "phenotype": "#dbdb8d",  # Light yellow
    "type": "#c7c7c7"      # Default gray
}

EDGE_STYLE_MAP = {
    "ENCODES": {"color": "#1f77b4", "width_factor": 1.25, "dashes": False},
    "EXPRESSES": {"color": "#17becf", "width_factor": 1.15, "dashes": False},
    "INTERACTS": {"color": "#7f7f7f", "width_factor": 1.0, "dashes": True},
    "TREATS": {"color": "#2ca02c", "width_factor": 1.2, "dashes": False},
    "PARTICIPATES": {"color": "#9467bd", "width_factor": 1.1, "dashes": False},
    "ACTIVATES": {"color": "#ff7f0e", "width_factor": 1.35, "dashes": False},
    "INHIBITS": {"color": "#d62728", "width_factor": 1.35, "dashes": True},
    "REGULATES": {"color": "#8c564b", "width_factor": 1.15, "dashes": False},
    "ASSOCIATES": {"color": "#6b7280", "width_factor": 0.95, "dashes": True},
}

EDGE_FALLBACK_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Utility functions
def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def rgb_to_rgba(rgb, alpha):
    """Convert RGB tuple to RGBA string."""
    return f"rgba{rgb + (alpha,)}"


def normalize_relation_label(relation):
    """Normalize relation labels for consistent styling."""
    normalized = str(relation or "ASSOCIATES").strip().upper()
    return normalized or "ASSOCIATES"


def _fallback_edge_color(relation_label):
    idx = abs(hash(relation_label)) % len(EDGE_FALLBACK_PALETTE)
    return EDGE_FALLBACK_PALETTE[idx]


def get_edge_style(relation, score=1.0):
    """Return style attributes for an edge relation."""
    relation_label = normalize_relation_label(relation)
    base_style = EDGE_STYLE_MAP.get(
        relation_label,
        {"color": _fallback_edge_color(relation_label), "width_factor": 1.0, "dashes": False},
    )
    safe_score = max(0.1, float(score or 0.1))
    width = max(1.0, safe_score * 3 * base_style["width_factor"])
    return {
        "relation": relation_label,
        "color": base_style["color"],
        "width": width,
        "dashes": bool(base_style["dashes"]),
    }


def apply_alpha_to_color(color, alpha):
    """Apply alpha to a hex color, fallback unchanged for non-hex input."""
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        return rgb_to_rgba(hex_to_rgb(color), alpha)
    return color

def safe_read_file(filename):
    """Safely read file with error handling."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def validate_data():
    """Validate data structure and content."""
    try:
        if not nodes_data or not edges_data:
            st.error("Network data is missing or empty")
            return False
        
        # Validate node references in edges
        node_ids = {node["id"] for node in nodes_data}
        invalid_edges = []
        for edge in edges_data:
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                invalid_edges.append(edge)
        
        if invalid_edges:
            st.warning(f"Found {len(invalid_edges)} edges with invalid node references")
        
        # Check for required node attributes
        required_attrs = ["id", "type", "cluster", "size"]
        invalid_nodes = []
        for node in nodes_data:
            if not all(attr in node for attr in required_attrs):
                invalid_nodes.append(node["id"])
        
        if invalid_nodes:
            st.warning(f"Found nodes missing required attributes: {', '.join(invalid_nodes)}")
        
        # Validate color scheme
        node_types = {node["type"] for node in nodes_data}
        missing_colors = node_types - set(color_scheme.keys())
        if missing_colors:
            st.error(f"Missing colors for node types: {missing_colors}")
            return False
        
        return True
    except Exception as e:
        st.error(f"Error validating data: {str(e)}")
        return False

def handle_large_network():
    """Handle performance issues with large networks."""
    if len(nodes_data) > 1000 or len(edges_data) > 5000:
        st.warning("Large network detected. This may affect performance.")
        with st.expander("Performance Tips"):
            st.markdown("""
            - Consider filtering nodes by cluster
            - Reduce number of displayed edges
            - Use search instead of visual navigation
            """)

def get_node_relationships(node_ids):
    """Get all relationships for specified nodes."""
    relationships = []
    for edge in edges_data:
        if edge["source"] in node_ids or edge["target"] in node_ids:
            relationships.append({
                "source": edge["source"],
                "target": edge["target"],
                "relation": edge["relation"],
                "score": edge["score"]
            })
    return relationships

def create_network(selected_cluster=None):
    """Create and configure the network visualization."""
    try:
        net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
        net.force_atlas_2based()

        # Add nodes
        for node in nodes_data:
            color = color_scheme.get(node["type"], color_scheme["type"])
            base_size = node["size"] * 10
            
            if selected_cluster:
                is_in_cluster = node["cluster"] == selected_cluster
                if is_in_cluster:
                    size = base_size * 1.5
                else:
                    size = base_size
                    rgb = hex_to_rgb(color)
                    color = f"rgba{rgb + (0.15,)}"
            else:
                size = base_size

            # Enhanced tooltip
            mentions = round(math.exp(node['size']))
            title = (f"Node: {node['id']}<br>"
                    f"Cluster: {node['cluster']}<br>"
                    f"Mentions: {mentions}<br>"
                    f"PMID: {node.get('PMID', 'N/A')}")

            net.add_node(
                node["id"],
                label=node["id"],
                color=color,
                title=title,
                size=size
            )

        # Add edges
        for edge in edges_data:
            edge_style = get_edge_style(edge.get("relation", "ASSOCIATES"), edge.get("score", 1.0))
            edge_color = edge_style["color"]
            edge_width = edge_style["width"]
            edge_dashes = edge_style["dashes"]

            if selected_cluster:
                source_node = next((n for n in nodes_data if n["id"] == edge["source"]), None)
                target_node = next((n for n in nodes_data if n["id"] == edge["target"]), None)
                
                if source_node and target_node:
                    source_in_cluster = source_node["cluster"] == selected_cluster
                    target_in_cluster = target_node["cluster"] == selected_cluster

                    if source_in_cluster or target_in_cluster:
                        edge_width = edge_width * 1.2
                    else:
                        edge_color = apply_alpha_to_color(edge_color, 0.16)
                        edge_width = max(1.0, edge_width * 0.7)

            net.add_edge(
                edge["source"],
                edge["target"],
                title=f"Relation: {edge['relation']}<br>Score: {edge['score']:.2f}",
                width=edge_width,
                color=edge_color,
                dashes=edge_dashes
            )

        # Physics options
        net.set_options("""
        var options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.015,
                    "springLength": 150,
                    "springConstant": 0.1,
                    "damping": 0.95
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
    except Exception as e:
        st.error(f"Error creating network: {str(e)}")
        return None


def display_network_stats(nodes_data, edges_data, selected_cluster):
    """Display network statistics in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.title("Network Statistics")

    total_nodes = len(nodes_data)
    total_edges = len(edges_data)
    st.sidebar.write(f"Total Nodes: {total_nodes}")
    st.sidebar.write(f"Total Edges: {total_edges}")

    # Node type counting
    node_types_count = {}
    for node in nodes_data:
        if node["type"] in node_types_count:
            node_types_count[node["type"]] += 1
        else:
            node_types_count[node["type"]] = 1

    st.sidebar.write("\nNode Types:")
    for node_type, count in node_types_count.items():
        percentage = (count / total_nodes) * 100
        st.sidebar.write(f"{node_type.capitalize()}: {count} ({percentage:.1f}%)")
def display_paper_analysis_tab():
    st.header("Scientific Publication Network Insights")
    
    # Count nodes per PMID
    pmid_counts = defaultdict(int)
    for node in nodes_data:
        pmid = str(node.get('PMID', 'Unknown'))
        pmid_counts[pmid] += 1
    
    # Create DataFrame
    df = pd.DataFrame(list(pmid_counts.items()), columns=['PMID', 'Node Count'])
    df = df.sort_values('Node Count', ascending=False)
    
    # Display metrics without relying on paper_results dictionary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers", len(pmid_counts))
    with col2:
        st.metric("Average Nodes per Paper", f"{df['Node Count'].mean():.1f}")
    with col3:
        st.metric("Max Nodes in Paper", df['Node Count'].max())
    
    # Create distribution plot
    fig = go.Figure(data=[
        go.Bar(x=df['PMID'], y=df['Node Count'], name='Nodes per Paper')
    ])
    
    fig.update_layout(
        title='Distribution of Nodes Across Papers',
        xaxis_title='Paper PMID',
        yaxis_title='Number of Nodes',
        height=500
    )
    
    st.plotly_chart(fig)
    
    # Option to show raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(df)
def add_search_functionality():
    """Add search box for finding specific nodes."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Search")
    search_term = st.sidebar.text_input("Search Nodes", "").strip()
    
    if search_term:
        matching_nodes = [node["id"] for node in nodes_data 
                         if search_term.lower() in node["id"].lower()]
        if matching_nodes:
            node_to_highlight = st.sidebar.selectbox(
                "Matching nodes:", matching_nodes
            )
            if node_to_highlight:
                node_data = next((n for n in nodes_data if n["id"]== node_to_highlight), None)
                if node_data:
                    st.sidebar.markdown(f"**Type:** {node_data['type']}")
                    st.sidebar.markdown(f"**Cluster:** {node_data['cluster']}")
                    
                    # Option to add to selection
                    if st.sidebar.button("Add to Selection"):
                        st.session_state.selected_nodes.add(node_to_highlight)
                        st.rerun()

def handle_selected_nodes():
    """Handle data download for multiple selected nodes."""
    selected_nodes = st.session_state.get('selected_nodes', set())
    
    if selected_nodes:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Selected Nodes")
        
        # Clear selection button
        if st.sidebar.button("Clear Selection"):
            st.session_state.selected_nodes = set()
            st.rerun()
        
        st.sidebar.markdown(f"**Selected ({len(selected_nodes)}):** {', '.join(selected_nodes)}")
        
        # Collect node and relationship data
        selected_node_data = []
        selected_relationships = []
        
        # Get node data and relationships
        for node_id in selected_nodes:
            # Get node data
            node_data = next((n for n in nodes_data if n["id"] == node_id), None)
            if node_data:
                selected_node_data.append(node_data)
            
            # Get relationships where either source or target is in selected nodes
            for edge in edges_data:
                if edge["source"] == node_id or edge["target"] == node_id:
                    if edge["source"] in selected_nodes and edge["target"] in selected_nodes:
                        selected_relationships.append(edge)
        
        # Create subgraph visualization for selected nodes
        if st.sidebar.button("Create Selected Nodes Network"):
            with st.spinner("Creating network of selected nodes..."):
                subnet = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
                subnet.force_atlas_2based()
                
                # Add selected nodes
                for node in selected_node_data:
                    color = color_scheme[node["type"]]
                    subnet.add_node(
                        node["id"],
                        label=node["id"],
                        color=color,
                        title=f"Type: {node['type']}<br>Cluster: {node['cluster']}<br>Size: {node['size']:.2f}",
                        size=node["size"] * 10
                    )
                
                # Add edges between selected nodes
                for edge in selected_relationships:
                    edge_style = get_edge_style(edge.get("relation", "ASSOCIATES"), edge.get("score", 1.0))
                    subnet.add_edge(
                        edge["source"],
                        edge["target"],
                        title=f"Relation: {edge['relation']}<br>Score: {edge['score']:.2f}",
                        width=edge_style["width"],
                        color=edge_style["color"],
                        dashes=edge_style["dashes"]
                    )
                
                subnet.save_graph("selected_network.html")
                with open("selected_network.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                    
                st.sidebar.download_button(
                    label="Download Selected Network HTML",
                    data=html_content,
                    file_name="selected_nodes_network.html",
                    mime="text/html"
                )
        
        # Prepare textual data for download
        text_data = {
            "nodes": selected_node_data,
            "relationships": selected_relationships,
            "summary": {
                "total_selected_nodes": len(selected_nodes),
                "node_types": {},
                "total_relationships": len(selected_relationships)
            }
        }
        
        # Count node types
        for node in selected_node_data:
            node_type = node["type"]
            if node_type in text_data["summary"]["node_types"]:
                text_data["summary"]["node_types"][node_type] += 1
            else:
                text_data["summary"]["node_types"][node_type] = 1
        
        # Create JSON download button
        json_data = json.dumps(text_data, indent=2)
        st.sidebar.download_button(
            label="Download Selected Nodes Data (JSON)",
            data=json_data,
            file_name="selected_nodes_data.json",
            mime="application/json"
        )
        
        # Create readable text format
        text_content = f"""Selected Nodes Analysis
----------------------------
Total Nodes: {len(selected_nodes)}
Total Relationships: {len(selected_relationships)}

Node Type Distribution:
{chr(10).join(f'- {type_}: {count}' for type_, count in text_data['summary']['node_types'].items())}

Detailed Node Information:
{chr(10).join(f'- {node["id"]} ({node["type"]}): Cluster={node["cluster"]}, Size={node["size"]:.2f}' for node in selected_node_data)}

Relationships:
{chr(10).join(f'- {rel["source"]} -> {rel["target"]}: {rel["relation"]} (Score: {rel["score"]:.2f})' for rel in selected_relationships)}
"""
        
        # Create text download button
        st.sidebar.download_button(
            label="Download Selected Nodes Report (TXT)",
            data=text_content,
            file_name="selected_nodes_report.txt",
            mime="text/plain"
        )

def add_help_section():
    """Add help and instructions section."""
    with st.sidebar.expander("Help & Instructions"):
        st.markdown("""
        **Navigation Tips:**
        - Use Ctrl+Click to select multiple nodes
        - Use the search box to find specific nodes
        - Download selected node data from the sidebar
        - Clear selection to start over
        
        """)

def initialize_session_state():
    """Initialize session state variables."""
    if 'selected_nodes' not in st.session_state:
        st.session_state.selected_nodes = set()
    if 'last_cluster' not in st.session_state:
        st.session_state.last_cluster = "All"
def main():
    """Main application function."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Load and clean data
        global nodes_data, edges_data, clusters_data
        nodes_data, edges_data, clusters_data = load_data()
        
        if not nodes_data or not edges_data or not clusters_data:
            st.error("Failed to load data")
            return
            
        # Data validation reporting
        valid_node_ids = {node["id"] for node in nodes_data}
        invalid_edges = [
            edge for edge in edges_data 
            if edge["source"] not in valid_node_ids or edge["target"] not in valid_node_ids
        ]
        if invalid_edges:
            st.warning(f"Found {len(invalid_edges)} edges with invalid node references")
            with st.expander("Show invalid edges"):
                for edge in invalid_edges:
                    st.write(f"- {edge['source']} -> {edge['target']}: {edge['relation']}")
        
        # Remove invalid edges
        edges_data = [
            edge for edge in edges_data 
            if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
        ]
        
        # Check network size and show performance tips if needed
        handle_large_network()
        
        # Initialize network analyzer
        analyzer = NetworkAnalyzer(nodes_data, edges_data)

        # Sidebar organization
        with st.sidebar:
            st.title("Network Navigation")
            
            # Cluster selection
            selected_cluster = st.selectbox(
                "Select Cluster to Highlight",
                ["All"] + list(clusters_data.keys())
            )

            # Help section
            add_help_section()

            # Search functionality
            add_search_functionality()

            # Handle selected nodes
            handle_selected_nodes()

            # Display network statistics
            # display_network_stats(nodes_data, edges_data, selected_cluster)

            # Export options
            st.markdown("---")
            st.subheader("Export Options")

            # Export full network
            if st.button("Export Full Network"):
                with st.spinner("Preparing full network..."):
                    full_net = create_network()
                    if full_net:
                        full_net.save_graph("network_export_full.html")
                        content = safe_read_file("network_export_full.html")
                        if content:
                            st.download_button(
                                label="Download Full Network",
                                data=content,
                                file_name="full_network.html",
                                mime="text/html"
                            )

            # Export cluster
            if selected_cluster != "All":
                if st.button("Export Selected Cluster"):
                    with st.spinner(f"Preparing {selected_cluster} cluster..."):
                        cluster_net = create_network(selected_cluster)
                        if cluster_net:
                            cluster_net.save_graph("network_export_cluster.html")
                            content = safe_read_file("network_export_cluster.html")
                            if content:
                                st.download_button(
                                    label="Download Cluster Network",
                                    data=content,
                                    file_name=f"{selected_cluster.lower()}_network.html",
                                    mime="text/html"
                                )

            # Export statistics
            if st.button("Export Statistics as JSON"):
                with st.spinner("Preparing statistics..."):
                    export_stats = (
                        analyzer.get_cluster_stats(selected_cluster)
                        if selected_cluster != "All"
                        else analyzer.get_basic_stats()
                    )
                    json_stats = json.dumps(export_stats, indent=2)
                    st.download_button(
                        label="Download Statistics",
                        data=json_stats,
                        file_name="network_statistics.json",
                        mime="application/json"
                    )
        
        # Main content area
        main_tab, stats_tab = st.tabs(["Network Visualization", "Detailed Analysis"])

        with main_tab:
            
            
            # Create and display network
            # with st.spinner("Loading network visualization..."):
            net = create_network(None if selected_cluster == "All" else selected_cluster)
            if net:
                # Add JavaScript for multi-node selection
                net.html = net.html.replace('</head>', '''
                    <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        network.on("click", function(params) {
                            if (params.nodes.length > 0 && params.event.srcEvent.ctrlKey) {
                                window.parent.postMessage({
                                    type: "nodes_selected",
                                    nodes: params.nodes
                                }, "*");
                            }
                        });
                    });
                    </script>
                    </head>
                ''')
                
                # Save and display network
                net.save_graph("network.html")
                content = safe_read_file("network.html")
                if content:
                    components.html(content, height=800)
                    st.success("Network visualization loaded successfully")

                # Display legend
                st.write("\n### Node Type Legend")
                cols = st.columns(4)
                for i, (node_type, color) in enumerate(color_scheme.items()):
                    with cols[i % 4]:
                        st.markdown(
                            f'<div style="display: flex; align-items: center;">'
                            f'<div style="width: 20px; height: 20px; background-color: {color}; '
                            f'margin-right: 10px; border-radius: 50%;"></div>'
                            f'<span style="font-weight: 500;">{node_type.capitalize()}</span></div>',
                            unsafe_allow_html=True
                        )

                st.write("\n### Edge Type Legend")
                edge_type_counts = Counter(
                    normalize_relation_label(edge.get("relation", "ASSOCIATES"))
                    for edge in edges_data
                )
                edge_types = sorted(edge_type_counts.keys())
                edge_cols = st.columns(3)
                for i, edge_type in enumerate(edge_types):
                    style = get_edge_style(edge_type, score=1.0)
                    dash_css = "2px dashed" if style["dashes"] else "3px solid"
                    with edge_cols[i % 3]:
                        st.markdown(
                            f'<div style="display: flex; align-items: center; margin: 6px 0;">'
                            f'<div style="width: 36px; border-top: {dash_css} {style["color"]}; margin-right: 10px;"></div>'
                            f'<span style="font-weight: 500;">{edge_type}</span>'
                            f'<span style="margin-left: 6px; color: #6b7280;">({edge_type_counts[edge_type]})</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # with stats_tab:
        #     # Display detailed network statistics and analysis
        #     analyzer.display_stats_streamlit(selected_cluster)
        with stats_tab:
            # Initialize analyzer if not already done
            analyzer = NetworkAnalyzer(nodes_data, edges_data)
            
            # Create three analysis subtabs
            basic_tab, advanced_tab, explorer_tab = st.tabs([
                "Basic Analysis", 
                "Advanced Analysis",
                "Network Explorer"
            ])
        
            with basic_tab:
                st.header("Basic Network Analysis")
                
                # Get and display basic stats
                basic_stats = analyzer.basic_analyzer.get_basic_stats()
                
                # Network Overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Nodes", basic_stats["Total Nodes"])
                    st.metric("Average Degree", f"{basic_stats['Average Degree']:.2f}")
                with col2:
                    st.metric("Total Edges", basic_stats["Total Edges"])
                    st.metric("Network Density", f"{basic_stats['Network Density']:.3f}")
                with col3:
                    st.metric("Connected Components", basic_stats["Connected Components"])
                    st.metric("Average Path Length", f"{basic_stats['Average Path Length']:.2f}")
                with col4:
                    st.metric("Clustering Coefficient", f"{basic_stats['Average Clustering Coefficient']:.3f}")
                    st.metric("Graph Diameter", basic_stats["Graph Diameter"])
        
                # Component Analysis
                if basic_stats["Connected Components"] > 1:
                    st.subheader("Component Analysis")
                    st.write(f"Largest Component Size: {basic_stats['Largest Component Size']} nodes "
                            f"({basic_stats['Largest Component Ratio']:.1%} of network)")
        
                # Display node type distribution
                if any(key.endswith("Count") for key in basic_stats):
                    st.subheader("Node Type Distribution")
                    type_counts = {k: v for k, v in basic_stats.items() if k.endswith("Count")}
                    st.bar_chart(type_counts)
        
                # Pathway Analysis section
                pathway_stats = analyzer.basic_analyzer.get_pathway_analysis()
                if pathway_stats:
                    st.subheader("Pathway Analysis")
                    
                    for pathway, stats in pathway_stats.items():
                        with st.expander(f"Pathway: {pathway}"):
                            cols = st.columns(3)
                            with cols[0]:
                                st.metric("Total Connections", stats["Total Connections"])
                            with cols[1]:
                                st.metric("Average Interaction", f"{stats['Average Interaction Strength']:.2f}")
                            with cols[2]:
                                st.metric("Max Interaction", f"{stats['Max Interaction Strength']:.2f}")
                            
                            # Connected Types Distribution
                            st.write("Connected Node Types")
                            st.bar_chart(stats["Connected Types"])
        
            with advanced_tab:
                st.header("Advanced Network Analysis")
                
                # Tabless layout for advanced analysis types
                analysis_type = st.radio(
                    "Select Analysis Type",
                    ["Community Structure", "Network Resilience", "Network Embedding", 
                     "Centrality Analysis", "Network Entropy"]
                )
        
                if analysis_type == "Community Structure":
                    community_results = analyzer.advanced_analyzer.community_structure_analysis()
                    
                    # Display community detection results
                    for method, results in community_results['communities'].items():
                        st.subheader(f"{method} Communities")
                        st.metric("Modularity Score", f"{community_results['modularity_scores'][method]:.3f}")
                        
                        # Display visualizations
                        if method in community_results['visualizations']:
                            for viz_name, viz in community_results['visualizations'][method].items():
                                st.plotly_chart(viz)
        
                elif analysis_type == "Network Resilience":
                    resilience_results = analyzer.advanced_analyzer.network_resilience_analysis()
                    st.plotly_chart(resilience_results['visualization'])
                    
                    # Display resilience metrics for each attack strategy
                    for strategy, results in resilience_results.items():
                        if strategy != 'visualization':
                            st.subheader(f"{strategy} Results")
                            st.line_chart(results['largest_component_ratio'])
        
                elif analysis_type == "Network Embedding":
                    embedding_methods = ["node2vec", "tsne", "pca"]
                    selected_method = st.selectbox("Select Embedding Method", embedding_methods)
                    
                    embedding_results = analyzer.advanced_analyzer.network_embedding(method=selected_method)
                    st.plotly_chart(embedding_results['visualization'])
        
                elif analysis_type == "Centrality Analysis":
                    centrality_results = analyzer.advanced_analyzer.advanced_centrality_analysis()
                    
                    # Display centrality visualizations
                    st.plotly_chart(centrality_results['centrality_boxplot'])
                    st.plotly_chart(centrality_results['correlation_heatmap'])
                    
                    # Display top nodes by each centrality measure
                    st.subheader("Top Nodes by Centrality")
                    for measure, nodes in centrality_results['top_nodes'].items():
                        with st.expander(f"Top {measure} Nodes"):
                            for node, score in nodes:
                                st.write(f"{node}: {score:.3f}")
        
                elif analysis_type == "Network Entropy":
                    entropy_results = analyzer.advanced_analyzer.network_entropy_analysis()
                    
                    # Display entropy metrics
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Degree Entropy", f"{entropy_results['degree_entropy']:.3f}")
                    with cols[1]:
                        st.metric("Clustering Entropy", f"{entropy_results['clustering_entropy']:.3f}")
                    with cols[2]:
                        avg_centrality_entropy = np.mean(list(entropy_results['centrality_entropies'].values()))
                        st.metric("Avg Centrality Entropy", f"{avg_centrality_entropy:.3f}")
                    
                    # Display entropy visualizations
                    for viz_name, viz in entropy_results['visualizations'].items():
                        st.plotly_chart(viz)

            
            with explorer_tab:
                st.header("Scientific Literature Network Explorer")
                
                # Sidebar for Analysis Controls
                st.sidebar.header("Network Exploration Options")
                analysis_type = st.sidebar.radio(
                    "Select Analysis Type",
                    [
                        "Publication Network Overview",
                        
                        "Research Cluster Analysis",
                        "Temporal Research Dynamics",
                        "Publication Relationship Mapping",
                        "Node Exploration",
                        "Filtered Network View",
                        "Network Entropy Analysis"
                    ]
                )

                # Comprehensive Error Handling Wrapper
                def safe_analysis(analysis_function, error_message):
                    try:
                        return analysis_function()
                    except Exception as e:
                        st.error(f"{error_message}: {e}")
                        st.exception(e)
                        return None

                

                # Publication Network Overview
                if analysis_type == "Publication Network Overview":
                    st.subheader("Publication Distribution")
                    pmid_stats = analyzer.get_pmid_distribution()
                    pmid_df = pd.DataFrame([
                        {"PMID": pmid, "Node Count": data["count"]}
                        for pmid, data in pmid_stats.items()
                    ]).sort_values("Node Count", ascending=False)
                    
                    st.write("Nodes per Publication:")
                    st.bar_chart(data=pmid_df.set_index("PMID")["Node Count"])
                    pmid_counts = defaultdict(int)
                    for node in nodes_data:
                        pmid = str(node.get('PMID', 'Unknown'))
                        pmid_counts[pmid] += 1
                    
                    # Convert to DataFrame and sort
                    df = pd.DataFrame(
                        [(pmid, count) for pmid, count in pmid_counts.items()],
                        columns=['PMID', 'Number of Nodes']
                    ).sort_values('Number of Nodes', ascending=False)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Papers", len(pmid_counts))
                    with col2:
                        st.metric("Average Nodes per Paper", f"{df['Number of Nodes'].mean():.1f}")
                    with col3:
                        st.metric("Max Nodes in Paper", df['Number of Nodes'].max())

                    if st.checkbox("Show Raw Data"):
                        st.dataframe(df)
                
                    

                # Temporal Research Dynamics
                elif analysis_type == "Temporal Research Dynamics":
                    st.subheader("Research Evolution and Dynamics")
                    
                    temporal_results = analyzer.advanced_analyzer.get_temporal_analysis()
                    if temporal_results:
                        # Convert to DataFrame for easier manipulation
                        df = pd.DataFrame(temporal_results)
                        df = df.sort_values('PMID')
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Publications", len(df))
                        with col2:
                            st.metric("Average Nodes per Publication", f"{df['nodes'].mean():.1f}")
                        
                        # Create visualization
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=(
                                'Nodes Over Time',
                                'Edges Over Time',
                                'Network Density',
                                'Average Degree'
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df['PMID'], y=df['nodes'], 
                                      mode='lines+markers', name='Nodes'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df['PMID'], y=df['edges'], 
                                      mode='lines+markers', name='Edges'),
                            row=1, col=2
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df['PMID'], y=df['density'], 
                                      mode='lines+markers', name='Density'),
                            row=2, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df['PMID'], y=df['avg_degree'], 
                                      mode='lines+markers', name='Avg Degree'),
                            row=2, col=2
                        )
                        
                        fig.update_layout(height=800, showlegend=True,
                                        title_text='Network Evolution Over Time')
                        st.plotly_chart(fig)

                # Research Cluster Analysis
                
                elif analysis_type == "Research Cluster Analysis":
                    st.subheader("Research Cluster Interaction Analysis")
                    
                    cluster_results = safe_analysis(
                        analyzer.get_cluster_interactions,
                        "Error in Cluster Interactions Analysis"
                    )
                    
                    if cluster_results:
                        # Analysis Parameters
                        st.write("### Analysis Overview")
                        params = cluster_results['metadata']['parameters']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Clusters", cluster_results['metadata']['total_clusters'])
                        with col2:
                            st.metric("Total Nodes", params['node_count'])
                        with col3:
                            st.metric("Total Edges", params['edge_count'])
                        with col4:
                            st.metric("Cluster Count", params['cluster_count'])
                        
                        # Display visualizations
                        if 'visualizations' in cluster_results:
                            st.write("### Inter-Cluster Interactions")
                            st.plotly_chart(cluster_results['visualizations']['dropdown'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("### Interaction Heatmap")
                                st.plotly_chart(cluster_results['visualizations']['heatmap'])
                            with col2:
                                st.write("### Node Type Distribution")
                                st.plotly_chart(cluster_results['visualizations']['composition'])
                        
                        # Detailed Cluster Analysis
                        st.write("### Detailed Cluster Analysis")
                        for cluster, details in cluster_results['cluster_details'].items():
                            with st.expander(f"Cluster: {cluster}"):
                                # Basic Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Nodes", details['total_nodes'])
                                with col2:
                                    st.metric("Internal Edges", details['internal_edges'])
                                with col3:
                                    st.metric("External Edges", details['external_edges'])
                                with col4:
                                    st.metric("Isolation Index", f"{details['isolation_index']:.3f}")
                                
                                # Additional Metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Modularity", f"{details['modularity']:.3f}")
                                with col2:
                                    connection_count = len(details['inter_cluster_connections'])
                                    st.metric("Connected Clusters", connection_count)
                                
                                # Node Type Distribution
                                st.write("#### Node Type Distribution")
                                st.bar_chart(details['node_types'])
                                
                                # Inter-cluster Connections
                                st.write("#### Connections to Other Clusters")
                                connections_df = pd.DataFrame([
                                    {"Target Cluster": target_cluster, "Number of Connections": count}  # Changed column name
                                    for target_cluster, count in details['inter_cluster_connections'].items()
                                ]).sort_values("Number of Connections", ascending=False)  # Sort by new column name
                                
                                # Display the sorted dataframe
                                st.dataframe(connections_df)
                                                        
                        # Analysis Timestamp
                        st.caption(f"Analysis performed at: {cluster_results['metadata']['analysis_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

                
                
                elif analysis_type == "Publication Relationship Mapping":
                    st.subheader("Publication Relationship Network")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # More meaningful filtering controls
                        st.write("### Filter Settings")
                        min_connections = st.slider(
                            "Minimum Shared Entities", 
                            min_value=1, 
                            max_value=10, 
                            value=2,
                            help="Minimum number of shared entities (genes, proteins, etc.) between publications"
                        )
                        
                        min_weight = st.slider(
                            "Minimum Relationship Strength", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.3,
                            step=0.1,
                            help="Minimum similarity score between publications"
                        )
                        
                    with col2:
                        # Publication Statistics
                        st.write("### Publication Network Statistics")
                        
                        # Calculate meaningful metrics
                        publication_groups = defaultdict(list)
                        for node in nodes_data:
                            pmid = node.get('PMID', 'Unknown')
                            publication_groups[pmid].append(node)
                            
                        strong_connections = []
                        for pmid1, pmid2 in combinations(publication_groups.keys(), 2):
                            similarity = analyzer.calculate_publication_similarity(
                                publication_groups[pmid1],
                                publication_groups[pmid2],
                                min_shared=min_connections,
                                min_weight=min_weight
                            )
                            if similarity > min_weight:
                                strong_connections.append((pmid1, pmid2, similarity))
                        
                        # Display meaningful metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric(
                                "Connected Publications",
                                len(set(pmid for conn in strong_connections for pmid in conn[:2]))
                            )
                        with metric_col2:
                            st.metric(
                                "Strong Connections",
                                len(strong_connections)
                            )
                        with metric_col3:
                            avg_similarity = np.mean([conn[2] for conn in strong_connections]) if strong_connections else 0
                            st.metric(
                                "Average Similarity",
                                f"{avg_similarity:.2f}"
                            )
                        
                        # Create publication network visualization
                        if strong_connections:
                            fig = analyzer.create_publication_network_plot(strong_connections, publication_groups)
                            st.plotly_chart(fig, width="stretch")
                            
                            # Add interaction details
                            if st.checkbox("Show Detailed Connections"):
                                st.write("### Publication Connections")
                                connections_df = pd.DataFrame(
                                    strong_connections,
                                    columns=['Publication 1', 'Publication 2', 'Similarity Score']
                                ).sort_values('Similarity Score', ascending=False)
                                st.dataframe(connections_df)
                        else:
                            st.warning("No strong connections found with current filter settings")
                
                
                # Node Exploration
                elif analysis_type == "Node Exploration":
                    st.subheader("Detailed Node Analysis")
                    
                    # Node Selection
                    selected_node = st.selectbox(
                        "Select Node to Explore", 
                        options=sorted(list(analyzer.node_ids))
                    )
                    
                    if selected_node:
                        node_results = safe_analysis(
                            lambda: analyzer.explore_node_details(selected_node),
                            f"Error exploring node {selected_node}"
                        )
                        
                        if node_results:
                            # Network Visualization
                            st.plotly_chart(node_results['visualizations']['network'])
                            
                            # Node Metrics
                            cols = st.columns(4)
                            with cols[0]:
                                st.metric("Degree", node_results['metrics']['degree'])
                            with cols[1]:
                                st.metric("Betweenness", f"{node_results['metrics']['betweenness_centrality']:.4f}")
                            with cols[2]:
                                st.metric("Closeness", f"{node_results['metrics']['closeness_centrality']:.4f}")
                            with cols[3]:
                                st.metric("Clustering", f"{node_results['metrics']['clustering_coefficient']:.4f}")
                            
                            # Neighborhood Analysis
                            st.plotly_chart(node_results['visualizations']['distributions'])

                # Filtered Network View
                elif analysis_type == "Filtered Network View":
                    st.subheader("Network Filtering")
                    
                    # Filtering Controls
                    node_types = st.multiselect(
                        "Filter by Node Types", 
                        options=sorted(set(node['type'] for node in nodes_data))
                    )

                    min_degree = st.slider(
                        "Minimum Node Degree", 
                        min_value=1, 
                        max_value=max(dict(analyzer.G.degree()).values()),
                        value=1
                    )

                    min_weight = st.slider(
                        "Minimum Edge Weight", 
                        min_value=0.0, 
                        max_value=max(d.get('weight', 0) for u, v, d in analyzer.G.edges(data=True)),
                        value=0.0,
                        step=0.1
                    )

                    # Apply Filtering
                    filtered_results = safe_analysis(
                        lambda: analyzer.get_filtered_view(
                            node_types=node_types, 
                            min_degree=min_degree, 
                            min_weight=min_weight
                        ),
                        "Error in Network Filtering"
                    )
                    
                    if filtered_results:
                        # Filtered Network Statistics
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Filtered Nodes", filtered_results['metrics']['nodes'])
                        with cols[1]:
                            st.metric("Filtered Edges", filtered_results['metrics']['edges'])
                        with cols[2]:
                            st.metric("Network Density", f"{filtered_results['metrics']['density']:.4f}")

                        # Visualization
                        st.plotly_chart(filtered_results['visualization'])

                # Network Entropy Analysis
                elif analysis_type == "Network Entropy Analysis":
                    st.subheader("Network Entropy Analysis")
                    
                    entropy_results = safe_analysis(
                        analyzer.get_network_entropy,
                        "Error in Network Entropy Analysis"
                    )
                    
                    if entropy_results:
                        # Entropy Metrics
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Degree Entropy", f"{entropy_results['entropy_measures']['degree_entropy']:.4f}")
                        with cols[1]:
                            st.metric("Type Entropy", f"{entropy_results['entropy_measures']['type_entropy']:.4f}")
                        with cols[2]:
                            st.metric("Cluster Entropy", f"{entropy_results['entropy_measures']['cluster_entropy']:.4f}")
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page or contact support if the issue persists.")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
