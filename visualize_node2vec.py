"""
═══════════════════════════════════════════════════════════════════════════════
NODE2VEC EMBEDDINGS & GRAPH VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

Visualizes:
1. Correlation Matrix → Network Graph Construction
2. Random Walks on the Graph
3. Node2Vec 32D Vector Embeddings (reduced to 2D/3D for visualization)
4. Embedding Space Clusters

Author: Harshvardhan
Date: 2026-02-15
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

EMBEDDING_DIM = 32
WALK_LENGTH = 10
NUM_WALKS = 20
CORRELATION_THRESHOLD = 0.7

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD PORTFOLIO DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_portfolio_data():
    """Load portfolio data from CSV files"""
    try:
        portfolio = pd.read_csv('alpha_portfolio_complete.csv')
        print(f"✅ Loaded portfolio: {len(portfolio)} stocks")
        return portfolio
    except FileNotFoundError:
        print("❌ Portfolio file not found. Please generate a portfolio first.")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD CORRELATION NETWORK
# ═══════════════════════════════════════════════════════════════════════════

def build_correlation_network(portfolio, correlation_threshold=0.7):
    """
    Build network graph from correlation matrix
    Returns: NetworkX graph, correlation matrix
    """
    print("\n🕸️  STEP 1: Building Correlation Network...")
    
    # Simulate correlation matrix (in real implementation, use actual returns)
    tickers = portfolio['ticker'].tolist()
    n = len(tickers)
    
    # Create synthetic correlation matrix for demonstration
    # In production, calculate from actual price returns
    np.random.seed(42)
    corr_matrix = np.random.rand(n, n)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(corr_matrix, 1.0)  # Diagonal = 1
    
    corr_df = pd.DataFrame(corr_matrix, index=tickers, columns=tickers)
    
    # Build graph
    G = nx.Graph()
    G.add_nodes_from(tickers)
    
    # Add node attributes
    for idx, row in portfolio.iterrows():
        ticker = row['ticker']
        G.nodes[ticker]['sector'] = row['sector']
        G.nodes[ticker]['weight'] = row['weight']
        G.nodes[ticker]['return'] = row['expected_return']
    
    # Add edges based on correlation threshold
    edges_added = 0
    for i, ticker_i in enumerate(tickers):
        for j, ticker_j in enumerate(tickers):
            if i >= j:
                continue
            
            corr = corr_df.loc[ticker_i, ticker_j]
            
            # Add edge if correlation exceeds threshold
            if corr > correlation_threshold:
                G.add_edge(ticker_i, ticker_j, weight=corr)
                edges_added += 1
    
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Avg Degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    return G, corr_df

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: GENERATE RANDOM WALKS
# ═══════════════════════════════════════════════════════════════════════════

def generate_random_walks(G, walk_length=10, num_walks=20):
    """Generate random walks for Node2Vec"""
    print("\n🚶 STEP 2: Generating Random Walks...")
    
    nodes = list(G.nodes())
    all_walks = []
    
    for node in nodes:
        for _ in range(num_walks):
            walk = [node]
            
            for _ in range(walk_length - 1):
                current = walk[-1]
                neighbors = list(G.neighbors(current))
                
                if not neighbors:
                    break
                
                # Weighted random selection
                weights = [G[current][n].get('weight', 1.0) for n in neighbors]
                total_weight = sum(weights)
                
                if total_weight == 0:
                    break
                
                weights = np.array(weights) / total_weight
                next_node = np.random.choice(neighbors, p=weights)
                walk.append(next_node)
            
            if len(walk) > 1:
                all_walks.append(walk)
    
    print(f"   Generated {len(all_walks)} random walks")
    print(f"   Avg walk length: {np.mean([len(w) for w in all_walks]):.2f}")
    
    return all_walks

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: GENERATE NODE2VEC EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════

def generate_node2vec_embeddings(G, all_walks, dimensions=32):
    """Generate Node2Vec embeddings using co-occurrence matrix"""
    print(f"\n🧠 STEP 3: Generating Node2Vec Embeddings (dim={dimensions})...")
    
    nodes = list(G.nodes())
    
    # Build co-occurrence matrix
    co_occurrence = defaultdict(lambda: defaultdict(int))
    
    for walk in all_walks:
        for i, node in enumerate(walk):
            context_start = max(0, i - 5)
            context_end = min(len(walk), i + 6)
            
            for j in range(context_start, context_end):
                if i != j:
                    co_occurrence[node][walk[j]] += 1
    
    # Convert to matrix
    nodes_list = sorted(nodes)
    matrix_size = len(nodes_list)
    co_matrix = np.zeros((matrix_size, matrix_size))
    
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
    
    for node1 in nodes_list:
        for node2, count in co_occurrence[node1].items():
            co_matrix[node_to_idx[node1], node_to_idx[node2]] = count
    
    # SVD for dimensionality reduction
    try:
        from scipy.sparse.linalg import svds
        from sklearn.preprocessing import normalize
        
        co_matrix += np.random.randn(*co_matrix.shape) * 0.01
        k_components = min(dimensions, matrix_size - 1)
        
        U, s, Vt = svds(co_matrix, k=k_components)
        
        if k_components < dimensions:
            padding = np.random.randn(U.shape[0], dimensions - k_components) * 0.01
            embeddings_matrix = np.hstack([U @ np.diag(s), padding])
        else:
            embeddings_matrix = U @ np.diag(s)
        
        # Normalize
        embeddings_matrix = normalize(embeddings_matrix, axis=1)
        
    except Exception as e:
        print(f"   ⚠️  SVD failed, using random projection")
        projection = np.random.randn(matrix_size, dimensions) * 0.1
        embeddings_matrix = co_matrix @ projection
    
    # Create embeddings dictionary
    embeddings = {}
    for i, node in enumerate(nodes_list):
        embeddings[node] = embeddings_matrix[i]
    
    print(f"   ✅ Generated {dimensions}D embeddings for {len(embeddings)} stocks")
    
    return embeddings, co_matrix

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 1: CORRELATION MATRIX HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def visualize_correlation_matrix(corr_df, portfolio):
    """Visualize correlation matrix as heatmap"""
    print("\n📊 Creating Correlation Matrix Heatmap...")
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_df.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Stock Correlation Matrix",
        xaxis_title="Ticker",
        yaxis_title="Ticker",
        width=800,
        height=800
    )
    
    fig.write_html("output_1_correlation_matrix.html")
    print("   ✅ Saved: output_1_correlation_matrix.html")
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 2: NETWORK GRAPH
# ═══════════════════════════════════════════════════════════════════════════

def visualize_network_graph(G, portfolio):
    """Visualize stock correlation network"""
    print("\n🕸️  Creating Network Graph Visualization...")
    
    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(G[edge[0]][edge[1]]['weight'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    # Color by sector
    sectors = portfolio['sector'].unique()
    sector_colors = {sector: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] 
                     for i, sector in enumerate(sectors)}
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        sector = G.nodes[node]['sector']
        weight = G.nodes[node]['weight']
        ret = G.nodes[node]['return']
        
        node_text.append(f"{node}<br>Sector: {sector}<br>Weight: {weight:.1%}<br>Return: {ret:.2%}")
        node_colors.append(sector_colors[sector])
        node_sizes.append(weight * 1000 + 10)  # Scale by portfolio weight
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        textfont=dict(size=8),
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Stock Correlation Network Graph",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        width=1000,
                        height=800
                    ))
    
    fig.write_html("output_2_network_graph.html")
    print("   ✅ Saved: output_2_network_graph.html")
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 3: RANDOM WALK PATHS
# ═══════════════════════════════════════════════════════════════════════════

def visualize_random_walks(G, all_walks, num_walks_to_show=5):
    """Visualize sample random walk paths"""
    print("\n🚶 Creating Random Walk Visualization...")
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    fig = go.Figure()
    
    # Add all edges in gray
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.3, color='lightgray'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add sample walks in different colors
    colors = px.colors.qualitative.Set1
    for i, walk in enumerate(all_walks[:num_walks_to_show]):
        walk_x = []
        walk_y = []
        
        for node in walk:
            x, y = pos[node]
            walk_x.append(x)
            walk_y.append(y)
        
        fig.add_trace(go.Scatter(
            x=walk_x,
            y=walk_y,
            mode='lines+markers',
            name=f"Walk {i+1}",
            line=dict(width=2, color=colors[i % len(colors)]),
            marker=dict(size=8)
        ))
    
    # Add all nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(size=10, color='lightblue', line=dict(width=1, color='black')),
        showlegend=False,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"Sample Random Walks (showing {num_walks_to_show} of {len(all_walks)})",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1000,
        height=800
    )
    
    fig.write_html("output_3_random_walks.html")
    print("   ✅ Saved: output_3_random_walks.html")
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 4: EMBEDDING SPACE (2D PCA)
# ═══════════════════════════════════════════════════════════════════════════

def visualize_embeddings_2d(embeddings, portfolio):
    """Visualize 32D embeddings reduced to 2D using PCA"""
    print("\n🎨 Creating 2D Embedding Visualization (PCA)...")
    
    tickers = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[t] for t in tickers])
    
    # PCA to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embedding_matrix)
    
    # Create dataframe
    df = pd.DataFrame({
        'ticker': tickers,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1]
    })
    
    # Merge with portfolio data
    df = df.merge(portfolio[['ticker', 'sector', 'weight', 'expected_return']], on='ticker')
    
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='sector',
        size='weight',
        hover_data=['ticker', 'expected_return'],
        text='ticker',
        title=f"Node2Vec Embeddings (32D → 2D PCA)<br>Explained Variance: {pca.explained_variance_ratio_.sum():.1%}",
        width=1000,
        height=800
    )
    
    fig.update_traces(textposition='top center', textfont_size=10)
    fig.update_layout(
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
    )
    
    fig.write_html("output_4_embeddings_2d_pca.html")
    print("   ✅ Saved: output_4_embeddings_2d_pca.html")
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 5: EMBEDDING SPACE (2D t-SNE)
# ═══════════════════════════════════════════════════════════════════════════

def visualize_embeddings_tsne(embeddings, portfolio):
    """Visualize 32D embeddings reduced to 2D using t-SNE"""
    print("\n🎨 Creating 2D Embedding Visualization (t-SNE)...")
    
    tickers = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[t] for t in tickers])
    
    # t-SNE to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tickers)-1))
    embeddings_2d = tsne.fit_transform(embedding_matrix)
    
    # Create dataframe
    df = pd.DataFrame({
        'ticker': tickers,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1]
    })
    
    # Merge with portfolio data
    df = df.merge(portfolio[['ticker', 'sector', 'weight', 'expected_return']], on='ticker')
    
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='sector',
        size='weight',
        hover_data=['ticker', 'expected_return'],
        text='ticker',
        title="Node2Vec Embeddings (32D → 2D t-SNE)",
        width=1000,
        height=800
    )
    
    fig.update_traces(textposition='top center', textfont_size=10)
    fig.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2"
    )
    
    fig.write_html("output_5_embeddings_2d_tsne.html")
    print("   ✅ Saved: output_5_embeddings_2d_tsne.html")
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 6: CO-OCCURRENCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════

def visualize_cooccurrence_matrix(co_matrix, tickers):
    """Visualize co-occurrence matrix from random walks"""
    print("\n📊 Creating Co-occurrence Matrix Heatmap...")
    
    fig = go.Figure(data=go.Heatmap(
        z=co_matrix,
        x=tickers,
        y=tickers,
        colorscale='Viridis',
        colorbar=dict(title="Co-occurrence Count")
    ))
    
    fig.update_layout(
        title="Stock Co-occurrence Matrix (from Random Walks)",
        xaxis_title="Ticker",
        yaxis_title="Ticker",
        width=800,
        height=800
    )
    
    fig.write_html("output_6_cooccurrence_matrix.html")
    print("   ✅ Saved: output_6_cooccurrence_matrix.html")
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main execution function"""
    print("═" * 80)
    print("NODE2VEC EMBEDDINGS & GRAPH VISUALIZATION")
    print("═" * 80)
    
    # Step 1: Load data
    portfolio = load_portfolio_data()
    if portfolio is None:
        return
    
    # Step 2: Build correlation network
    G, corr_df = build_correlation_network(portfolio, CORRELATION_THRESHOLD)
    
    # Step 3: Generate random walks
    all_walks = generate_random_walks(G, WALK_LENGTH, NUM_WALKS)
    
    # Step 4: Generate Node2Vec embeddings
    embeddings, co_matrix = generate_node2vec_embeddings(G, all_walks, EMBEDDING_DIM)
    
    # Step 5: Create visualizations
    print("\n" + "═" * 80)
    print("GENERATING VISUALIZATIONS")
    print("═" * 80)
    
    visualize_correlation_matrix(corr_df, portfolio)
    visualize_network_graph(G, portfolio)
    visualize_random_walks(G, all_walks, num_walks_to_show=5)
    visualize_embeddings_2d(embeddings, portfolio)
    visualize_embeddings_tsne(embeddings, portfolio)
    visualize_cooccurrence_matrix(co_matrix, list(embeddings.keys()))
    
    print("\n" + "═" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("═" * 80)
    print("\nGenerated Files:")
    print("  1. output_1_correlation_matrix.html")
    print("  2. output_2_network_graph.html")
    print("  3. output_3_random_walks.html")
    print("  4. output_4_embeddings_2d_pca.html")
    print("  5. output_5_embeddings_2d_tsne.html")
    print("  6. output_6_cooccurrence_matrix.html")
    print("\nOpen these HTML files in your browser to view the visualizations!")

if __name__ == "__main__":
    main()
