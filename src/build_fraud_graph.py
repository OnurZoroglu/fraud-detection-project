"""Build a card-merchant bipartite graph and extract fraud-ring-relevant features.

Nodes are cards and merchants; an edge exists between a card and a merchant
if the card transacted there, weighted by transaction count. Louvain
community detection is used to find tightly-connected clusters, which fraud
rings tend to form (see README.md, "Graph-Based Fraud Ring Detection").
"""

import community as community_louvain
import networkx as nx
import pandas as pd

DATA_PATH = "data/features_with_graph_ids.csv"
OUTPUT_PATH = "data/graph_features.csv"


def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a bipartite card-merchant graph, weighted by transaction count."""
    graph = nx.Graph()
    graph.add_nodes_from((f"card_{c}" for c in df["card_id"].unique()), bipartite=0)
    graph.add_nodes_from((f"merchant_{m}" for m in df["merchant_id"].unique()), bipartite=1)

    edge_weights = df.groupby(["card_id", "merchant_id"]).size().reset_index(name="weight")
    for _, row in edge_weights.iterrows():
        graph.add_edge(f"card_{row['card_id']}", f"merchant_{row['merchant_id']}", weight=row["weight"])
    return graph


def compute_card_centrality(graph: nx.Graph) -> pd.DataFrame:
    """Degree centrality per card: how many distinct merchants it has transacted with."""
    degree_centrality = nx.degree_centrality(graph)
    return pd.DataFrame([
        {"card_id": int(node.removeprefix("card_")), "degree_centrality": score}
        for node, score in degree_centrality.items()
        if node.startswith("card_")
    ])


def detect_communities(graph: nx.Graph) -> pd.DataFrame:
    """Assign each node to a community via the Louvain algorithm."""
    partition = community_louvain.best_partition(graph)
    return pd.DataFrame([{"node": node, "community": comm_id} for node, comm_id in partition.items()])


df = pd.read_csv(DATA_PATH)
graph = build_graph(df)
print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

card_centrality = compute_card_centrality(graph)
print("\nTop 10 cards by degree centrality:")
print(card_centrality.sort_values("degree_centrality", ascending=False).head(10))

community_df = detect_communities(graph)
community_sizes = community_df["community"].value_counts()
print(f"\nTotal communities: {len(community_sizes)}")
print("Smallest 10 communities (small, tightly-connected clusters can indicate fraud rings):")
print(community_sizes.tail(10))

card_community = community_df[community_df["node"].str.startswith("card_")].copy()
card_community["card_id"] = card_community["node"].str.removeprefix("card_").astype(int)
card_community = card_community[["card_id", "community"]]

graph_features = card_centrality.merge(card_community, on="card_id", how="left")
graph_features["community_size"] = graph_features["community"].map(community_sizes.to_dict())

graph_features.to_csv(OUTPUT_PATH, index=False)
print(f"\nGraph features saved to {OUTPUT_PATH}")