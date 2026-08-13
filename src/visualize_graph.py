"""Visualize the simulated fraud ring against a sample of normal card-merchant activity.

The full graph has ~6,000 nodes, too dense to plot meaningfully, so this
renders a subgraph: every ring node plus 100 randomly sampled normal
cards (and the merchants they transacted with). The ring should appear
as a visually isolated cluster, disconnected from the normal traffic --
see README.md ("Graph-Based Fraud Ring Detection") for the resulting plot.
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import networkx as nx
from build_fraud_graph import build_graph

DATA_PATH = "data/features_with_graph_ids.csv"
OUTPUT_PATH = "data/fraud_ring_network.png"

# Must match the ID ranges assigned in simulate_graph_data.py
RING_CARD_IDS = set(range(5000, 5150))
RING_MERCHANT_IDS = set(range(800, 810))

RANDOM_SEED = 42
N_NORMAL_CARD_SAMPLES = 100


def is_ring_node(node: str) -> bool:
    if node.startswith("card_"):
        return int(node.removeprefix("card_")) in RING_CARD_IDS
    if node.startswith("merchant_"):
        return int(node.removeprefix("merchant_")) in RING_MERCHANT_IDS
    return False


def build_display_subgraph(graph: nx.Graph, df: pd.DataFrame) -> nx.Graph:
    ring_nodes = [n for n in graph.nodes() if is_ring_node(n)]

    np.random.seed(RANDOM_SEED)
    normal_card_ids = df.loc[df["card_id"] < min(RING_CARD_IDS), "card_id"].unique()
    normal_cards = [f"card_{c}" for c in np.random.choice(normal_card_ids, size=N_NORMAL_CARD_SAMPLES, replace=False)]

    # Include the merchants those cards transacted with, otherwise they appear as isolated points.
    normal_merchants = set()
    for card in normal_cards:
        normal_merchants.update(graph.neighbors(card))

    subgraph = graph.subgraph(ring_nodes + normal_cards + list(normal_merchants)).copy()
    subgraph.remove_nodes_from(list(nx.isolates(subgraph)))
    return subgraph


df = pd.read_csv(DATA_PATH)
graph = build_graph(df)
subgraph = build_display_subgraph(graph, df)
print(f"Rendering subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

node_colors, node_sizes = [], []
for node in subgraph.nodes():
    if is_ring_node(node):
        node_colors.append("crimson")
        node_sizes.append(120)
    elif node.startswith("merchant_"):
        node_colors.append("darkorange")
        node_sizes.append(150)
    else:
        node_colors.append("steelblue")
        node_sizes.append(40)

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(subgraph, k=0.3, seed=RANDOM_SEED, iterations=50)
nx.draw_networkx_edges(subgraph, pos, alpha=0.25, width=0.6)
nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85)

plt.title("Card-Merchant Network: Fraud Ring (Red) vs. Normal Activity (Blue/Orange)", fontsize=13)
plt.axis("off")
plt.legend(
    handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=10, label="Fraud ring (card/merchant)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=10, label="Normal card"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange", markersize=10, label="Normal merchant"),
    ],
    loc="upper right",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Plot saved to {OUTPUT_PATH}")