"""Verify that Louvain community detection correctly isolates the simulated fraud ring.

Checks whether the ring's cards and merchants (IDs assigned by
simulate_graph_data.py) all land in a single community, as opposed to
being scattered across multiple communities. See README.md
("Graph-Based Fraud Ring Detection") for the result.
"""

from collections import Counter
import pandas as pd
from build_fraud_graph import build_graph, detect_communities

DATA_PATH = "data/features_with_graph_ids.csv"

# Must match the ID ranges assigned in simulate_graph_data.py
RING_CARD_IDS = set(range(5000, 5150))
RING_MERCHANT_IDS = set(range(800, 810))


df = pd.read_csv(DATA_PATH)
graph = build_graph(df)
community_df = detect_communities(graph)
partition = dict(zip(community_df["node"], community_df["community"]))

ring_communities = []
for node, community in partition.items():
    if node.startswith("card_") and int(node.removeprefix("card_")) in RING_CARD_IDS:
        ring_communities.append(community)
    elif node.startswith("merchant_") and int(node.removeprefix("merchant_")) in RING_MERCHANT_IDS:
        ring_communities.append(community)

print("Communities the ring's nodes were assigned to:")
print(Counter(ring_communities))