"""Simulate card and merchant IDs, including a synthetic fraud ring.

The original dataset has no card/merchant identifiers (removed for
privacy), so this script generates them: random IDs for normal traffic,
plus a deliberately concentrated cluster of cards and merchants for a
subset of fraud transactions, mimicking how real fraud rings reuse a
small set of accounts across a small set of merchants. This is a
hybrid approach -- clearly labeled as simulated -- used to demonstrate
graph-based fraud detection techniques (see build_fraud_graph.py and
README.md, "Graph-Based Fraud Ring Detection").
"""

import numpy as np
import pandas as pd

FEATURES_PATH = "data/features.csv"
OUTPUT_PATH = "data/features_with_graph_ids.csv"

RANDOM_SEED = 42
N_CARDS = 5000
N_MERCHANTS = 800

RING_FRACTION = 0.6  # share of fraud transactions routed through the simulated ring
N_RING_CARDS = 150
N_RING_MERCHANTS = 10


np.random.seed(RANDOM_SEED)

df = pd.read_csv(FEATURES_PATH)
n = len(df)

df["card_id"] = np.random.randint(0, N_CARDS, size=n).astype("int64")
df["merchant_id"] = np.random.randint(0, N_MERCHANTS, size=n).astype("int64")

fraud_idx = df[df["is_fraud"] == 1].index.tolist()
n_fraud = len(fraud_idx)
n_ring_fraud = int(n_fraud * RING_FRACTION)
ring_fraud_idx = np.random.choice(fraud_idx, size=n_ring_fraud, replace=False)

# New, out-of-range IDs so the ring's cards/merchants never coincide with normal traffic.
ring_card_ids = np.arange(N_CARDS, N_CARDS + N_RING_CARDS)
ring_merchant_ids = np.arange(N_MERCHANTS, N_MERCHANTS + N_RING_MERCHANTS)

df.loc[ring_fraud_idx, "card_id"] = np.random.choice(ring_card_ids, size=n_ring_fraud)
df.loc[ring_fraud_idx, "merchant_id"] = np.random.choice(ring_merchant_ids, size=n_ring_fraud)

print(f"Total transactions: {n}")
print(f"Total fraud transactions: {n_fraud}")
print(f"Fraud transactions routed through the ring: {n_ring_fraud}")
print(f"Ring cards: {N_RING_CARDS}, ring merchants: {N_RING_MERCHANTS}")

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nData saved to {OUTPUT_PATH}")