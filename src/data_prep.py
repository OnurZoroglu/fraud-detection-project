"""Shared data loading utilities used across the project's training and analysis scripts."""

import pandas as pd

FEATURES_PATH = "data/features.csv"
RAW_DATA_PATH = "data/creditcard.csv"


def load_dataset(features_path: str = FEATURES_PATH, raw_path: str = RAW_DATA_PATH):
    """Load SQL-engineered features and merge them with the original PCA components (v1..v28).

    Returns:
        A tuple of (merged dataframe, list of PCA column names).
    """
    df = pd.read_csv(features_path)
    df["time_since_last_txn"] = df["time_since_last_txn"].fillna(0)

    raw = pd.read_csv(raw_path)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={"class": "is_fraud", "time": "time_seconds"})
    v_columns = [c for c in raw.columns if c.startswith("v")]

    df = df.merge(
        raw[["time_seconds"] + v_columns].reset_index().rename(columns={"index": "orig_idx"}),
        on="time_seconds",
        how="left",
    )
    return df.drop_duplicates(subset="transaction_id"), v_columns


def get_base_feature_cols(v_columns: list[str]) -> list[str]:
    """Return the standard (non-graph) feature column list, in training order."""
    return ["amount", "txn_count_last_hour", "avg_amount_last_hour", "time_since_last_txn"] + v_columns


def load_dataset_with_graph_features(graph_features_path: str = "data/graph_features.csv"):
    """Load the base dataset joined with card-level graph features (degree centrality, community size).

    Requires data/features_with_graph_ids.csv (see simulate_graph_data.py) instead of
    the plain features.csv, since graph features are joined on card_id.
    """
    df = pd.read_csv("data/features_with_graph_ids.csv")
    df["time_since_last_txn"] = df["time_since_last_txn"].fillna(0)

    raw = pd.read_csv(RAW_DATA_PATH)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={"class": "is_fraud", "time": "time_seconds"})
    v_columns = [c for c in raw.columns if c.startswith("v")]

    df = df.merge(
        raw[["time_seconds"] + v_columns].reset_index().rename(columns={"index": "orig_idx"}),
        on="time_seconds",
        how="left",
        suffixes=("", "_orig"),
    )
    df = df.drop_duplicates(subset="transaction_id")

    graph_features = pd.read_csv(graph_features_path)
    df = df.merge(graph_features, on="card_id", how="left")
    df["degree_centrality"] = df["degree_centrality"].fillna(0)
    df["community_size"] = df["community_size"].fillna(df["community_size"].median())

    return df, v_columns


def get_graph_feature_cols(v_columns: list[str]) -> list[str]:
    """Return the extended feature column list including graph-based features."""
    return get_base_feature_cols(v_columns) + ["degree_centrality", "community_size"]