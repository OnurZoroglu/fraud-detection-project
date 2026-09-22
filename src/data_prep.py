"""Shared data loading utilities used across the project's training and analysis scripts."""

import numpy as np
import pandas as pd

FEATURES_PATH = "data/features.csv"
RAW_DATA_PATH = "data/creditcard.csv"


def load_raw(raw_path: str = RAW_DATA_PATH):
    """Load the raw Kaggle CSV with the same normalization and transaction_id as load_data.py.

    transaction_id is the 1-based row number in the raw file. load_data.py assigns it
    the same way before loading into PostgreSQL, and the SQL feature query passes it
    through unchanged, so it is the key that links data/features.csv back to the raw
    row. (time_seconds is not a valid key: most timestamps are shared by several
    transactions.)

    Returns:
        A tuple of (raw dataframe, list of PCA column names).
    """
    raw = pd.read_csv(raw_path)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={"class": "is_fraud", "time": "time_seconds"})
    raw.insert(0, "transaction_id", range(1, len(raw) + 1))
    v_columns = [c for c in raw.columns if c.startswith("v")]
    return raw, v_columns


def check_alignment(df: pd.DataFrame, raw: pd.DataFrame, v_columns: list[str]) -> None:
    """Raise if any row's features and PCA components don't come from the same raw transaction."""
    if len(df) != len(raw):
        raise ValueError(f"Alignment check failed: expected {len(raw)} rows, got {len(df)}")

    ref = raw.set_index("transaction_id").loc[df["transaction_id"]]
    problems = {
        "missing PCA components": int(df[v_columns].isna().any(axis=1).sum()),
        "amount mismatch": int((~np.isclose(df["amount"].to_numpy(), ref["amount"].to_numpy())).sum()),
        "time mismatch": int((df["time_seconds"].to_numpy() != ref["time_seconds"].to_numpy()).sum()),
        "label mismatch": int((df["is_fraud"].to_numpy() != ref["is_fraud"].to_numpy()).sum()),
    }
    failed = {name: count for name, count in problems.items() if count}
    if failed:
        raise ValueError(f"Alignment check failed (rows affected): {failed}")


def attach_pca_components(df: pd.DataFrame, raw_path: str = RAW_DATA_PATH):
    """Join v1..v28 from the raw CSV onto df by transaction_id and verify the alignment."""
    raw, v_columns = load_raw(raw_path)
    df = df.merge(raw[["transaction_id"] + v_columns], on="transaction_id", how="left", validate="one_to_one")
    check_alignment(df, raw, v_columns)
    return df, v_columns


def load_dataset(features_path: str = FEATURES_PATH, raw_path: str = RAW_DATA_PATH):
    """Load SQL-engineered features and merge them with the original PCA components (v1..v28).

    Returns:
        A tuple of (merged dataframe, list of PCA column names).
    """
    df = pd.read_csv(features_path)
    df["time_since_last_txn"] = df["time_since_last_txn"].fillna(0)
    return attach_pca_components(df, raw_path)


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
    df, v_columns = attach_pca_components(df)

    graph_features = pd.read_csv(graph_features_path)
    df = df.merge(graph_features, on="card_id", how="left")
    df["degree_centrality"] = df["degree_centrality"].fillna(0)
    df["community_size"] = df["community_size"].fillna(df["community_size"].median())

    return df, v_columns


def get_graph_feature_cols(v_columns: list[str]) -> list[str]:
    """Return the extended feature column list including graph-based features."""
    return get_base_feature_cols(v_columns) + ["degree_centrality", "community_size"]
