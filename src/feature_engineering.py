"""Derive time-based behavioral features from transaction history using SQL window functions.

Runs against the PostgreSQL database populated by load_data.py and writes
the result to data/features.csv, which downstream scripts (via
data_prep.load_dataset) merge back with the original PCA components.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_URL = os.environ.get("FRAUD_DB_URL")
OUTPUT_PATH = "data/features.csv"

if DB_URL is None:
    raise RuntimeError(
        "FRAUD_DB_URL is not set. Copy .env.example to .env and fill in your database credentials."
    )

QUERY = """
SELECT
    transaction_id,
    time_seconds,
    amount,
    is_fraud,
    COUNT(*) OVER (
        ORDER BY time_seconds
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS txn_count_last_hour,
    AVG(amount) OVER (
        ORDER BY time_seconds
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS avg_amount_last_hour,
    time_seconds - LAG(time_seconds) OVER (ORDER BY time_seconds) AS time_since_last_txn
FROM transactions
ORDER BY time_seconds;
"""

engine = create_engine(DB_URL)
df_features = pd.read_sql(QUERY, engine)

print(df_features.head(10))
print(f"\nTotal rows: {len(df_features)}")

df_features.to_csv(OUTPUT_PATH, index=False)
print(f"Features saved to {OUTPUT_PATH}")