"""Load the raw Kaggle credit card fraud dataset into PostgreSQL.

Normalizes column names, adds a surrogate transaction_id, and writes the
result to the `transactions` table used by feature_engineering.py.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

RAW_DATA_PATH = "data/creditcard.csv"
DB_URL = os.environ.get("FRAUD_DB_URL")

if DB_URL is None:
    raise RuntimeError(
        "FRAUD_DB_URL is not set. Copy .env.example to .env and fill in your database credentials."
    )

df = pd.read_csv(RAW_DATA_PATH)

df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={"class": "is_fraud", "time": "time_seconds"})
df.insert(0, "transaction_id", range(1, len(df) + 1))

engine = create_engine(DB_URL)
df.to_sql("transactions", engine, if_exists="replace", index=False)

print(f"{len(df)} transactions loaded.")
print(f"Fraud rate: {df['is_fraud'].mean() * 100:.3f}%")