"""Send a batch of realistic transactions to the running API to populate prediction logs.

Used to build a large enough sample for drift_monitor.py to produce a
statistically meaningful result. Requires the API service (api_service.py)
to be running.

Transactions are drawn only from the held-out test split (the same split
the served model was trained on, see data_prep.get_train_test_indices), so
the logs reflect data the model has never seen. Each request carries a
"POPULATE-<id>" transaction_id so these rows can be told apart in the log.

Start the API *without* N8N_WEBHOOK_URL while running this script,
otherwise every flagged transaction is also sent to the n8n fraud-alert
workflow (and on to Telegram).
"""

import os

import requests
from dotenv import load_dotenv

from data_prep import get_train_test_indices, load_dataset_with_graph_features

API_URL = "http://127.0.0.1:8000/predict"
SAMPLE_SIZE = 300
PROGRESS_INTERVAL = 50


def build_payload(row) -> dict:
    return {
        "transaction_id": f"POPULATE-{int(row['transaction_id'])}",
        "amount": float(row["amount"]),
        "txn_count_last_hour": int(row["txn_count_last_hour"]),
        "avg_amount_last_hour": float(row["avg_amount_last_hour"]),
        "time_since_last_txn": float(row["time_since_last_txn"]),
        "v_features": {f"v{i}": float(row[f"v{i}"]) for i in range(1, 29)},
    }


load_dotenv()
if os.getenv("N8N_WEBHOOK_URL"):
    print(
        "WARNING: N8N_WEBHOOK_URL is set in this environment/.env. If the API was started with it,\n"
        "flagged transactions from this run will trigger n8n fraud alerts. Restart the API without it\n"
        "to avoid that.\n"
    )

df, _ = load_dataset_with_graph_features()
_, test_idx = get_train_test_indices(df)
sample = df.loc[test_idx].sample(SAMPLE_SIZE)  # no fixed seed: repeated runs should draw different transactions

print(f"Sending {len(sample)} test-split transactions to the API...\n")

success_count = 0
error_count = 0

for _, row in sample.iterrows():
    try:
        response = requests.post(API_URL, json=build_payload(row), timeout=5)
        if response.status_code == 200:
            success_count += 1
        else:
            error_count += 1
    except requests.RequestException:
        error_count += 1

    total = success_count + error_count
    if total % PROGRESS_INTERVAL == 0:
        print(f"Progress: {total}/{len(sample)}")

print(f"\nDone. Succeeded: {success_count}, Failed: {error_count}")
print("You can now re-run drift_monitor.py.")
