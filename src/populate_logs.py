"""Send a batch of realistic transactions to the running API to populate prediction logs.

Used to build a large enough sample for drift_monitor.py to produce a
statistically meaningful result. Requires the API service (api_service.py)
to be running.
"""

import requests

from data_prep import load_dataset_with_graph_features

API_URL = "http://127.0.0.1:8000/predict"
SAMPLE_SIZE = 300
PROGRESS_INTERVAL = 50


def build_payload(row) -> dict:
    return {
        "amount": float(row["amount"]),
        "txn_count_last_hour": int(row["txn_count_last_hour"]),
        "avg_amount_last_hour": float(row["avg_amount_last_hour"]),
        "time_since_last_txn": float(row["time_since_last_txn"]),
        "v_features": {f"v{i}": float(row[f"v{i}"]) for i in range(1, 29)},
    }


df, _ = load_dataset_with_graph_features()
sample = df.sample(SAMPLE_SIZE)  # no fixed seed: repeated runs should draw different transactions

print(f"Sending {len(sample)} transactions to the API...\n")

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