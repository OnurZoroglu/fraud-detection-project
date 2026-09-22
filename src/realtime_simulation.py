"""Simulate a real-time transaction stream against the trained baseline model.

Transactions are drawn only from the test split (never seen during
training) to give an honest read on live performance. An earlier version
of this script sampled from the full dataset, which leaked training
examples into the simulation and produced an artificially perfect result
-- see README.md ("Real-Time Simulation") for that finding.
"""

import time
from datetime import datetime
import joblib
import pandas as pd
from data_prep import get_train_test_indices, load_dataset_with_graph_features

MODEL_PATH = "models/xgboost_baseline_model.pkl"
FEATURE_COLS_PATH = "models/baseline_feature_columns.pkl"

SAMPLE_SEED = 7
N_NORMAL_SAMPLES = 50
ALERT_THRESHOLD = 0.002  # cost-optimal at 100:1 FN:FP cost, see cost_analysis.py / README.md
STREAM_DELAY_SECONDS = 0.1


def build_stream_sample(df):
    """Reproduce the training split and sample from the test set only."""
    _, test_idx = get_train_test_indices(df)
    test_df = df.loc[test_idx]
    print(f"Test set size: {len(test_df)}")
    print(f"Test set fraud count: {test_df['is_fraud'].sum()}\n")

    normal_sample = test_df[test_df["is_fraud"] == 0].sample(N_NORMAL_SAMPLES, random_state=SAMPLE_SEED)
    fraud_sample = test_df[test_df["is_fraud"] == 1]  # every fraud case in the test set
    return pd.concat([normal_sample, fraud_sample]).sample(frac=1, random_state=SAMPLE_SEED).reset_index(drop=True)


def run_simulation(stream_data, model, feature_cols):
    alarm_count = 0
    true_positives = 0
    false_positives = 0

    for idx, row in stream_data.iterrows():
        time.sleep(STREAM_DELAY_SECONDS)

        proba = model.predict_proba(row[feature_cols].values.reshape(1, -1))[0][1]
        is_alert = proba >= ALERT_THRESHOLD
        is_actual_fraud = row["is_fraud"] == 1

        if is_alert:
            alarm_count += 1
            outcome = "correctly flagged" if is_actual_fraud else "false positive"
        else:
            outcome = "missed fraud" if is_actual_fraud else "correct"
        if is_alert and is_actual_fraud:
            true_positives += 1
        elif is_alert and not is_actual_fraud:
            false_positives += 1

        timestamp = datetime.now().strftime("%H:%M:%S")
        status = "ALERT" if is_alert else "normal"
        print(f"[{timestamp}] txn #{idx + 1:03d} | amount: {row['amount']:>8.2f} | "
              f"score: {proba:.4f} | {status} ({outcome})")

    return alarm_count, true_positives, false_positives


model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)
print(f"Model loaded. Feature count: {len(feature_cols)}\n")

df, v_columns = load_dataset_with_graph_features()
stream_data = build_stream_sample(df)

print("=" * 70)
print("REAL-TIME FRAUD DETECTION SIMULATION")
print("=" * 70)
print(f"Alert threshold: {ALERT_THRESHOLD}\n")

alarm_count, true_positives, false_positives = run_simulation(stream_data, model, feature_cols)

print("\n" + "=" * 70)
print("SIMULATION SUMMARY")
print("=" * 70)
print(f"Total transactions: {len(stream_data)}")
print(f"Total alerts:        {alarm_count}")
print(f"True positives:      {true_positives}")
print(f"False positives:     {false_positives}")
print(f"Actual fraud count:  {stream_data['is_fraud'].sum()}")