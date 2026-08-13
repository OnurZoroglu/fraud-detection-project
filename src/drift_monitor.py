"""Statistical drift check for the fraud detection API.

Uses Population Stability Index (PSI) as the primary drift metric, since
it doesn't assume a Gaussian distribution the way a Z-score does. A
Z-score on the mean is kept as a simple secondary check. See README.md
("FastAPI Service & Drift Monitoring") for the methodology note on why
sample size matters here.
"""

import json
import numpy as np
import pandas as pd

RAW_DATA_PATH = "data/creditcard.csv"
LOG_PATH = "logs/prediction_log.jsonl"

DRIFT_Z_THRESHOLD = 2.0
PSI_MODERATE_THRESHOLD = 0.1  # PSI < 0.1: no significant shift
PSI_SIGNIFICANT_THRESHOLD = 0.25  # PSI > 0.25: significant shift
HIGH_ALERT_RATE_THRESHOLD = 0.20  # true fraud rate is ~0.17%, so this is a generous ceiling
PSI_N_BINS = 10


def load_reference_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df


def load_prediction_logs(path: str) -> pd.DataFrame:
    records = []
    skipped = 0
    with open(path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"Skipped {skipped} malformed log line(s).")
    return pd.DataFrame(records)


def population_stability_index(reference: pd.Series, live: pd.Series, n_bins: int = PSI_N_BINS) -> float:
    """PSI between a reference and live distribution, binned by reference quantiles.

    PSI doesn't assume normality, unlike a Z-score, which makes it a more
    reliable choice for skewed features like transaction amount.
    """
    quantile_edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    quantile_edges[0], quantile_edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=quantile_edges)
    live_counts, _ = np.histogram(live, bins=quantile_edges)

    ref_pct = np.maximum(ref_counts / len(reference), 1e-4)  # avoid division/log by zero
    live_pct = np.maximum(live_counts / len(live), 1e-4)

    return float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))


def interpret_psi(psi: float) -> str:
    if psi < PSI_MODERATE_THRESHOLD:
        return "no significant shift"
    if psi < PSI_SIGNIFICANT_THRESHOLD:
        return "moderate shift -- monitor"
    return "significant shift -- investigate"


reference_df = load_reference_data(RAW_DATA_PATH)
print("=== Reference Statistics (Training Data) ===")
print(f"Mean amount: {reference_df['amount'].mean():.2f}")
print(f"Std amount:  {reference_df['amount'].std():.2f}\n")

live_df = load_prediction_logs(LOG_PATH)

if len(live_df) == 0:
    print("No log data yet.")
else:
    live_mean_amount = live_df["amount"].mean()
    live_mean_fraud_prob = live_df["fraud_probability"].mean()
    live_alert_rate = live_df["is_fraud_alert"].mean()

    print("=== Live Data Statistics ===")
    print(f"Total records:          {len(live_df)}")
    print(f"Mean amount:            {live_mean_amount:.2f}")
    print(f"Mean fraud probability: {live_mean_fraud_prob:.4f}")
    print(f"Alert rate:             {live_alert_rate * 100:.2f}%\n")

    print("=== Drift Check ===")

    psi = population_stability_index(reference_df["amount"], live_df["amount"])
    print(f"Amount PSI: {psi:.4f} ({interpret_psi(psi)})")

    z_score = (live_mean_amount - reference_df["amount"].mean()) / reference_df["amount"].std()
    print(f"Amount Z-score (secondary check): {z_score:.2f}")
    if abs(z_score) > DRIFT_Z_THRESHOLD:
        print(f"WARNING: Z-score exceeds +/-{DRIFT_Z_THRESHOLD}.")

    if live_alert_rate > HIGH_ALERT_RATE_THRESHOLD:
        print(
            f"WARNING: alert rate ({live_alert_rate * 100:.2f}%) is much higher than "
            "expected -- may indicate excessive false positives."
        )