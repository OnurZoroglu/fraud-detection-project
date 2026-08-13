"""Cost-sensitive threshold optimization for the fraud detection model.

The default 0.5 classification threshold ignores the fact that missing a
fraud case and raising a false alarm have very different costs. This script
sweeps the decision threshold and finds the one that minimizes total
expected cost, given assumed per-error costs. See README.md
("Cost-Sensitive Threshold Optimization") for the resulting analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from data_prep import get_base_feature_cols, load_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.5

# Assumed costs -- in a real bank these would come from finance/risk teams,
# not be hardcoded. Used here as a reasonable illustrative estimate.
COST_FALSE_NEGATIVE = 500  # average cost of a missed fraud transaction ($)
COST_FALSE_POSITIVE = 5  # average cost of reviewing a false alarm ($)

PLOT_PATH = "data/cost_analysis.png"


def total_cost_by_threshold(y_true, y_proba, thresholds):
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        false_negatives = ((y_true == 1) & (y_pred == 0)).sum()
        false_positives = ((y_true == 0) & (y_pred == 1)).sum()
        costs.append(false_negatives * COST_FALSE_NEGATIVE + false_positives * COST_FALSE_POSITIVE)
    return np.array(costs)


df, v_columns = load_dataset()
feature_cols = get_base_feature_cols(v_columns)
X, y = df[feature_cols], df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric="logloss")
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.01, 1.0, 0.01)
total_costs = total_cost_by_threshold(y_test, y_proba, thresholds)

optimal_idx = np.argmin(total_costs)
optimal_threshold = thresholds[optimal_idx]
default_idx = np.argmin(np.abs(thresholds - DEFAULT_THRESHOLD))

print(f"Optimal threshold: {optimal_threshold:.2f}")
print(f"Total cost at optimal threshold: ${total_costs[optimal_idx]:,.0f}")
print(f"Total cost at default threshold ({DEFAULT_THRESHOLD}): ${total_costs[default_idx]:,.0f}")

plt.figure(figsize=(10, 6))
plt.plot(thresholds, total_costs, color="darkred")
plt.axvline(optimal_threshold, color="green", linestyle="--", label=f"Optimal threshold: {optimal_threshold:.2f}")
plt.axvline(DEFAULT_THRESHOLD, color="gray", linestyle="--", label=f"Default threshold: {DEFAULT_THRESHOLD}")
plt.xlabel("Decision threshold")
plt.ylabel("Total cost ($)")
plt.title("Total Cost vs. Decision Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\nPlot saved to {PLOT_PATH}")