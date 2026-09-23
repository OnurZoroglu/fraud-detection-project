"""Cost-sensitive threshold optimization for the fraud detection model.

The default 0.5 classification threshold ignores the fact that missing a
fraud case and raising a false alarm have very different costs. This script
picks the threshold that minimizes total expected cost, given assumed
per-error costs, without touching the test split:

1. 5-fold stratified CV on the training split produces out-of-fold (OOF)
   probabilities from the same model configuration that is served.
2. The cost-optimal threshold is selected on those OOF predictions; the
   per-fold optima are reported to show how stable it is.
3. The selected threshold is evaluated once on the held-out test split,
   using a model trained on the full training split (the served model).

See README.md ("Cost-Sensitive Threshold Optimization") for the resulting
analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from data_prep import get_base_feature_cols, get_train_test_indices, load_dataset_with_graph_features

RANDOM_STATE = 42
N_FOLDS = 5
DEFAULT_THRESHOLD = 0.5
THRESHOLDS = np.round(np.arange(0.001, 1.0, 0.001), 3)

# Assumed costs -- in a real bank these would come from finance/risk teams,
# not be hardcoded. Used here as a reasonable illustrative estimate.
COST_FALSE_NEGATIVE = 500  # average cost of a missed fraud transaction ($)
COST_FALSE_POSITIVE = 5  # average cost of reviewing a false alarm ($)
SENSITIVITY_RATIOS = (100, 50, 20)  # FN:FP cost ratios; 100:1 is the assumption above

PLOT_PATH = "data/cost_analysis.png"


def train_model(X, y):
    """Same configuration as the served baseline model (train_with_graph_features.py)."""
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric="logloss")
    return model.fit(X, y)


def total_cost_by_threshold(y_true, y_proba, fn_cost, fp_cost=COST_FALSE_POSITIVE):
    y_true = np.asarray(y_true)
    predicted = np.asarray(y_proba)[None, :] >= THRESHOLDS[:, None]
    false_negatives = (~predicted & (y_true == 1)).sum(axis=1)
    false_positives = (predicted & (y_true == 0)).sum(axis=1)
    return false_negatives * fn_cost + false_positives * fp_cost


def optimal_threshold(y_true, y_proba, fn_cost):
    """Return (threshold, is_at_grid_edge, cost curve)."""
    costs = total_cost_by_threshold(y_true, y_proba, fn_cost)
    idx = int(np.argmin(costs))
    return THRESHOLDS[idx], idx in (0, len(THRESHOLDS) - 1), costs


def evaluate(y_true, y_proba, threshold, fn_cost):
    y_true = np.asarray(y_true)
    predicted = np.asarray(y_proba) >= threshold
    tp = int((predicted & (y_true == 1)).sum())
    fp = int((predicted & (y_true == 0)).sum())
    fn = int((y_true == 1).sum()) - tp
    return {
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "alerts": tp + fp,
        "cost": fn * fn_cost + fp * COST_FALSE_POSITIVE,
    }


df, v_columns = load_dataset_with_graph_features()
feature_cols = get_base_feature_cols(v_columns)
train_idx, test_idx = get_train_test_indices(df)
X_train, y_train = df.loc[train_idx, feature_cols], df.loc[train_idx, "is_fraud"].to_numpy()
X_test, y_test = df.loc[test_idx, feature_cols], df.loc[test_idx, "is_fraud"].to_numpy()

# --- 1. Out-of-fold probabilities on the training split ---
oof_proba = np.zeros(len(y_train))
fold_ids = np.zeros(len(y_train), dtype=int)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for fold, (fit_idx, holdout_idx) in enumerate(skf.split(X_train, y_train)):
    fold_model = train_model(X_train.iloc[fit_idx], y_train[fit_idx])
    oof_proba[holdout_idx] = fold_model.predict_proba(X_train.iloc[holdout_idx])[:, 1]
    fold_ids[holdout_idx] = fold

# --- 3. The served model, for the single test-split evaluation ---
test_proba = train_model(X_train, y_train).predict_proba(X_test)[:, 1]

# --- 2. Threshold selection on OOF predictions, per cost ratio ---
print(f"Threshold selected on OOF predictions ({N_FOLDS}-fold CV on the training split), "
      f"evaluated once on the test split.\n")
selected = None
for ratio in SENSITIVITY_RATIOS:
    fn_cost = COST_FALSE_POSITIVE * ratio
    threshold, at_edge, oof_costs = optimal_threshold(y_train, oof_proba, fn_cost)
    near_optimal = THRESHOLDS[oof_costs <= oof_costs.min() * 1.05]
    fold_optima = [optimal_threshold(y_train[fold_ids == k], oof_proba[fold_ids == k], fn_cost) for k in range(N_FOLDS)]
    test_result = evaluate(y_test, test_proba, threshold, fn_cost)
    test_default = evaluate(y_test, test_proba, DEFAULT_THRESHOLD, fn_cost)

    print(f"=== FN:FP cost {ratio}:1 (missed fraud ${fn_cost}, false alarm ${COST_FALSE_POSITIVE}) ===")
    print(f"OOF-optimal threshold: {threshold:.3f}" + ("  WARNING: at the edge of the search grid" if at_edge else ""))
    print(f"OOF cost within 5% of its minimum for thresholds {near_optimal.min():.3f}-{near_optimal.max():.3f}")
    print("Per-fold optimal thresholds: " + ", ".join(
        f"{t:.3f}{' (grid edge)' if edge else ''}" for t, edge, _ in fold_optima))
    print(f"Test split @ {threshold:.3f}: recall {test_result['recall']:.3f}, precision {test_result['precision']:.3f}, "
          f"alerts {test_result['alerts']}, total cost ${test_result['cost']:,}")
    print(f"Test split @ {DEFAULT_THRESHOLD}: recall {test_default['recall']:.3f}, "
          f"precision {test_default['precision']:.3f}, alerts {test_default['alerts']}, "
          f"total cost ${test_default['cost']:,} (saving {1 - test_result['cost'] / test_default['cost']:.1%})\n")

    if fn_cost == COST_FALSE_NEGATIVE:
        selected = (threshold, oof_costs)

threshold, oof_costs = selected
plt.figure(figsize=(10, 6))
plt.plot(THRESHOLDS, oof_costs, color="darkred")
plt.axvline(threshold, color="green", linestyle="--", label=f"OOF-optimal threshold: {threshold:.3f}")
plt.axvline(DEFAULT_THRESHOLD, color="gray", linestyle="--", label=f"Default threshold: {DEFAULT_THRESHOLD}")
plt.xscale("log")
plt.xlabel("Decision threshold (log scale)")
plt.ylabel("Total cost on OOF training predictions ($)")
plt.title(f"Total Cost vs. Decision Threshold ({COST_FALSE_NEGATIVE // COST_FALSE_POSITIVE}:1 cost ratio, out-of-fold)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Plot saved to {PLOT_PATH}")
