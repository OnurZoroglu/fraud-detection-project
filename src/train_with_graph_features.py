"""Compare XGBoost with and without graph-based features, and persist both.

Trains two models on an identical train/test split: one with only
transaction-level features (baseline), one with card-level graph features
(degree centrality, community size) added. Both are persisted -- the
baseline is served in production by api_service.py, while the graph
model is kept for offline analysis. See README.md ("Effect of Graph
Features on Model Performance" and "Model Versioning") for why.
"""

import os
import joblib
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from data_prep import get_base_feature_cols, get_graph_feature_cols, load_dataset_with_graph_features

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASELINE_MODEL_PATH = "models/xgboost_baseline_model.pkl"
BASELINE_FEATURE_COLS_PATH = "models/baseline_feature_columns.pkl"
FINAL_MODEL_PATH = "models/xgboost_final_model.pkl"
FINAL_FEATURE_COLS_PATH = "models/final_feature_columns.pkl"


def train_and_evaluate(X_train, X_test, y_train, y_test, label):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric="logloss")
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)

    print(f"\n=== {label} ===")
    print(classification_report(y_test, model.predict(X_test)))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    return pr_auc, roc_auc, model


df, v_columns = load_dataset_with_graph_features()
base_feature_cols = get_base_feature_cols(v_columns)
all_feature_cols = get_graph_feature_cols(v_columns)

y = df["is_fraud"]
X_base, X_with_graph = df[base_feature_cols], df[all_feature_cols]

# Split on the index once so both feature sets use identical train/test rows.
train_idx, test_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

pr_base, roc_base, xgb_baseline = train_and_evaluate(
    X_base.loc[train_idx], X_base.loc[test_idx], y_train, y_test, "XGBoost (baseline, no graph features)"
)
pr_graph, roc_graph, xgb_final = train_and_evaluate(
    X_with_graph.loc[train_idx], X_with_graph.loc[test_idx], y_train, y_test, "XGBoost (with graph features)"
)

print("\n" + "=" * 50)
print("COMPARISON SUMMARY")
print("=" * 50)
print(f"{'Metric':<10}{'Baseline':<12}{'With graph':<12}{'Delta':<10}")
print(f"{'PR-AUC':<10}{pr_base:<12.4f}{pr_graph:<12.4f}{pr_graph - pr_base:+.4f}")
print(f"{'ROC-AUC':<10}{roc_base:<12.4f}{roc_graph:<12.4f}{roc_graph - roc_base:+.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(xgb_baseline, BASELINE_MODEL_PATH)
joblib.dump(base_feature_cols, BASELINE_FEATURE_COLS_PATH)
print(f"\nBaseline model saved to {BASELINE_MODEL_PATH}")

joblib.dump(xgb_final, FINAL_MODEL_PATH)
joblib.dump(all_feature_cols, FINAL_FEATURE_COLS_PATH)
print(f"Graph-augmented model saved to {FINAL_MODEL_PATH}")