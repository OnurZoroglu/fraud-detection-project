"""Train and compare baseline fraud detection models.

Compares three approaches to class imbalance -- SMOTE oversampling,
XGBoost's scale_pos_weight, and Logistic Regression's class_weight -- and
persists the best-performing model (XGBoost with scale_pos_weight) along
with its precision-recall threshold curve. See README.md ("Classical
Machine Learning Baseline" and "Imbalanced Data: SMOTE vs. Class
Weighting") for the resulting comparison.
"""

import os
import joblib
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from data_prep import get_base_feature_cols, load_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_OUTPUT_PATH = "models/xgboost_fraud_model.pkl"
FEATURE_COLS_OUTPUT_PATH = "models/feature_columns.pkl"
THRESHOLD_PLOT_PATH = "data/threshold_analysis.png"


def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_true, y_proba):.4f}")


df, v_columns = load_dataset()
feature_cols = get_base_feature_cols(v_columns)
X, y = df[feature_cols], df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train size: {X_train.shape}, test size: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean() * 100:.3f}%")
print(f"Test fraud rate:  {y_test.mean() * 100:.3f}%")

# --- SMOTE oversampling ---
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"\nSMOTE: train size {X_train.shape} -> {X_train_smote.shape} "
      f"(fraud rate {y_train.mean() * 100:.3f}% -> {y_train_smote.mean() * 100:.3f}%)")

xgb_smote = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")  # no scale_pos_weight: SMOTE already balances
xgb_smote.fit(X_train_smote, y_train_smote)
xgb_smote_proba = xgb_smote.predict_proba(X_test)[:, 1]
evaluate("XGBoost + SMOTE", y_test, xgb_smote.predict(X_test), xgb_smote_proba)

# --- Logistic Regression baseline ---
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
evaluate("Logistic Regression", y_test, lr.predict(X_test_scaled), lr_proba)

# --- XGBoost with scale_pos_weight (best performer, persisted below) ---
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric="logloss")
xgb.fit(X_train, y_train)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
evaluate("XGBoost", y_test, xgb.predict(X_test), xgb_proba)

os.makedirs("models", exist_ok=True)
joblib.dump(xgb, MODEL_OUTPUT_PATH)
joblib.dump(feature_cols, FEATURE_COLS_OUTPUT_PATH)
print(f"\nModel saved to {MODEL_OUTPUT_PATH}")

precision, recall, thresholds = precision_recall_curve(y_test, xgb_proba)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Decision threshold")
plt.ylabel("Score")
plt.title("XGBoost: Precision-Recall vs. Decision Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(THRESHOLD_PLOT_PATH)
print(f"Plot saved to {THRESHOLD_PLOT_PATH}")