"""SHAP-based interpretability analysis for the trained XGBoost model.

Explaining individual predictions is a practical requirement in banking,
where a "black box" score is rarely acceptable. See README.md ("Model
Interpretability: SHAP Analysis") for the finding that the SQL-derived
behavioral features rank among the most influential.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from data_prep import load_dataset

MODEL_PATH = "models/xgboost_fraud_model.pkl"
FEATURE_COLS_PATH = "models/feature_columns.pkl"

RANDOM_STATE = 42  # must match the split used during training
TEST_SIZE = 0.2
SHAP_SAMPLE_SIZE = 2000  # full test set is too slow for TreeExplainer

PLOT_PATH = "data/shap_summary.png"
IMPORTANCE_CSV_PATH = "data/feature_importance.csv"

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)

df, _ = load_dataset()
X, y = df[feature_cols], df["is_fraud"]
_, X_test, _, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_sample = X_test.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"SHAP summary plot saved to {PLOT_PATH}")

importance = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

print("\nTop 10 features:")
print(importance.head(10).to_string(index=False))

importance.to_csv(IMPORTANCE_CSV_PATH, index=False)
print(f"Feature importance saved to {IMPORTANCE_CSV_PATH}")