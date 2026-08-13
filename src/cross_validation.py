"""5-fold stratified cross-validation for the XGBoost baseline model.

A single train/test split can give a misleadingly optimistic or pessimistic
performance estimate, especially with severe class imbalance. Stratified
K-fold keeps the fraud ratio consistent across folds and reports a mean and
standard deviation instead of a single point estimate. See README.md
("Cross-Validation") for the resulting analysis.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from data_prep import get_base_feature_cols, load_dataset

N_SPLITS = 5
RANDOM_STATE = 42


def run_cross_validation(X, y, n_splits: int = N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pr_auc_scores, roc_auc_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric="logloss")
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, proba)
        roc_auc = roc_auc_score(y_val, proba)
        pr_auc_scores.append(pr_auc)
        roc_auc_scores.append(roc_auc)
        print(f"Fold {fold}: PR-AUC = {pr_auc:.4f}, ROC-AUC = {roc_auc:.4f}")

    return np.array(pr_auc_scores), np.array(roc_auc_scores)


df, v_columns = load_dataset()
feature_cols = get_base_feature_cols(v_columns)
X = df[feature_cols].values
y = df["is_fraud"].values

print(f"=== {N_SPLITS}-Fold Stratified Cross-Validation (XGBoost) ===\n")
pr_auc_scores, roc_auc_scores = run_cross_validation(X, y)

print(f"\nMean PR-AUC:  {pr_auc_scores.mean():.4f} (± {pr_auc_scores.std():.4f})")
print(f"Mean ROC-AUC: {roc_auc_scores.mean():.4f} (± {roc_auc_scores.std():.4f})")