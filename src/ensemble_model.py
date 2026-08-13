"""Ensemble of the graph-augmented XGBoost model and the autoencoder.

Combines both models' scores via a weighted average and checks whether the
autoencoder catches any fraud cases the supervised model misses. See
README.md ("Ensemble System") for why the autoencoder ultimately adds no
measurable value here -- its scores overlap almost entirely with XGBoost's.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from xgboost import XGBClassifier
from data_prep import get_base_feature_cols, get_graph_feature_cols, load_dataset_with_graph_features

RANDOM_STATE = 42
TEST_SIZE = 0.2
AUTOENCODER_EPOCHS = 30
AUTOENCODER_BATCH_SIZE = 256

WEIGHTS_TO_TEST = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.0]
WEIGHT_SEARCH_CSV_PATH = "data/ensemble_weight_search.csv"
WEIGHT_SEARCH_PLOT_PATH = "data/ensemble_weight_search.png"


def train_xgboost(X_train, y_train, X_test):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def train_autoencoder(X_train, y_train, X_test):
    """Train on normal transactions only; return reconstruction error on X_test as an anomaly score."""
    X_train_normal = X_train[y_train == 0]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(X_train_scaled.shape[1], activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_scaled, X_train_scaled, epochs=AUTOENCODER_EPOCHS, batch_size=AUTOENCODER_BATCH_SIZE, verbose=0)

    X_test_reconstructed = model.predict(X_test_scaled, verbose=0)
    return np.mean(np.square(X_test_scaled - X_test_reconstructed), axis=1)


def count_complementary_catches(y_test, xgb_scaled, ae_scaled):
    """Count fraud cases XGBoost ranks in its bottom half but the autoencoder ranks in its top decile."""
    fraud_mask = (y_test == 1).values
    xgb_rank = pd.Series(xgb_scaled).rank(pct=True).values
    ae_rank = pd.Series(ae_scaled).rank(pct=True).values
    complementary = fraud_mask & (xgb_rank < 0.5) & (ae_rank > 0.9)
    return complementary.sum(), fraud_mask.sum()


def sweep_ensemble_weights(y_test, xgb_scaled, ae_scaled, weights):
    results = []
    for w_xgb in weights:
        w_ae = 1 - w_xgb
        score = w_xgb * xgb_scaled + w_ae * ae_scaled
        results.append({
            "xgb_weight": w_xgb,
            "autoencoder_weight": w_ae,
            "pr_auc": average_precision_score(y_test, score),
            "roc_auc": roc_auc_score(y_test, score),
        })
        print(f"XGB: {w_xgb:.2f} | AE: {w_ae:.2f}  ->  PR-AUC: {results[-1]['pr_auc']:.4f}, ROC-AUC: {results[-1]['roc_auc']:.4f}")
    return pd.DataFrame(results)


df, v_columns = load_dataset_with_graph_features()
base_feature_cols = get_base_feature_cols(v_columns)
all_feature_cols = get_graph_feature_cols(v_columns)

X, y = df[all_feature_cols], df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

xgb_proba = train_xgboost(X_train, y_train, X_test)
reconstruction_error = train_autoencoder(X_train[base_feature_cols], y_train, X_test[base_feature_cols])

# Both scores are rescaled to [0, 1] so they contribute comparably to the weighted average.
minmax = MinMaxScaler()
xgb_scaled = minmax.fit_transform(xgb_proba.reshape(-1, 1)).flatten()
ae_scaled = minmax.fit_transform(reconstruction_error.reshape(-1, 1)).flatten()

ensemble_score = 0.85 * xgb_scaled + 0.15 * ae_scaled

print("=== Model Comparison ===\n")
print("--- XGBoost only (graph features) ---")
print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, xgb_proba):.4f}")

print("\n--- Autoencoder only ---")
print(f"ROC-AUC: {roc_auc_score(y_test, reconstruction_error):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, reconstruction_error):.4f}")

print("\n--- Ensemble (85% XGBoost + 15% Autoencoder) ---")
print(f"ROC-AUC: {roc_auc_score(y_test, ensemble_score):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, ensemble_score):.4f}")

n_complementary, n_fraud = count_complementary_catches(y_test, xgb_scaled, ae_scaled)
print(f"\nFraud cases XGBoost missed but the autoencoder caught: {n_complementary} / {n_fraud}")

print("\n=== Ensemble Performance Across Weight Combinations ===")
results_df = sweep_ensemble_weights(y_test, xgb_scaled, ae_scaled, WEIGHTS_TO_TEST)
results_df.to_csv(WEIGHT_SEARCH_CSV_PATH, index=False)

plt.figure(figsize=(10, 6))
plt.plot(results_df["xgb_weight"], results_df["pr_auc"], marker="o", label="PR-AUC")
plt.plot(results_df["xgb_weight"], results_df["roc_auc"], marker="s", label="ROC-AUC")
plt.xlabel("XGBoost weight (1 - autoencoder weight)")
plt.ylabel("Score")
plt.title("Ensemble Performance vs. XGBoost Weight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(WEIGHT_SEARCH_PLOT_PATH)
print(f"\nPlot saved to {WEIGHT_SEARCH_PLOT_PATH}")