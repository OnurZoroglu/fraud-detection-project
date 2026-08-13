"""Autoencoder-based anomaly detection for credit card fraud.

The model is trained exclusively on normal (non-fraudulent) transactions,
so it learns to reconstruct normal behavior well. Reconstruction error on
unseen transactions is then used as an anomaly score. See README.md
("Autoencoder-Based Anomaly Detection") for why this unsupervised approach
underperforms the supervised XGBoost baseline on this dataset.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_prep import get_base_feature_cols, load_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.2
BOTTLENECK_DIM = 8
MAX_EPOCHS = 50
BATCH_SIZE = 256

PLOT_PATH = "data/autoencoder_error_distribution.png"
MODEL_PATH = "models/autoencoder_fraud.keras"
SCALER_PATH = "models/autoencoder_scaler.pkl"


def build_autoencoder(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(24, activation="elu", activity_regularizer=regularizers.l1(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(16, activation="elu"),
        layers.BatchNormalization(),
        layers.Dense(BOTTLENECK_DIM, activation="linear"),
        layers.Dense(16, activation="elu"),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(24, activation="elu"),
        layers.BatchNormalization(),
        layers.Dense(input_dim, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


df, v_columns = load_dataset()
feature_cols = get_base_feature_cols(v_columns)
X, y = df[feature_cols], df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Trained only on normal transactions -- the label is used to filter rows,
# never as a model input, which is what makes this approach unsupervised.
X_train_normal = X_train[y_train == 0]
print(f"Training set (normal transactions only): {X_train_normal.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)

autoencoder = build_autoencoder(input_dim=X_train_scaled.shape[1])
autoencoder.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-4, verbose=1),
]

autoencoder.fit(
    X_train_scaled,
    X_train_scaled,  # autoencoders reconstruct their own input
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

X_test_reconstructed = autoencoder.predict(X_test_scaled)
reconstruction_error = np.mean(np.square(X_test_scaled - X_test_reconstructed), axis=1)

print("\n=== Autoencoder Results ===")
print(f"ROC-AUC: {roc_auc_score(y_test, reconstruction_error):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, reconstruction_error):.4f}")

plt.figure(figsize=(10, 6))
plt.hist(reconstruction_error[y_test == 0], bins=50, alpha=0.6, label="Normal", color="steelblue", density=True)
plt.hist(reconstruction_error[y_test == 1], bins=50, alpha=0.6, label="Fraud", color="crimson", density=True)
plt.xlabel("Reconstruction error")
plt.ylabel("Density")
plt.yscale("log")  # error distribution is heavily right-skewed
plt.title("Autoencoder: Reconstruction Error (Normal vs. Fraud)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\nPlot saved to {PLOT_PATH}")

autoencoder.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")