"""FastAPI service exposing the fraud detection model as a REST endpoint.

The active model version is resolved from models/model_registry.json at
startup. See README.md ("Model Versioning") for why the production model
is the baseline (no graph features) rather than the highest-scoring one:
graph-based features require a full re-computation of the card-merchant
network and are not practical to compute per-request.
"""

import json
import os
from datetime import datetime
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REGISTRY_PATH = "models/model_registry.json"
LOG_PATH = "logs/prediction_log.jsonl"
ALERT_THRESHOLD = 0.10  # from the cost-sensitive threshold analysis, see README.md


def load_active_model():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
    model_info = next(m for m in registry["models"] if m["version"] == registry["active_version"])

    model = joblib.load(model_info["file_path"])
    feature_cols_path = (
        "models/final_feature_columns.pkl"
        if model_info["feature_set"] == "base + graph"
        else "models/baseline_feature_columns.pkl"
    )
    feature_cols = joblib.load(feature_cols_path)
    return registry, model_info, model, feature_cols


registry, active_model_info, model, feature_cols = load_active_model()

app = FastAPI(title="Fraud Detection API", version=registry["active_version"])
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


class TransactionInput(BaseModel):
    amount: float = Field(..., description="Transaction amount")
    txn_count_last_hour: int = Field(..., description="Number of transactions in the last hour")
    avg_amount_last_hour: float = Field(..., description="Average transaction amount in the last hour")
    time_since_last_txn: float = Field(..., description="Seconds since the previous transaction")
    v_features: dict = Field(..., description="PCA components v1..v28, e.g. {'v1': 0.1, 'v2': -0.5, ...}")


class PredictionOutput(BaseModel):
    fraud_probability: float
    is_fraud_alert: bool
    alert_threshold: float
    model_version: str
    timestamp: str


def log_prediction(input_data: dict, prediction: dict) -> None:
    """Append the request/response pair to the prediction log for drift monitoring."""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({**input_data, **prediction}) + "\n")


@app.get("/")
def root():
    return {"service": "Fraud Detection API", "model_version": registry["active_version"], "status": "active"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/models")
def list_models():
    """List all registered model versions and their metrics."""
    return {
        "active_version": registry["active_version"],
        "active_model_details": active_model_info,
        "all_versions": registry["models"],
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_fraud(transaction: TransactionInput):
    try:
        row = {
            "amount": transaction.amount,
            "txn_count_last_hour": transaction.txn_count_last_hour,
            "avg_amount_last_hour": transaction.avg_amount_last_hour,
            "time_since_last_txn": transaction.time_since_last_txn,
            **transaction.v_features,
        }
        X = pd.DataFrame([row])[feature_cols]  # enforce training-time column order

        proba = float(model.predict_proba(X)[0][1])
        result = {
            "fraud_probability": round(proba, 4),
            "is_fraud_alert": proba >= ALERT_THRESHOLD,
            "alert_threshold": ALERT_THRESHOLD,
            "model_version": registry["active_version"],
            "timestamp": datetime.now().isoformat(),
        }

        log_prediction(row, result)
        return result

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))