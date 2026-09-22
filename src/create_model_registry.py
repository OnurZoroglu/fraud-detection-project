"""Create the model registry (models/model_registry.json).

Records every model trained during this project, along with its metrics
and production status, so the active model version served by the API
(see api_service.py) is explicit and auditable.
"""

import json
import os

REGISTRY_PATH = "models/model_registry.json"

registry = {
    "active_version": "2.0.0",
    "models": [
        {
            "version": "1.0.0",
            "name": "Logistic Regression Baseline",
            "file_path": None,  # not persisted, exploratory run only
            "feature_set": "base",
            "training_date": "2026-09-23",
            "metrics": {"pr_auc": 0.720, "roc_auc": 0.972},
            "notes": "Linear baseline, balanced via class_weight='balanced'.",
        },
        {
            "version": "2.0.0",
            "name": "XGBoost Baseline",
            "file_path": "models/xgboost_baseline_model.pkl",
            "feature_set": "base",
            "training_date": "2026-09-23",
            "metrics": {"pr_auc": 0.876, "roc_auc": 0.975},
            "notes": "Balanced via scale_pos_weight, base feature set (amount, SQL features, V1-V28).",
        },
        {
            "version": "3.0.0",
            "name": "XGBoost + Graph Features (Final)",
            "file_path": "models/xgboost_final_model.pkl",
            "feature_set": "base + graph",
            "training_date": "2026-09-23",
            "metrics": {"pr_auc": 0.928, "roc_auc": 0.987},
            "notes": (
                "Adds degree_centrality and community_size graph features. Highest PR-AUC of "
                "any model, but not used in production: graph features require a full "
                "recomputation of the card-merchant network and are impractical to compute "
                "per-request. The baseline (2.0.0) is served instead."
            ),
        },
        {
            "version": "3.1.0-experimental",
            "name": "Ensemble (XGBoost + Autoencoder)",
            "file_path": None,  # reproduced from code, not persisted separately
            "feature_set": "base + graph",
            "training_date": "2026-09-23",
            "metrics": {"pr_auc": 0.919, "roc_auc": 0.980},
            "notes": "Adding the autoencoder did not improve PR-AUC (see Week 5 analysis). Not promoted to production.",
        },
    ],
}

os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2, ensure_ascii=False)

print(f"Model registry created: {REGISTRY_PATH}")
print(f"Active version: {registry['active_version']}")
print(f"Total registered models: {len(registry['models'])}")