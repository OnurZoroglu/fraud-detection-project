"""Interactive Streamlit dashboard for the fraud detection model.

Lets a user enter transaction details and see the model's fraud
probability live. Uses the same baseline model served by api_service.py
(see README.md, "Model Versioning", for why the baseline rather than the
graph-augmented model is used for single-transaction scoring).
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "models/xgboost_baseline_model.pkl"
FEATURE_COLS_PATH = "models/baseline_feature_columns.pkl"
ALERT_THRESHOLD = 0.10  # from the cost-sensitive threshold analysis, see README.md


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, feature_cols


st.set_page_config(page_title="Fraud Detection Demo", page_icon="🔍", layout="centered")
st.title("Credit Card Fraud Detection — Live Demo")
st.markdown(
    "Enter transaction details below. The model scores the transaction using the same "
    "baseline XGBoost pipeline served by the production API (`api_service.py`)."
)

model, feature_cols = load_model()

st.subheader("Transaction Details")
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=1.0)
    txn_count_last_hour = st.number_input("Transactions in last hour", min_value=0, value=2, step=1)

with col2:
    avg_amount_last_hour = st.number_input("Avg. amount in last hour ($)", min_value=0.0, value=100.0, step=1.0)
    time_since_last_txn = st.number_input("Seconds since last transaction", min_value=0.0, value=300.0, step=1.0)

with st.expander("PCA components (v1–v28)", expanded=False):
    st.caption(
        "These are anonymized components from the original dataset (see README.md, "
        "'Dataset'). Defaults to 0 for all; adjust individual values to explore the model's "
        "sensitivity, or use the presets below."
    )
    preset = st.radio("Preset", ["All zeros", "Random (fixed seed)"], horizontal=True)
    if preset == "Random (fixed seed)":
        v_values = np.random.RandomState(0).normal(0, 1, size=28)
    else:
        v_values = np.zeros(28)

    v_features = {}
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            v_features[f"v{i}"] = st.number_input(f"v{i}", value=float(v_values[i - 1]), key=f"v{i}", format="%.3f")

if st.button("Score Transaction", type="primary"):
    row = {
        "amount": amount,
        "txn_count_last_hour": txn_count_last_hour,
        "avg_amount_last_hour": avg_amount_last_hour,
        "time_since_last_txn": time_since_last_txn,
        **v_features,
    }
    X = pd.DataFrame([row])[feature_cols]
    proba = float(model.predict_proba(X)[0][1])
    is_alert = proba >= ALERT_THRESHOLD

    st.subheader("Result")
    st.metric("Fraud probability", f"{proba:.2%}")

    if is_alert:
        st.error(f"⚠️ FLAGGED — score exceeds the alert threshold ({ALERT_THRESHOLD:.0%})")
    else:
        st.success(f"✅ Not flagged — score is below the alert threshold ({ALERT_THRESHOLD:.0%})")

    st.caption(
        "The alert threshold (0.10) comes from the cost-sensitive threshold optimization "
        "in cost_analysis.py, not the default 0.5 cutoff. See README.md for the reasoning."
    )