"""Healthy Meals churn predictor — Streamlit app.

Inputs are built from category and range metadata saved in churn_model.joblib.
Output is churn probability = 1 - P(renew).
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Healthy Meals Churn Predictor", page_icon="🍽️")

bundle         = joblib.load("churn_model.joblib")
model          = bundle["model"]
num_cols       = bundle["num_cols"]
cat_cols       = bundle["cat_cols"]
cat_categories = bundle.get("cat_categories", {})
num_stats      = bundle.get("num_stats", {})

LABELS = {
    "total_num_sessions": "Total sessions (prior year)",
    "total_session_length": "Total session length (prior year)",
    "active_days": "Active days (prior year)",
    "active_quarters": "Active quarters (prior year)",
    "avg_sessions_per_active_quarter": "Avg sessions per active quarter",
    "age": "Age",
    "tech_comfort_score": "Tech comfort score",
    "education": "Education",
    "income_level": "Income level",
    "device_type": "Device type",
}


def label(col):
    return LABELS.get(col, col.replace("_", " ").title())


st.title("Healthy Meals — Churn Risk Predictor")
st.write("Enter a customer's prior-year activity and demographics to estimate churn probability.")

values = {}
for c in num_cols:
    s = num_stats.get(c, {})
    lo = float(s.get("min", 0.0))
    hi = float(s.get("max", lo + 100.0))
    if hi <= lo:
        hi = lo + 1.0
    default = float(s.get("median", lo))
    step = 0.5 if c == "avg_sessions_per_active_quarter" else 1.0
    values[c] = st.slider(label(c), min_value=lo, max_value=hi, value=default, step=step)

for c in cat_cols:
    choices = cat_categories.get(c, [])
    values[c] = st.selectbox(label(c), choices)

if st.button("Predict churn"):
    row = pd.DataFrame([values])
    p_renew = float(model.predict_proba(row)[:, 1][0])
    p_churn = 1.0 - p_renew

    col1, col2 = st.columns(2)
    col1.metric("Churn probability", f"{p_churn:.1%}")
    col2.metric("Renewal probability", f"{p_renew:.1%}")

    if p_churn >= 0.5:
        st.error("High churn risk")
    else:
        st.success("Lower churn risk")
