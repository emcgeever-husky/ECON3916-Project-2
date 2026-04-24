import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="World Bank Project Risk Predictor", layout="wide")

st.title("World Bank Development Project Risk Predictor")
st.markdown(
    "Predicts the probability that a World Bank project will receive a **Satisfactory** outcome "
    "based on project characteristics observable at approval. Built on IEG ratings data (N=12,472, 1956-2024)."
)

# Load or train model
@st.cache_resource
def load_model():
    import os
    from sklearn.ensemble import RandomForestClassifier

    if os.path.exists("model.pkl") and os.path.exists("training_columns.json"):
        model = joblib.load("model.pkl")
        with open("training_columns.json") as f:
            cols = json.load(f)
        return model, cols

    # Retrain if model.pkl not present
    st.info("model.pkl not found — training model from dataset. This takes ~30 seconds.")
    DATA_PATH = "ieg_world_bank_project_performance_ratings_04-19-2026.csv"
    df = pd.read_csv(DATA_PATH, encoding="latin1")

    # Binarize outcome
    sat = {"Highly Satisfactory", "Satisfactory", "Moderately Satisfactory"}
    unsat = {"Highly Unsatisfactory", "Unsatisfactory", "Moderately Unsatisfactory"}
    df = df[df["IEG Outcome"].isin(sat | unsat)].copy()
    df["Outcome_Binary"] = df["IEG Outcome"].apply(lambda x: 1 if x in sat else 0)

    # Era feature
    def assign_era(fy):
        if fy < 1975: return "Bretton Woods Era"
        elif fy < 1990: return "Structural Adjustment Era"
        elif fy < 2000: return "Post-Washington Consensus"
        elif fy < 2010: return "MDG Era"
        else: return "SDG Era"
    df["Approval Era"] = df["Approval FY"].apply(assign_era)
    df["Practice Group"] = df["Practice Group"].fillna("Unknown")

    # Outlier removal
    Q1, Q3 = df["Years_to_Evaluation"].quantile(0.25), df["Years_to_Evaluation"].quantile(0.75) if "Years_to_Evaluation" in df.columns else (None, None)
    if Q1 is not None:
        IQR = Q3 - Q1
        df = df[df["Years_to_Evaluation"] <= Q3 + 1.5 * IQR]

    features = ["WB Region", "Lending Instrument Type", "Agreement Type",
                "Practice Group", "Country / Economy FCS Status",
                "Country / Economy Lending Group", "Approval Era"]
    df_model = df[features + ["Outcome_Binary"]].dropna()
    X = pd.get_dummies(df_model[features])
    y = df_model["Outcome_Binary"]

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X, y)

    cols = list(X.columns)
    joblib.dump(model, "model.pkl")
    with open("training_columns.json", "w") as f:
        json.dump(cols, f)

    return model, cols

model, training_cols = load_model()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Project Characteristics")

region = st.sidebar.selectbox("WB Region", [
    "Africa",
    "East Asia and Pacific",
    "Eastern and Southern Africa",
    "Europe and Central Asia",
    "Latin America and Caribbean",
    "Middle East, North Africa, Afghanistan, and Pakistan",
    "Other",
    "South Asia",
    "Western and Central Africa",
])

lending = st.sidebar.selectbox("Lending Instrument Type", ["IPF", "DPF"])

agreement = st.sidebar.selectbox("Agreement Type", [
    "IBRD", "IDA", "GEF", "RETF", "SPF", "CARB", "MONT", "OTHER"
])

practice = st.sidebar.selectbox("Practice Group", [
    "Infrastructure",
    "Human Development",
    "Equitable Growth, Finance and Institutions",
    "Sustainable Development",
    "Digital",
    "Other",
    "Unknown",
])

fcs = st.sidebar.selectbox("Country FCS Status", ["non-FCS", "FCS"])

lending_group = st.sidebar.selectbox("Country Lending Group", ["IBRD", "IDA", "BLEND", "OTHER"])

era = st.sidebar.selectbox("Approval Era", [
    "Bretton Woods Era",
    "Structural Adjustment Era",
    "Post-Washington Consensus",
    "MDG Era",
    "SDG Era",
], index=4)

st.sidebar.caption("Bretton Woods <1975 | Structural Adjustment 1975-1990 | Post-Washington 1990-2000 | MDG 2000-2010 | SDG 2010+")

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(region, lending, agreement, practice, fcs, lending_group, era, cols):
    row = {
        "WB Region": region,
        "Lending Instrument Type": lending,
        "Agreement Type": agreement,
        "Practice Group": practice,
        "Country / Economy FCS Status": fcs,
        "Country / Economy Lending Group": lending_group,
        "Approval Era": era,
    }
    df_row = pd.DataFrame([row])
    df_encoded = pd.get_dummies(df_row)
    df_aligned = df_encoded.reindex(columns=cols, fill_value=0)
    return df_aligned

X_input = build_features(region, lending, agreement, practice, fcs, lending_group, era, training_cols)

# ── Prediction ────────────────────────────────────────────────────────────────
prob_satisfactory = model.predict_proba(X_input)[0][1]
prob_unsatisfactory = 1 - prob_satisfactory
predicted_class = "Satisfactory" if prob_satisfactory >= 0.5 else "Unsatisfactory"

# ── Display ───────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)

col1.metric("Predicted Outcome", predicted_class)
col2.metric("Probability: Satisfactory", f"{prob_satisfactory:.1%}")
col3.metric("Probability: Unsatisfactory", f"{prob_unsatisfactory:.1%}")

# Risk bar
risk_label = "Low Risk" if prob_satisfactory >= 0.65 else ("Moderate Risk" if prob_satisfactory >= 0.45 else "High Risk")

st.progress(prob_satisfactory)
st.markdown(f"**{risk_label}** — {prob_satisfactory:.1%} predicted probability of satisfactory outcome")

# Interactive probability bar chart
st.markdown("---")
fig, ax = plt.subplots(figsize=(6, 2))
bars = ax.barh(
    ["Unsatisfactory", "Satisfactory"],
    [prob_unsatisfactory, prob_satisfactory],
    color=["#d73027", "#1a9850"],
)
ax.set_xlim(0, 1)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Predicted Probability")
ax.set_title("Outcome Probability Breakdown", fontsize=11)
for bar, val in zip(bars, [prob_unsatisfactory, prob_satisfactory]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=10)
st.pyplot(fig)
plt.close(fig)

# Context note
st.markdown("---")
st.markdown(
    "**Model:** Random Forest (default, 200 trees) | **CV Accuracy:** 72.1% (±0.2%) | "
    "**Training data:** IEG World Bank Project Performance Ratings, April 2026  \n"
    "*Prediction reflects statistical patterns in historical data, not causal relationships. "
    "This model identifies structural risk factors observable at approval (region, instrument, era) "
    "but cannot detect project-level design quality factors the literature identifies as most important. "
    "Use for portfolio-level triage, not individual project decisions.*"
)

# Feature importance mini-table
with st.expander("Top features by importance (Gini impurity decrease)"):
    importance_data = {
        "Feature": [
            "Practice Group: Econ & Finance",
            "Era: SDG (2010+)",
            "Region: Eastern & Southern Africa",
            "Agreement: IDA Credit",
            "Era: Post-Washington Consensus (1990-2000)",
            "Practice Group: Infrastructure",
            "Region: Western & Central Africa",
            "Era: MDG (2000-2010)",
            "Practice Group: Sustainable Dev.",
            "Practice Group: Human Development",
        ],
        "Importance (Gini)": [
            0.04903, 0.04419, 0.04370, 0.04277, 0.04235,
            0.04084, 0.04037, 0.04015, 0.03989, 0.03981,
        ],
    }
    st.caption("Predictive importance only — does not imply causal effect.")
    st.dataframe(pd.DataFrame(importance_data), use_container_width=True)
