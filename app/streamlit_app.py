import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from config import (
    READMISSION_DATASET,
    ED_FORECAST_DATASET,
    READMISSION_MODEL,
    ED_FORECAST_MODEL,
)

st.set_page_config(
    page_title="Predictive Healthcare Intelligence",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
    }

    html, body {
        color: black !important;
    }

    h1, h2, h3, h4, h5, h6,
    p, label {
        color: black !important;
    }

    .main-title {
        font-size: 30px;
        font-weight: 700;
        color: black !important;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 15px;
        color: black !important;
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #d1d5db;
        padding: 14px;
        border-radius: 14px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: black !important;
    }

    div[data-testid="stMetricValue"] {
        color: black !important;
    }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_readmission_data():
    return pd.read_csv(READMISSION_DATASET)


@st.cache_data
def load_ed_data():
    return pd.read_csv(ED_FORECAST_DATASET)


@st.cache_resource
def load_readmission_model():
    return joblib.load(READMISSION_MODEL)


@st.cache_resource
def load_ed_model():
    return joblib.load(ED_FORECAST_MODEL)


def get_readmission_model():
    package = load_readmission_model()

    if isinstance(package, dict):
        return package["model"], package.get("threshold", 0.50)

    return package, 0.50


def risk_band(score):
    if score >= 0.60:
        return "Critical"
    elif score >= 0.50:
        return "High"
    elif score >= 0.35:
        return "Medium"
    else:
        return "Low"


risk_colors = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#ef4444",
    "Critical": "#7f1d1d",
}


readmission_df = load_readmission_data()
ed_df = load_ed_data()
readmission_model, threshold = get_readmission_model()
ed_model = load_ed_model()

drop_cols = [
    "readmitted_within_30d",
    "readmission_reason",
    "patient_id",
    "admission_id",
    "lab_id",
    "medication_id",
    "diagnosis_id",
    "mrn",
    "first_name",
    "last_name",
    "admission_date",
    "discharge_date",
    "registered_date",
    "date_of_birth",
    "total_cost_usd",
    "insurance_paid_usd",
    "zip_code",
]

X_readmission = readmission_df.drop(
    columns=[col for col in drop_cols if col in readmission_df.columns]
)

datetime_cols = [
    col for col in X_readmission.columns
    if "date" in col.lower() or "datetime" in col.lower()
]

X_readmission = X_readmission.drop(columns=datetime_cols, errors="ignore")

readmission_df["risk_score"] = readmission_model.predict_proba(X_readmission)[:, 1]
readmission_df["risk_band"] = readmission_df["risk_score"].apply(risk_band)

ed_df["week_start"] = pd.to_datetime(ed_df["week_start"], errors="coerce")
ed_df = ed_df.dropna(subset=["week_start"]).sort_values("week_start")

X_ed = ed_df.drop(columns=["week_start", "ed_arrivals"], errors="ignore")
ed_df["predicted_ed_arrivals"] = ed_model.predict(X_ed)

total_patients = readmission_df["patient_id"].nunique()
total_admissions = len(readmission_df)
readmission_rate = readmission_df["readmitted_within_30d"].mean() * 100
total_ed_visits = int(ed_df["ed_arrivals"].sum())

st.markdown(
    """
    <div class="main-title">Predictive Healthcare Intelligence Dashboard</div>
    <div class="subtitle">
        Single-screen overview of readmission risk and emergency department demand.
    </div>
    """,
    unsafe_allow_html=True,
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("Total Patients", f"{total_patients:,}")
kpi2.metric("Total Admissions", f"{total_admissions:,}")
kpi3.metric("Readmission Rate", f"{readmission_rate:.1f}%")
kpi4.metric("ED Visits", f"{total_ed_visits:,}")
kpi5.metric("Model Threshold", f"{threshold:.2f}")

left, middle, right = st.columns([1.1, 1.1, 1.3])

with left:
    st.subheader("Risk Band Distribution")

    risk_order = ["Low", "Medium", "High", "Critical"]

    risk_counts = (
        readmission_df["risk_band"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .reset_index()
    )

    risk_counts.columns = ["risk_band", "count"]

    fig_risk = px.bar(
        risk_counts,
        x="risk_band",
        y="count",
        color="risk_band",
        color_discrete_map=risk_colors,
        text="count",
    )

    fig_risk.update_traces(
        textposition="inside",
        textfont=dict(color="black", size=11),
    )

    fig_risk.update_layout(
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(color="black"),
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=None,
        yaxis_title="Patients",
    )

    fig_risk.update_xaxes(tickfont=dict(color="black"))
    fig_risk.update_yaxes(tickfont=dict(color="black"))

    st.plotly_chart(fig_risk, use_container_width=True)

with middle:
    st.subheader("Readmission Outcome")

    outcome_counts = (
        readmission_df["readmitted_within_30d"]
        .map({0: "Not Readmitted", 1: "Readmitted"})
        .value_counts()
        .reset_index()
    )

    outcome_counts.columns = ["outcome", "count"]

    fig_outcome = px.pie(
        outcome_counts,
        names="outcome",
        values="count",
        hole=0.55,
    )

    fig_outcome.update_layout(
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.1),
    )

    fig_outcome.update_traces(
        textfont=dict(color="black"),
    )

    st.plotly_chart(fig_outcome, use_container_width=True)

with right:
    st.subheader("Actual vs Predicted ED Arrivals")

    weekly = ed_df.groupby("week_start").agg(
        actual_arrivals=("ed_arrivals", "sum"),
        predicted_arrivals=("predicted_ed_arrivals", "sum"),
    ).reset_index()

    weekly["actual_4wk_avg"] = (
        weekly["actual_arrivals"]
        .rolling(window=4, min_periods=1)
        .mean()
    )

    weekly["predicted_4wk_avg"] = (
        weekly["predicted_arrivals"]
        .rolling(window=4, min_periods=1)
        .mean()
    )

    fig_ed = px.line(
        weekly,
        x="week_start",
        y=["actual_4wk_avg", "predicted_4wk_avg"],
        labels={
            "value": "ED Arrivals",
            "week_start": "Week",
            "variable": "Metric",
        },
    )

    fig_ed.update_layout(
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=None,
        yaxis_title="ED Arrivals",
        legend_title=None,
        legend=dict(orientation="h", y=-0.2),
    )

    fig_ed.update_xaxes(tickfont=dict(color="black"))
    fig_ed.update_yaxes(tickfont=dict(color="black"))

    st.plotly_chart(fig_ed, use_container_width=True)

bottom_left, bottom_right = st.columns([1.2, 1.8])

with bottom_left:
    st.subheader("Risk Summary")

    risk_summary = risk_counts.copy()
    risk_summary["percentage"] = (
        risk_summary["count"] / risk_summary["count"].sum() * 100
    ).round(1)

    st.dataframe(
        risk_summary,
        use_container_width=True,
        height=190,
        hide_index=True,
    )

with bottom_right:
    st.subheader("Top High-Risk Patients")

    high_risk = readmission_df[
        readmission_df["risk_band"].isin(["High", "Critical"])
    ].copy()

    columns_to_show = [
        col for col in [
            "patient_id",
            "admission_id",
            "hospital",
            "ward",
            "age",
            "risk_score",
            "risk_band",
            "charlson_comorbidity_index",
            "max_news2_score",
            "abnormal_labs",
            "high_alert_medications",
        ]
        if col in high_risk.columns
    ]

    if not high_risk.empty:
        high_risk["risk_score"] = high_risk["risk_score"].round(3)

        st.dataframe(
            high_risk[columns_to_show]
            .sort_values("risk_score", ascending=False)
            .head(10),
            use_container_width=True,
            height=190,
            hide_index=True,
        )
    else:
        st.info("No high-risk patients found.")