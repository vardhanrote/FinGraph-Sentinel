
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\ADMIN\FinGraph-Sentinel")
RESULTS_DIR = PROJECT_DIR / "data" / "results"
PHASE17_DIR = RESULTS_DIR / "phase17"

st.set_page_config(
    page_title="FinGraph-Sentinel",
    page_icon="",
    layout="wide"
)

@st.cache_data
def load_data():
    predictions = pd.read_csv(
        PHASE17_DIR / "phase18_risk_predictions.csv"
    )
    systemic = pd.read_csv(
        RESULTS_DIR / "dynamic_systemic_risk.csv"
    )
    shock = pd.read_csv(
        RESULTS_DIR / "dynamic_shock_propagation.csv"
    )

    for df in [predictions, systemic, shock]:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    return predictions, systemic, shock


predictions, systemic, shock = load_data()

st.title("FinGraph-Sentinel")
st.subheader("Adaptive Regime-Aware Contagion Early-Warning System")

st.info(
    "Research prototype: model outputs are exploratory "
    "and require external and prospective validation."
)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Dashboard Filters")

categories = sorted(
    predictions["risk_category"].dropna().unique()
)

selected_categories = st.sidebar.multiselect(
    "Risk categories",
    categories,
    default=categories
)

min_date = predictions["date"].min().date()
max_date = predictions["date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_predictions = predictions[
    (predictions["risk_category"].isin(selected_categories)) &
    (predictions["date"].dt.date >= start_date) &
    (predictions["date"].dt.date <= end_date)
].copy()

# -----------------------------
# KPIs
# -----------------------------
st.divider()
st.subheader("Key Performance Indicators")

total_observations = len(filtered_predictions)

total_alerts = int(
    (filtered_predictions["alert_status"] == "ALERT").sum()
)

alert_percentage = (
    total_alerts / total_observations * 100
    if total_observations > 0 else 0
)

maximum_probability = (
    filtered_predictions["risk_probability"].max()
    if total_observations > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Observations", total_observations)
col2.metric("Alerts", total_alerts)
col3.metric("Alert Percentage", f"{alert_percentage:.2f}%")
col4.metric("Maximum Risk Probability", f"{maximum_probability:.4f}")

# -----------------------------
# Systemic Risk Trend
# -----------------------------
st.divider()
st.subheader("Systemic Risk Trend")

systemic_trend = (
    systemic.groupby("date")
    .agg(
        mean_systemic_risk=("systemic_risk_score", "mean"),
        maximum_systemic_risk=("systemic_risk_score", "max")
    )
    .reset_index()
)

systemic_trend = systemic_trend[
    (systemic_trend["date"].dt.date >= start_date) &
    (systemic_trend["date"].dt.date <= end_date)
]

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    systemic_trend["date"],
    systemic_trend["mean_systemic_risk"],
    label="Mean systemic risk"
)

ax.plot(
    systemic_trend["date"],
    systemic_trend["maximum_systemic_risk"],
    label="Maximum systemic risk"
)

ax.set_title("Systemic Risk Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Risk Score")
ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("Systemic risk metric explanation"):
    st.write(
        "Mean systemic risk is the average risk score across "
        "companies. Maximum systemic risk represents the highest "
        "company-level score recorded on each date."
    )

# -----------------------------
# Risk Categories
# -----------------------------
st.divider()
st.subheader("Risk Category Distribution")

category_counts = (
    filtered_predictions["risk_category"]
    .value_counts()
    .reindex(["Low", "Moderate", "High", "Critical"])
    .fillna(0)
)

st.bar_chart(category_counts)

# -----------------------------
# Early Warning Alerts
# -----------------------------
st.divider()
st.subheader("Early-Warning Alerts")

alerts_df = filtered_predictions[
    filtered_predictions["alert_status"] == "ALERT"
].sort_values(
    "risk_probability",
    ascending=False
)

st.dataframe(
    alerts_df,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Latest Company Risk
# -----------------------------
st.divider()
st.subheader("Latest Company-Level Systemic Risk")

latest_date = systemic["date"].max()

latest_company_risk = systemic[
    systemic["date"] == latest_date
].sort_values(
    "systemic_risk_score",
    ascending=False
)

st.caption(
    f"Latest available systemic risk date: {latest_date.date()}"
)

st.dataframe(
    latest_company_risk,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Shock Propagation
# -----------------------------
st.divider()
st.subheader("Shock Propagation Analysis")

shock_trend = (
    shock.groupby("date")
    .agg(
        total_network_impact=("total_impact", "sum"),
        maximum_network_impact=("total_impact", "max"),
        total_affected_companies=("affected_companies", "sum"),
        maximum_affected_companies=("affected_companies", "max")
    )
    .reset_index()
)

shock_trend = shock_trend[
    (shock_trend["date"].dt.date >= start_date) &
    (shock_trend["date"].dt.date <= end_date)
]

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Affected Companies Over Time**")

    fig1, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(
        shock_trend["date"],
        shock_trend["maximum_affected_companies"]
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Affected Companies")
    ax1.grid(True, alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.markdown("**Total Network Impact Over Time**")

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.plot(
        shock_trend["date"],
        shock_trend["total_network_impact"]
    )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Impact")
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

with st.expander("Shock propagation explanation"):
    st.write(
        "Affected companies represents the number of companies "
        "impacted by a simulated shock. Total network impact "
        "summarizes the magnitude of simulated propagation. "
        "These are model-derived simulation results and not "
        "observed real-world financial losses."
    )

# -----------------------------
# Methodology
# -----------------------------
st.divider()
st.subheader("Methodology")

st.markdown("""
- **Model:** Logistic Regression
- **Feature group:** Network Change Features
- **Features:** Degree change, betweenness change, PageRank change
- **Alert threshold:** 0.55
- **Purpose:** Exploratory systemic-risk early warning
- **Validation:** Historical walk-forward and robustness analysis
""")

st.warning(
    "This model is a research prototype. Alerts should not "
    "be interpreted as confirmed future financial events."
)

st.caption("FinGraph-Sentinel | Research Prototype")
