import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide")

THEME_COLOUR = "#597AAC"

st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Model Card</h1>", unsafe_allow_html=True)
st.write("We trained three models and compared them to find the best one for triaging support cases.")

# --- Load results ---
comparison = pd.read_csv(os.path.join("results", "model_comparison.csv"))
reports = pd.read_csv(os.path.join("results", "classification_reports.csv"))

# --- Model Comparison Table ---
st.subheader("Model Comparison")

# Round all numeric columns so they display consistently
comparison_display = comparison.copy()
numeric_cols = ["Accuracy", "Macro F1", "Weighted F1", "Urgent Recall", "Urgent F1"]
comparison_display[numeric_cols] = comparison_display[numeric_cols].round(3)

st.table(comparison_display.set_index("Model"))

# --- Bar chart comparing key metrics ---
st.subheader("Key Metrics by Model")

chart_df = comparison.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Macro F1", "Urgent Recall"],
    var_name="Metric",
    value_name="Score",
)

fig = px.bar(
    chart_df,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    text=chart_df["Score"].round(3),
)
fig.update_layout(yaxis_range=[0, 1], xaxis_title="", yaxis_title="Score")
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# --- Why Logistic Regression was selected ---
lr = comparison[comparison["Model"] == "Logistic Regression"].iloc[0]
st.subheader("Why Logistic Regression?")
st.write(
    f"Although Random Forest scored slightly higher on Accuracy and Macro F1, "
    f"**Logistic Regression** was selected because it has the highest **Urgent Recall ({lr['Urgent Recall']:.3f})**. "
    f"For a triage tool, catching urgent cases is the most important metric - "
    f"a missed urgent case has a much higher cost than a minor misclassification elsewhere. "
    f"LR is also simpler, faster, and easier to interpret."
)

# --- Per-class performance ---
st.subheader("Per-Class Performance")

model_choice = st.selectbox("Select a model", comparison["Model"].tolist())
filtered = reports[reports["Model"] == model_choice].drop(columns=["Model"])

# Round numeric columns and use st.table for consistent alignment
filtered_display = filtered.copy()
for col in ["Precision", "Recall", "F1-Score"]:
    filtered_display[col] = filtered_display[col].round(3)
filtered_display["Support"] = filtered_display["Support"].astype(int)

st.table(filtered_display.set_index("Class"))

# --- Link ---
st.markdown(
    "[View Project on GitHub](https://github.com/OzzyHF/Northstar-Desk-Decision-Support-Tool-)",
    unsafe_allow_html=True,
)
