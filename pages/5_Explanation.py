"""
Streamlit page for case priority explanations.

Provides interactive UI for:
- Selecting individual cases
- Viewing prediction explanations
- Understanding feature contributions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(
    page_title="Case Explanation",
    page_icon="🔍",
    layout="wide",
)

THEME_COLOUR = "#707275"

# Custom CSS
st.markdown(f"""
<style>
.block-container {{
    padding-top: 2rem;
    padding-bottom: 1rem;
}}

div[data-testid="metric-container"] {{
    background-color: #F7F7F8;
    border: 1px solid #E0E0E0;
    padding: 10px 15px;
    border-radius: 8px;
}}

div[data-testid="metric-container"] label {{
    color: {THEME_COLOUR} !important;
    font-weight: 500;
}}

.explanation-card {{
    background-color: #F7F7F8;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}}

.factor-positive {{
    color: #2E7D32;
}}

.factor-negative {{
    color: #C62828;
}}

.priority-urgent {{
    background-color: #FFEBEE;
    border-left: 4px solid #C62828;
}}

.priority-high {{
    background-color: #FFF3E0;
    border-left: 4px solid #EF6C00;
}}

.priority-medium {{
    background-color: #FFF8E1;
    border-left: 4px solid #FBC02D;
}}

.priority-low {{
    background-color: #E8F5E9;
    border-left: 4px solid #2E7D32;
}}
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load case data."""
    path = Path("Data/clean.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


def get_priority_color(priority):
    """Get color for priority level."""
    colors = {
        "Urgent": "#C62828",
        "High": "#EF6C00",
        "Medium": "#FBC02D",
        "Low": "#2E7D32",
    }
    return colors.get(priority, THEME_COLOUR)


def create_probability_chart(probabilities):
    """Create probability bar chart."""
    df = pd.DataFrame([
        {"Priority": k, "Probability": v * 100}
        for k, v in probabilities.items()
    ])

    # Sort by priority order
    priority_order = ["Low", "Medium", "High", "Urgent"]
    df["sort_key"] = df["Priority"].apply(
        lambda x: priority_order.index(x) if x in priority_order else 99
    )
    df = df.sort_values("sort_key")

    colors = [get_priority_color(p) for p in df["Priority"]]

    fig = go.Figure(data=[
        go.Bar(
            x=df["Priority"],
            y=df["Probability"],
            marker_color=colors,
            text=[f"{p:.0f}%" for p in df["Probability"]],
            textposition="auto",
        )
    ])

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def create_contribution_chart(contributions, title="Feature Contributions"):
    """Create horizontal bar chart of feature contributions."""
    if not contributions:
        return None

    df = pd.DataFrame(contributions)
    df["abs_contrib"] = df["contribution"].abs()
    df = df.nlargest(6, "abs_contrib")

    colors = ["#2E7D32" if c > 0 else "#C62828" for c in df["contribution"]]

    fig = go.Figure(data=[
        go.Bar(
            y=df["feature"],
            x=df["contribution"],
            orientation="h",
            marker_color=colors,
            text=[f"{c:+.3f}" for c in df["contribution"]],
            textposition="auto",
        )
    ])

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
        title=dict(text=title, font_size=14),
        xaxis_title="SHAP Contribution",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def render_explanation_card(card_data, explanation_text=None):
    """Render explanation card UI."""
    header = card_data["header"]
    priority = header["priority"]
    priority_class = f"priority-{priority.lower()}"

    st.markdown(f"""
    <div class="explanation-card {priority_class}">
        <h3 style="margin:0; color:{THEME_COLOUR}">Case {header['case_id']}</h3>
        <p style="margin:0.5rem 0;">
            <strong>Predicted Priority:</strong>
            <span style="color:{get_priority_color(priority)}; font-weight:bold;">
                {priority}
            </span>
            <span style="color:#666;"> ({header['confidence_pct']} confidence)</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if explanation_text:
        st.markdown("### Explanation")
        st.info(explanation_text)


def render_factor_list(factors, title, positive=True):
    """Render list of contributing factors."""
    if not factors:
        return

    st.markdown(f"**{title}**")

    for factor in factors:
        sign = "+" if positive else ""
        color = "#2E7D32" if positive else "#C62828"

        st.markdown(f"""
        <div style="padding: 0.5rem; margin: 0.25rem 0; background: #fafafa; border-radius: 4px;">
            <span style="font-weight:500;">{factor['name'].replace('_', ' ').title()}</span>
            <span style="color:#666;"> = {factor['value']}</span>
            <span style="color:{color}; float:right;">({sign}{factor['impact']:.3f})</span>
            <br/>
            <small style="color:#666;">{factor['explanation']}</small>
        </div>
        """, unsafe_allow_html=True)


# Main page content
st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Case Explanation</h1>", unsafe_allow_html=True)
st.write("Understand why cases are classified with specific priority levels.")

# Load data
data = load_data()

if data is None:
    st.error("Could not load case data. Please ensure Data/clean.csv exists.")
    st.stop()

# Sidebar: Case selection
st.sidebar.markdown(f"### Select Case")

# Filter options
priority_filter = st.sidebar.multiselect(
    "Filter by Priority",
    options=["Low", "Medium", "High", "Urgent"],
    default=["High", "Urgent"],
)

if priority_filter:
    filtered_data = data[data["priority"].isin(priority_filter)]
else:
    filtered_data = data

# Case selector
case_ids = filtered_data["case_id"].tolist()
selected_case = st.sidebar.selectbox(
    "Case ID",
    options=case_ids,
    index=0 if case_ids else None,
)

if not selected_case:
    st.warning("No cases match the selected filters.")
    st.stop()

# Get selected case data
case_data = data[data["case_id"] == selected_case].iloc[0]

# Display case info
st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Case Details: {selected_case}</h2>", unsafe_allow_html=True)

# Case summary
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Case Summary")
    st.text_area(
        "Summary",
        value=case_data.get("case_summary", "No summary available"),
        height=100,
        disabled=True,
        label_visibility="collapsed",
    )

with col2:
    st.markdown("### Current Priority")
    priority = case_data.get("priority", "Unknown")
    st.markdown(f"""
    <div style="text-align:center; padding:1rem; background:{get_priority_color(priority)}20;
                border-radius:8px; border:2px solid {get_priority_color(priority)};">
        <span style="font-size:2rem; font-weight:bold; color:{get_priority_color(priority)};">
            {priority}
        </span>
    </div>
    """, unsafe_allow_html=True)

# Feature values
st.markdown("### Case Features")
feature_cols = ["channel", "case_type", "category", "plan_tier", "customer_tenure_months"]
col1, col2, col3, col4, col5 = st.columns(5)

cols = [col1, col2, col3, col4, col5]
for i, feat in enumerate(feature_cols):
    with cols[i]:
        value = case_data.get(feat, "N/A")
        st.metric(feat.replace("_", " ").title(), str(value))

# Explanation section (placeholder until model is integrated)
st.markdown("---")
st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Model Explanation</h2>", unsafe_allow_html=True)

st.info("""
**Note:** This section will show SHAP-based explanations once a trained model is integrated.

To enable full explanations:
1. Train a model using the scripts in `scripts/`
2. Save the model and vectorizer using joblib
3. Update this page to load and use the ExplainabilityCore class

Example integration:
```python
from explainability import ExplainabilityCore
import joblib

model = joblib.load('models/priority_classifier.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

explainer = ExplainabilityCore(
    model=model,
    vectorizer=vectorizer,
    tabular_features=['channel', 'case_type', 'category', 'plan_tier', 'customer_tenure_months']
)

result = explainer.explain(features, case_id=selected_case)
```
""")

# Demo visualization with simulated data
st.markdown("### Demo: Explanation Visualization")
st.caption("Below is a demonstration using simulated data.")

# Simulated probability distribution based on actual priority
demo_probs = {
    "Low": 0.05,
    "Medium": 0.15,
    "High": 0.65,
    "Urgent": 0.15,
}

if priority == "Urgent":
    demo_probs = {"Low": 0.02, "Medium": 0.08, "High": 0.15, "Urgent": 0.75}
elif priority == "High":
    demo_probs = {"Low": 0.05, "Medium": 0.15, "High": 0.70, "Urgent": 0.10}
elif priority == "Medium":
    demo_probs = {"Low": 0.15, "Medium": 0.65, "High": 0.15, "Urgent": 0.05}
elif priority == "Low":
    demo_probs = {"Low": 0.70, "Medium": 0.20, "High": 0.08, "Urgent": 0.02}

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Predicted Probabilities")
    fig = create_probability_chart(demo_probs)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Top Contributing Factors")
    # Simulated contributions
    demo_contributions = [
        {"feature": "case_type", "contribution": 0.25},
        {"feature": "plan_tier", "contribution": 0.18},
        {"feature": "channel", "contribution": -0.08},
        {"feature": "category", "contribution": 0.12},
        {"feature": "customer_tenure_months", "contribution": 0.05},
    ]
    fig = create_contribution_chart(demo_contributions)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Explanations are generated using SHAP (SHapley Additive exPlanations) values.")
