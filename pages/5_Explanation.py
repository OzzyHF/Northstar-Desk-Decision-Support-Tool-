"""
Streamlit page for case priority explanations.

Provides interactive UI for:
- Selecting individual cases
- Viewing prediction explanations
- Understanding feature contributions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np
from pathlib import Path

from explainability.core import ExplainabilityCore
from explainability.llm_generator import FallbackGenerator

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


@st.cache_resource
def load_model():
    """Load trained model and preprocessor."""
    model_path = Path("models/rf_model.pkl")
    preprocessor_path = Path("Data/processed/preprocessor.pkl")

    if model_path.exists() and preprocessor_path.exists():
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    return None, None


@st.cache_resource
def load_explainer(_model, _preprocessor):
    """Load or create the explainability core."""
    tabular_features = ["channel", "case_type", "category", "plan_tier", "customer_tenure_months"]

    # Extract all feature names from preprocessor in correct order
    feature_names = []

    # Text features (TF-IDF) come first
    if "text" in _preprocessor.named_transformers_:
        text_transformer = _preprocessor.named_transformers_["text"]
        if hasattr(text_transformer, "get_feature_names_out"):
            feature_names.extend(text_transformer.get_feature_names_out().tolist())

    # Categorical features (one-hot encoded)
    if "cat" in _preprocessor.named_transformers_:
        cat_transformer = _preprocessor.named_transformers_["cat"]
        if hasattr(cat_transformer, "get_feature_names_out"):
            feature_names.extend(cat_transformer.get_feature_names_out().tolist())

    # Tenure features (binned)
    if "tenure" in _preprocessor.named_transformers_:
        tenure_transformer = _preprocessor.named_transformers_["tenure"]
        if hasattr(tenure_transformer, "get_feature_names_out"):
            feature_names.extend(tenure_transformer.get_feature_names_out().tolist())

    # Load background data for SHAP
    background_path = Path("Data/splits/X_train.csv")
    background_data = None
    if background_path.exists():
        bg_df = pd.read_csv(background_path)
        bg_sample = bg_df.head(50)
        bg_sample["case_summary"] = bg_sample.get("case_summary", "").fillna("")
        try:
            background_data = _preprocessor.transform(bg_sample)
        except Exception:
            pass

    # Get vectorizer if exists
    vectorizer = None
    if "text" in _preprocessor.named_transformers_:
        vectorizer = _preprocessor.named_transformers_["text"]

    explainer = ExplainabilityCore(
        model=_model,
        vectorizer=vectorizer,
        feature_names=feature_names,
        tabular_features=tabular_features,
        background_data=background_data,
        use_llm=False,  # Use template-based for now (faster)
    )
    return explainer


def get_prediction(model, preprocessor, case_row):
    """Get model prediction and probabilities for a case."""
    # Prepare features matching preprocessor expectations
    feature_cols = ["case_id", "created_at", "case_summary", "channel", "case_type",
                    "category", "plan_tier", "customer_tenure_months"]

    case_df = pd.DataFrame([case_row[feature_cols]])
    case_df["case_summary"] = case_df["case_summary"].fillna("")

    # Transform and predict
    X_transformed = preprocessor.transform(case_df)

    proba = model.predict_proba(X_transformed)[0]
    pred_idx = np.argmax(proba)

    priority_labels = ["Low", "Medium", "High", "Urgent"]
    probabilities = dict(zip(priority_labels, proba.tolist()))
    predicted_priority = priority_labels[pred_idx]

    return predicted_priority, probabilities


def get_feature_importance(model, preprocessor):
    """Get top feature importances from the model."""
    importances = model.feature_importances_

    # Get feature names from preprocessor
    feature_names = []

    # Text features (TF-IDF)
    if hasattr(preprocessor.named_transformers_.get("text", None), "get_feature_names_out"):
        feature_names.extend(preprocessor.named_transformers_["text"].get_feature_names_out().tolist())

    # Categorical features
    if "cat" in preprocessor.named_transformers_:
        cat_transformer = preprocessor.named_transformers_["cat"]
        if hasattr(cat_transformer, "named_steps") and "onehot" in cat_transformer.named_steps:
            feature_names.extend(cat_transformer.named_steps["onehot"].get_feature_names_out().tolist())

    # Tenure bins
    if "tenure" in preprocessor.named_transformers_:
        n_bins = preprocessor.named_transformers_["tenure"].n_bins_[0]
        feature_names.extend([f"tenure_bin_{i}" for i in range(n_bins)])

    # Match importances to names
    contributions = []
    for i, name in enumerate(feature_names[:len(importances)]):
        # Simplify feature names for display
        display_name = name
        if name.startswith("x0_") or name.startswith("x1_") or name.startswith("x2_") or name.startswith("x3_"):
            display_name = name[3:]  # Remove prefix

        contributions.append({
            "feature": display_name,
            "contribution": float(importances[i]) if i < len(importances) else 0.0
        })

    # Sort by importance and return top features
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:10]


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

# Model Explanation section
st.markdown("---")
st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Model Prediction</h2>", unsafe_allow_html=True)

# Load model
model, preprocessor = load_model()

if model is None or preprocessor is None:
    st.warning("Model not found. Run `python scripts/s_01_preprocess.py` and `python scripts/s_06_train_rf.py` to train.")
    st.stop()

# Get real prediction and explanation
try:
    # Basic prediction for probabilities chart
    predicted_priority, probabilities = get_prediction(model, preprocessor, case_data)

    # Prepare features for explainer
    feature_cols = ["case_id", "created_at", "case_summary", "channel", "case_type",
                    "category", "plan_tier", "customer_tenure_months"]
    case_df = pd.DataFrame([case_data[feature_cols]])
    case_df["case_summary"] = case_df["case_summary"].fillna("")

    # Transform for model
    X_transformed = preprocessor.transform(case_df)

    # Get raw feature values for explanation
    raw_features = {
        "channel": case_data.get("channel"),
        "case_type": case_data.get("case_type"),
        "category": case_data.get("category"),
        "plan_tier": case_data.get("plan_tier"),
        "customer_tenure_months": case_data.get("customer_tenure_months"),
    }

    # Generate SHAP-based explanation
    explainer = load_explainer(model, preprocessor)
    explanation_result = explainer.explain(
        X=X_transformed,
        case_id=str(selected_case),
        raw_features=raw_features,
        generate_text=True,
    )

    # Display natural language explanation
    st.markdown("### Why This Priority?")
    explanation_text = explanation_result.get("explanation", "")
    if explanation_text:
        st.info(explanation_text)
    else:
        st.write("Explanation generation unavailable.")

    # Show prediction comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Model Predicted Probabilities")
        fig = create_probability_chart(probabilities)
        st.plotly_chart(fig, use_container_width=True)

        # Show if prediction matches actual
        if predicted_priority == priority:
            st.success(f"Model prediction ({predicted_priority}) matches actual priority")
        else:
            st.warning(f"Model predicts **{predicted_priority}**, actual is **{priority}**")

    with col2:
        st.markdown("#### Top Feature Contributions")
        # Use SHAP-based contributions from explainer
        card_data = explanation_result.get("card_data", {})
        positive_factors = card_data.get("top_positive_factors", [])
        negative_factors = card_data.get("top_negative_factors", [])

        if positive_factors or negative_factors:
            contributions = []
            for f in positive_factors:
                contributions.append({"feature": f["name"], "contribution": f["impact"]})
            for f in negative_factors:
                contributions.append({"feature": f["name"], "contribution": f["impact"]})

            fig = create_contribution_chart(contributions, title="SHAP Contributions")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to feature importance
            feature_contributions = get_feature_importance(model, preprocessor)
            fig = create_contribution_chart(feature_contributions, title="Feature Importance")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # Show contributing factors detail
    if positive_factors:
        st.markdown("---")
        st.markdown("### Contributing Factors")
        render_factor_list(positive_factors, "Factors Increasing Priority", positive=True)

    if negative_factors:
        render_factor_list(negative_factors, "Factors Decreasing Priority", positive=False)

except Exception as e:
    st.error(f"Error generating prediction: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Footer
st.markdown("---")
st.caption("Explanations are generated using SHAP (SHapley Additive exPlanations) values with natural language generation.")
