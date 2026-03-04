"""
Streamlit page for predicting priority of new cases.

Allows users to input case details and get predictions with explanations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path

from explainability.core import ExplainabilityCore
from explainability.groq_generator import GroqGenerator

st.set_page_config(
    page_title="New Case Prediction",
    page_icon="🆕",
    layout="wide",
)

THEME_COLOUR = "#597AAC"

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

.prediction-card {{
    background-color: #F7F7F8;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
}}
</style>
""", unsafe_allow_html=True)

# Available models configuration
AVAILABLE_MODELS = {
    "Logistic Regression": {
        "file": "lr_model.pkl",
        "description": "Fast linear model, good interpretability",
    },
    "Random Forest": {
        "file": "rf_model.pkl",
        "description": "Ensemble model with best overall performance",
    },
    "Decision Tree": {
        "file": "dt_model.pkl",
        "description": "Simple tree model, easy to visualize",
    },
}

# Feature options
CHANNEL_OPTIONS = ["email", "phone", "webchat", "mobile_app"]
CASE_TYPE_OPTIONS = ["incident", "service_request", "question", "problem"]
CATEGORY_OPTIONS = [
    "billing_payments",
    "account_admin",
    "technical_support",
    "product_feedback",
    "feature_request",
    "general_enquiry",
]
PLAN_TIER_OPTIONS = ["free", "standard", "pro", "enterprise"]

PRIORITY_COLORS = {
    "Low": "#2E7D32",
    "Medium": "#FBC02D",
    "High": "#EF6C00",
    "Urgent": "#C62828",
}


@st.cache_resource
def load_model(model_name: str):
    """Load trained model and preprocessor."""
    model_config = AVAILABLE_MODELS.get(model_name)
    if not model_config:
        return None, None

    model_path = Path(f"models/{model_config['file']}")
    preprocessor_path = Path("Data/processed/preprocessor.pkl")

    if model_path.exists() and preprocessor_path.exists():
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    return None, None


@st.cache_resource
def load_explainer(_model, _preprocessor, model_name: str):
    """Load or create the explainability core."""
    tabular_features = ["channel", "case_type", "category", "plan_tier", "customer_tenure_months"]

    # Extract feature names from preprocessor
    feature_names = []

    if "text" in _preprocessor.named_transformers_:
        text_transformer = _preprocessor.named_transformers_["text"]
        if hasattr(text_transformer, "get_feature_names_out"):
            feature_names.extend(text_transformer.get_feature_names_out().tolist())

    if "cat" in _preprocessor.named_transformers_:
        cat_transformer = _preprocessor.named_transformers_["cat"]
        if hasattr(cat_transformer, "get_feature_names_out"):
            feature_names.extend(cat_transformer.get_feature_names_out().tolist())

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

    vectorizer = None
    if "text" in _preprocessor.named_transformers_:
        vectorizer = _preprocessor.named_transformers_["text"]

    groq_gen = GroqGenerator()
    use_llm = groq_gen.is_available()

    explainer = ExplainabilityCore(
        model=_model,
        vectorizer=vectorizer,
        feature_names=feature_names,
        tabular_features=tabular_features,
        background_data=background_data,
        use_llm=False,
    )
    return explainer, groq_gen if use_llm else None


def get_priority_color(priority):
    """Get color for priority level."""
    return PRIORITY_COLORS.get(priority, "#666666")


def create_probability_chart(probabilities):
    """Create a horizontal bar chart for probabilities."""
    priorities = ["Low", "Medium", "High", "Urgent"]
    probs = [probabilities.get(p, 0) for p in priorities]
    colors = [get_priority_color(p) for p in priorities]

    fig = go.Figure(go.Bar(
        x=probs,
        y=priorities,
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition='inside',
    ))

    fig.update_layout(
        xaxis_title="Probability",
        yaxis_title="Priority",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=250,
        margin=dict(l=20, r=20, t=20, b=40),
    )

    return fig


# Main page content
st.markdown(f"<h1 style='color:{THEME_COLOUR};'>New Case Prediction</h1>", unsafe_allow_html=True)
st.write("Enter case details to predict priority and understand the reasoning.")

# Sidebar: Model selection
st.sidebar.markdown("### Model Selection")

available_model_names = [
    name for name, config in AVAILABLE_MODELS.items()
    if Path(f"models/{config['file']}").exists()
]

if not available_model_names:
    st.error("No models found. Please ensure models are trained.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Prediction Model",
    options=available_model_names,
    index=0,
    help="Select which ML model to use for predictions",
)

st.sidebar.caption(AVAILABLE_MODELS[selected_model]["description"])

# Input form
st.markdown("---")
st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Case Details</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    channel = st.selectbox(
        "Channel",
        options=CHANNEL_OPTIONS,
        index=0,
        help="How did the customer contact support?",
    )

    case_type = st.selectbox(
        "Case Type",
        options=CASE_TYPE_OPTIONS,
        index=0,
        help="What type of case is this?",
    )

    category = st.selectbox(
        "Category",
        options=CATEGORY_OPTIONS,
        index=0,
        help="What category does this case fall under?",
    )

with col2:
    plan_tier = st.selectbox(
        "Plan Tier",
        options=PLAN_TIER_OPTIONS,
        index=1,
        help="Customer's subscription tier",
    )

    customer_tenure_months = st.number_input(
        "Customer Tenure (months)",
        min_value=0,
        max_value=240,
        value=12,
        help="How long has the customer been with us?",
    )

st.markdown("### Case Summary")
case_summary = st.text_area(
    "Describe the case",
    value="",
    height=120,
    placeholder="Enter the customer's issue or request description...",
    help="The text description of the case - this will be analyzed for keywords",
)

# Predict button
st.markdown("---")

if st.button("🔮 Predict Priority", type="primary", use_container_width=True):
    if not case_summary.strip():
        st.warning("Please enter a case summary.")
        st.stop()

    # Load model
    model, preprocessor = load_model(selected_model)

    if model is None or preprocessor is None:
        st.error(f"Could not load model '{selected_model}'.")
        st.stop()

    with st.spinner("Analyzing case..."):
        # Prepare input data
        case_data = {
            "case_id": "NEW-CASE",
            "created_at": pd.Timestamp.now(),
            "case_summary": case_summary.lower(),
            "channel": channel,
            "case_type": case_type,
            "category": category,
            "plan_tier": plan_tier,
            "customer_tenure_months": customer_tenure_months,
        }

        # Create DataFrame
        feature_cols = ["case_id", "created_at", "case_summary", "channel", "case_type",
                        "category", "plan_tier", "customer_tenure_months"]
        case_df = pd.DataFrame([case_data])

        # Transform
        X_transformed = preprocessor.transform(case_df)

        # Predict
        prediction = model.predict(X_transformed)[0]
        probabilities = model.predict_proba(X_transformed)[0]

        priority_map = {0: "Low", 1: "Medium", 2: "High", 3: "Urgent"}
        predicted_priority = priority_map.get(prediction, "Unknown")

        prob_dict = {priority_map[i]: p for i, p in enumerate(probabilities)}

        # Get raw features for explanation
        raw_features = {
            "channel": channel,
            "case_type": case_type,
            "category": category,
            "plan_tier": plan_tier,
            "customer_tenure_months": customer_tenure_months,
        }

        # Generate explanation
        explainer, groq_gen = load_explainer(model, preprocessor, selected_model)
        explanation_result = explainer.explain(
            X=X_transformed,
            case_id="NEW-CASE",
            raw_features=raw_features,
            generate_text=True,
        )

        # Use Groq for better explanations if available
        if groq_gen is not None:
            try:
                groq_result = groq_gen.generate_explanation(
                    explanation_result["structured_output"],
                    validate=True,
                )
                explanation_result["explanation"] = groq_result["explanation"]
            except Exception:
                st.warning("LLM unavailable, using template explanation.")

    # Display results
    st.markdown("---")
    st.markdown(f"<h2 style='color:{THEME_COLOUR};'>Prediction Results</h2>", unsafe_allow_html=True)

    # Main prediction
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        priority_color = get_priority_color(predicted_priority)
        confidence = prob_dict.get(predicted_priority, 0) * 100

        st.markdown(f"""
        <div style="text-align:center; padding:2rem; background:{priority_color}15;
                    border-radius:12px; border:3px solid {priority_color};">
            <p style="font-size:1.2rem; margin-bottom:0.5rem; color:#666;">Predicted Priority</p>
            <p style="font-size:3rem; font-weight:bold; color:{priority_color}; margin:0;">
                {predicted_priority}
            </p>
            <p style="font-size:1.1rem; color:#666; margin-top:0.5rem;">
                {confidence:.1f}% confidence
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Explanation and probabilities
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Why This Priority?")
        explanation_text = explanation_result.get("explanation", "")
        if explanation_text:
            st.info(explanation_text)
        else:
            st.write("Explanation unavailable.")

    with col2:
        st.markdown("### Probability Distribution")
        fig = create_probability_chart(prob_dict)
        st.plotly_chart(fig, use_container_width=True)

    # Contributing factors
    st.markdown("### Contributing Factors")

    structured = explanation_result.get("structured_output", {})
    feature_contribs = structured.get("feature_contributions", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Factors Increasing Priority**")
        positive = feature_contribs.get("positive", [])[:5]
        if positive:
            for f in positive:
                st.markdown(f"""
                <div style="padding:0.5rem; margin:0.25rem 0; background:#E8F5E9; border-radius:4px; border-left:3px solid #2E7D32;">
                    <strong>{f['feature'].replace('_', ' ').title()}</strong> = {f['value']}
                    <span style="color:#2E7D32; float:right;">+{f['contribution']:.3f}</span>
                    <br/><small style="color:#666;">{f.get('explanation', '')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("None identified.")

    with col2:
        st.markdown("**Factors Decreasing Priority**")
        negative = feature_contribs.get("negative", [])[:5]
        if negative:
            for f in negative:
                st.markdown(f"""
                <div style="padding:0.5rem; margin:0.25rem 0; background:#FFEBEE; border-radius:4px; border-left:3px solid #C62828;">
                    <strong>{f['feature'].replace('_', ' ').title()}</strong> = {f['value']}
                    <span style="color:#C62828; float:right;">{f['contribution']:.3f}</span>
                    <br/><small style="color:#666;">{f.get('explanation', '')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("None identified.")

    # Text analysis
    text_analysis = structured.get("text_analysis", {})
    top_tokens = text_analysis.get("top_tokens", [])

    if top_tokens:
        st.markdown("### Key Words Detected")
        token_cols = st.columns(min(len(top_tokens), 5))
        for i, token_info in enumerate(top_tokens[:5]):
            with token_cols[i]:
                token = token_info.get("token", "")
                contrib = token_info.get("contribution", 0)
                color = "#2E7D32" if contrib > 0 else "#C62828"
                st.markdown(f"""
                <div style="text-align:center; padding:0.5rem; background:#f5f5f5; border-radius:4px;">
                    <strong>"{token}"</strong><br/>
                    <span style="color:{color};">{'+' if contrib > 0 else ''}{contrib:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
