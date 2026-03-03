"""
Core explainability module that orchestrates the full explanation pipeline.

Pipeline:
1. Model prediction with probabilities
2. SHAP value computation
3. Feature grouping and aggregation
4. Structured JSON output
5. Natural language generation
"""

import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd

from .shap_wrapper import SHAPExplainer
from .feature_grouper import FeatureGrouper, TabularFeatureProcessor
from .structured_output import StructuredOutputBuilder, format_explanation_card
from .llm_generator import LLMGenerator, FallbackGenerator
from .config import PRIORITY_LEVELS

logger = logging.getLogger(__name__)


class ExplainabilityCore:
    """
    Main class orchestrating the explainability pipeline.

    Provides model-agnostic explanations for any sklearn classifier
    with support for mixed tabular + text features.
    """

    def __init__(
        self,
        model,
        vectorizer=None,
        feature_names: Optional[List[str]] = None,
        tabular_features: Optional[List[str]] = None,
        background_data=None,
        use_llm: bool = True,
    ):
        """
        Initialize the explainability core.

        Args:
            model: Trained sklearn classifier with predict_proba
            vectorizer: Fitted TF-IDF vectorizer (optional)
            feature_names: List of all feature names in order
            tabular_features: Names of non-text features
            background_data: Background samples for SHAP (required for some models)
            use_llm: Whether to use LLM for natural language (default True)
        """
        self.model = model
        self.vectorizer = vectorizer
        self.feature_names = feature_names or []
        self.tabular_features = tabular_features or []
        self.background_data = background_data
        self.use_llm = use_llm

        # Initialize components
        self.shap_explainer = SHAPExplainer(model, background_data)
        self.feature_grouper = FeatureGrouper(vectorizer) if vectorizer else None
        self.tabular_processor = TabularFeatureProcessor(
            self.tabular_features,
            categorical_columns=["channel", "case_type", "category", "plan_tier"],
        )
        self.output_builder = StructuredOutputBuilder()

        if use_llm:
            self.llm_generator = LLMGenerator()
        else:
            self.llm_generator = FallbackGenerator()

        # Calculate where text features start
        self.text_feature_start_idx = len(self.tabular_features)

    def explain(
        self,
        X: Union[np.ndarray, pd.DataFrame, pd.Series],
        case_id: str = "UNKNOWN",
        raw_features: Optional[Dict] = None,
        generate_text: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate complete explanation for a single prediction.

        Args:
            X: Input features (transformed, ready for model)
            case_id: Case identifier for output
            raw_features: Original feature values before transformation
            generate_text: Whether to generate natural language

        Returns:
            Dict containing structured output and optional text explanation
        """
        # Ensure proper format
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        elif isinstance(X, pd.DataFrame):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get prediction and probabilities
        prediction_raw = self.model.predict(X)[0]
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            # Find the predicted class index for SHAP
            predicted_idx = list(classes).index(prediction_raw)
            # Convert to string labels for output
            probabilities = {
                PRIORITY_LEVELS[int(c)] if isinstance(c, (int, np.integer)) else c: p
                for c, p in zip(classes, proba.tolist())
            }
        else:
            predicted_idx = 0
            probabilities = {prediction_raw: 1.0}

        # Map prediction to string if numeric
        if isinstance(prediction_raw, (int, np.integer)):
            prediction = PRIORITY_LEVELS[int(prediction_raw)]
        else:
            prediction = prediction_raw

        # Compute SHAP values
        shap_result = self.shap_explainer.explain_single(X, class_idx=predicted_idx)
        shap_values = shap_result["shap_values"]

        # Process tabular features
        tabular_contribs = self._process_tabular_features(
            shap_values,
            raw_features,
        )

        # Process text features if vectorizer exists
        text_analysis = {"semantic_groups": {}, "top_tokens": []}
        if self.feature_grouper and self.text_feature_start_idx < len(shap_values):
            text_analysis = self.feature_grouper.aggregate_shap_values(
                shap_values,
                self.feature_names,
                self.text_feature_start_idx,
            )

        # Build structured output
        structured_output = self.output_builder.build(
            case_id=case_id,
            prediction=prediction,
            probabilities=probabilities,
            feature_contributions=tabular_contribs,
            text_analysis=text_analysis,
            raw_features=raw_features,
        )

        result = {
            "structured_output": structured_output,
            "card_data": format_explanation_card(structured_output),
        }

        # Generate natural language if requested
        if generate_text:
            if isinstance(self.llm_generator, LLMGenerator):
                text_result = self.llm_generator.generate_explanation(structured_output)
            else:
                text_result = {
                    "explanation": self.llm_generator.generate(structured_output),
                    "validation": {"is_valid": True, "warnings": []},
                    "source": "template",
                }

            result["explanation"] = text_result["explanation"]
            result["explanation_source"] = text_result.get("source", "unknown")
            result["validation"] = text_result.get("validation", {})

        return result

    def _process_tabular_features(
        self,
        shap_values: np.ndarray,
        raw_features: Optional[Dict],
    ) -> List[Dict]:
        """Process tabular feature contributions by aggregating one-hot encoded columns."""
        contributions = []

        # Get all feature names from the model/preprocessor if available
        all_feature_names = self.feature_names

        # If we have feature names, aggregate SHAP values by original feature
        if all_feature_names and len(all_feature_names) == len(shap_values):
            for feature_name in self.tabular_features:
                # Find all columns that belong to this feature (e.g., channel_email, channel_phone)
                prefix = f"{feature_name}_"
                feature_indices = [
                    i for i, name in enumerate(all_feature_names)
                    if name.startswith(prefix) or name == feature_name
                ]

                if feature_indices:
                    # Sum SHAP values for all columns of this feature
                    total_contribution = sum(shap_values[i] for i in feature_indices)
                else:
                    total_contribution = 0.0

                value = None
                if raw_features and feature_name in raw_features:
                    value = raw_features[feature_name]

                contributions.append({
                    "feature": feature_name,
                    "value": value,
                    "contribution": float(total_contribution),
                })
        else:
            # Fallback: use first N values (less accurate)
            for i, feature_name in enumerate(self.tabular_features):
                if i >= len(shap_values):
                    break

                value = None
                if raw_features and feature_name in raw_features:
                    value = raw_features[feature_name]

                contributions.append({
                    "feature": feature_name,
                    "value": value,
                    "contribution": float(shap_values[i]),
                })

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return contributions

    def explain_batch(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        case_ids: List[str],
        raw_features_list: Optional[List[Dict]] = None,
        generate_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple cases.

        Args:
            X: Batch of input features
            case_ids: List of case identifiers
            raw_features_list: List of raw feature dicts
            generate_text: Whether to generate natural language

        Returns:
            List of explanation dicts
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        results = []
        for i in range(len(X)):
            raw_features = raw_features_list[i] if raw_features_list else None
            result = self.explain(
                X[i],
                case_id=case_ids[i],
                raw_features=raw_features,
                generate_text=generate_text,
            )
            results.append(result)

        return results

    def get_global_feature_importance(
        self,
        X_sample: Union[np.ndarray, pd.DataFrame],
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Compute global feature importance across a sample.

        Args:
            X_sample: Sample of input features
            n_samples: Number of samples to use

        Returns:
            Dict of feature_name -> mean |SHAP value|
        """
        if isinstance(X_sample, pd.DataFrame):
            X_sample = X_sample.values

        # Sample if too large
        if len(X_sample) > n_samples:
            indices = np.random.choice(len(X_sample), n_samples, replace=False)
            X_sample = X_sample[indices]

        # Compute SHAP values for all samples
        shap_values = self.shap_explainer.explain(X_sample)

        # Handle multi-class output
        if isinstance(shap_values, list):
            # Average across classes
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

        # Mean absolute SHAP per feature
        mean_importance = np.mean(shap_values, axis=0)

        # Map to feature names
        importance_dict = {}
        for i, name in enumerate(self.feature_names):
            if i < len(mean_importance):
                importance_dict[name] = float(mean_importance[i])

        # Sort by importance
        return dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True,
        ))


def create_explainer_from_pipeline(
    pipeline,
    vectorizer_step_name: str = "tfidf",
    classifier_step_name: str = "classifier",
    tabular_features: List[str] = None,
    background_data=None,
) -> ExplainabilityCore:
    """
    Factory function to create ExplainabilityCore from sklearn Pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline
        vectorizer_step_name: Name of TF-IDF step in pipeline
        classifier_step_name: Name of classifier step
        tabular_features: List of tabular feature names
        background_data: Background samples for SHAP

    Returns:
        Configured ExplainabilityCore instance
    """
    # Extract components from pipeline
    steps = dict(pipeline.named_steps)

    vectorizer = steps.get(vectorizer_step_name)
    classifier = steps.get(classifier_step_name, pipeline)

    # Build feature names
    feature_names = list(tabular_features or [])
    if vectorizer and hasattr(vectorizer, "get_feature_names_out"):
        feature_names.extend(vectorizer.get_feature_names_out().tolist())

    return ExplainabilityCore(
        model=classifier,
        vectorizer=vectorizer,
        feature_names=feature_names,
        tabular_features=tabular_features,
        background_data=background_data,
        use_llm=True,
    )
