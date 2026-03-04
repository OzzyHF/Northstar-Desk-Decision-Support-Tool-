"""
Structured output builder for explainability JSON.

Generates a standardized JSON format containing all grounded facts
about a prediction, designed for consumption by an LLM to generate
natural language explanations.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .config import (
    PRIORITY_LEVELS,
    FEATURE_EXPLANATIONS,
    TENURE_THRESHOLDS,
)


class StructuredOutputBuilder:
    """
    Builds structured JSON output for model explanations.
    """

    def __init__(self):
        """Initialize the output builder."""
        self.feature_explanations = FEATURE_EXPLANATIONS
        self.priority_levels = PRIORITY_LEVELS

    def build(
        self,
        case_id: str,
        prediction: str,
        probabilities: Dict[str, float],
        feature_contributions: List[Dict],
        text_analysis: Dict,
        raw_features: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Build complete structured output.

        Args:
            case_id: Unique case identifier
            prediction: Predicted priority class
            probabilities: Dict of class -> probability
            feature_contributions: List of feature contribution dicts
            text_analysis: Dict with semantic groups and top tokens
            raw_features: Optional dict of original feature values

        Returns:
            Complete structured output dictionary
        """
        # Split contributions into positive and negative
        positive_contributions = []
        negative_contributions = []

        for contrib in feature_contributions:
            enriched = self._enrich_contribution(contrib, raw_features)
            if contrib["contribution"] > 0:
                positive_contributions.append(enriched)
            elif contrib["contribution"] < 0:
                negative_contributions.append(enriched)

        # Get confidence from predicted class probability
        confidence = probabilities.get(prediction, 0.0)

        return {
            "case_id": case_id,
            "prediction": {
                "priority": prediction,
                "confidence": round(confidence, 3),
                "probabilities": {
                    k: round(v, 3) for k, v in probabilities.items()
                },
            },
            "feature_contributions": {
                "positive": positive_contributions,
                "negative": negative_contributions,
            },
            "text_analysis": {
                "top_tokens": text_analysis.get("top_tokens", []),
                "semantic_groups": {
                    k: round(v, 3)
                    for k, v in text_analysis.get("semantic_groups", {}).items()
                },
            },
            "metadata": {
                "raw_features": raw_features or {},
            },
        }

    def _enrich_contribution(
        self,
        contribution: Dict,
        raw_features: Optional[Dict],
    ) -> Dict[str, Any]:
        """
        Enrich a feature contribution with human-readable explanation.

        Args:
            contribution: Dict with feature, value, contribution
            raw_features: Optional raw feature values

        Returns:
            Enriched contribution dict
        """
        feature = contribution["feature"]
        value = contribution.get("value")

        # Get value from raw features if not provided
        if value is None and raw_features:
            value = raw_features.get(feature)

        # Get explanation based on feature type
        explanation = self._get_feature_explanation(feature, value)

        return {
            "feature": feature,
            "value": self._format_value(value),
            "contribution": round(contribution["contribution"], 3),
            "explanation": explanation,
        }

    def _get_feature_explanation(
        self,
        feature: str,
        value: Any,
    ) -> str:
        """Get human-readable explanation for a feature value."""
        # Check if we have explanations for this feature
        if feature in self.feature_explanations:
            feature_explns = self.feature_explanations[feature]

            # Handle customer tenure specially
            if feature == "customer_tenure_months":
                tier = self._categorize_tenure(value)
                return feature_explns.get(tier, f"Customer tenure: {value} months")

            # Look up value-specific explanation
            if value is not None:
                value_str = str(value).lower()
                if value_str in feature_explns:
                    return feature_explns[value_str]

            # Return generic explanation
            return f"{feature} = {value}"

        return f"{feature} value: {value}"

    def _categorize_tenure(self, months: Any) -> str:
        """Categorize customer tenure into tiers."""
        try:
            months = float(months)
        except (TypeError, ValueError):
            return "medium"

        if months < TENURE_THRESHOLDS["low"]:
            return "low"
        elif months > TENURE_THRESHOLDS["high"]:
            return "high"
        return "medium"

    def _format_value(self, value: Any) -> Any:
        """Format value for JSON output."""
        if value is None:
            return None
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def to_prompt_context(self, structured_output: Dict) -> str:
        """
        Convert structured output to context string for LLM prompt.

        Args:
            structured_output: Complete structured output dict

        Returns:
            Formatted string for prompt injection
        """
        pred = structured_output["prediction"]
        contribs = structured_output["feature_contributions"]
        text = structured_output["text_analysis"]

        lines = [
            f"Case ID: {structured_output['case_id']}",
            f"Predicted Priority: {pred['priority']}",
            f"Confidence: {pred['confidence'] * 100:.1f}%",
            "",
            "Class Probabilities:",
        ]

        for cls, prob in pred["probabilities"].items():
            lines.append(f"  - {cls}: {prob * 100:.1f}%")

        lines.extend(["", "Positive Contributing Factors:"])
        for c in contribs["positive"][:5]:
            lines.append(
                f"  - {c['feature']}={c['value']}: +{c['contribution']:.3f}"
            )
            lines.append(f"    Reason: {c['explanation']}")

        if contribs["negative"]:
            lines.extend(["", "Negative Contributing Factors:"])
            for c in contribs["negative"][:3]:
                lines.append(
                    f"  - {c['feature']}={c['value']}: {c['contribution']:.3f}"
                )
                lines.append(f"    Reason: {c['explanation']}")

        if text["top_tokens"]:
            lines.extend(["", "Key Text Tokens:"])
            for t in text["top_tokens"][:5]:
                sign = "+" if t["contribution"] > 0 else ""
                lines.append(f"  - \"{t['token']}\": {sign}{t['contribution']:.3f}")

        if text["semantic_groups"]:
            lines.extend(["", "Semantic Theme Contributions:"])
            for group, val in text["semantic_groups"].items():
                if abs(val) > 0.01:
                    sign = "+" if val > 0 else ""
                    lines.append(f"  - {group}: {sign}{val:.3f}")

        return "\n".join(lines)


def format_explanation_card(structured_output: Dict) -> Dict[str, Any]:
    """
    Format structured output for display in UI cards.

    Args:
        structured_output: Complete structured output dict

    Returns:
        Dict formatted for Streamlit display
    """
    pred = structured_output["prediction"]
    contribs = structured_output["feature_contributions"]

    return {
        "header": {
            "case_id": structured_output["case_id"],
            "priority": pred["priority"],
            "confidence_pct": f"{pred['confidence'] * 100:.0f}%",
        },
        "probability_chart": pred["probabilities"],
        "top_positive_factors": [
            {
                "name": c["feature"],
                "value": c["value"],
                "impact": c["contribution"],
                "explanation": c["explanation"],
            }
            for c in contribs["positive"][:3]
        ],
        "top_negative_factors": [
            {
                "name": c["feature"],
                "value": c["value"],
                "impact": c["contribution"],
                "explanation": c["explanation"],
            }
            for c in contribs["negative"][:2]
        ],
        "text_insights": structured_output["text_analysis"],
    }
