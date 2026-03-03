"""
Explainability module for Northstar Desk Priority Classifier.

Provides model-agnostic explanations using SHAP and natural language generation.
"""

from .core import ExplainabilityCore
from .shap_wrapper import SHAPExplainer
from .feature_grouper import FeatureGrouper
from .structured_output import StructuredOutputBuilder
from .llm_generator import LLMGenerator
from .config import SEMANTIC_GROUPS, FEATURE_EXPLANATIONS

__all__ = [
    "ExplainabilityCore",
    "SHAPExplainer",
    "FeatureGrouper",
    "StructuredOutputBuilder",
    "LLMGenerator",
    "SEMANTIC_GROUPS",
    "FEATURE_EXPLANATIONS",
]
