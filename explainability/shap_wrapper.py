"""
SHAP wrapper for model-agnostic feature importance extraction.

Automatically selects the appropriate SHAP explainer based on model type:
- LinearExplainer for LogisticRegression, LinearSVC, etc.
- TreeExplainer for RandomForest, GradientBoosting, XGBoost, etc.
- KernelExplainer as fallback for any other sklearn-compatible model
"""

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from .config import SHAP_CONFIG


class SHAPExplainer:
    """
    Model-agnostic SHAP explainer that auto-selects the optimal algorithm.
    """

    LINEAR_MODELS = (LogisticRegression, SGDClassifier, LinearSVC)
    TREE_MODELS = (
        RandomForestClassifier,
        GradientBoostingClassifier,
        DecisionTreeClassifier,
        AdaBoostClassifier,
    )

    def __init__(self, model, background_data=None):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained sklearn classifier
            background_data: Training data sample for KernelExplainer (required for ensemble models)
        """
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self.explainer_type = None

        self._init_explainer()

    def _init_explainer(self):
        """Select and initialize the appropriate SHAP explainer."""
        model = self._unwrap_model(self.model)

        if isinstance(model, self.LINEAR_MODELS):
            self.explainer_type = "linear"
            self.explainer = shap.LinearExplainer(
                self.model,
                self.background_data,
            )

        elif isinstance(model, self.TREE_MODELS):
            self.explainer_type = "tree"
            self.explainer = shap.TreeExplainer(
                self.model,
                feature_perturbation="interventional",
            )

        else:
            # Fallback to KernelExplainer for ensembles and unknown models
            self.explainer_type = "kernel"
            if self.background_data is None:
                raise ValueError(
                    "background_data is required for KernelExplainer "
                    "with ensemble or unknown model types"
                )

            # Sample background data if too large
            bg_data = self._sample_background(self.background_data)
            self.explainer = shap.KernelExplainer(
                self._get_predict_fn(),
                bg_data,
            )

    def _unwrap_model(self, model):
        """Extract base model from sklearn wrappers/pipelines."""
        # Handle VotingClassifier - use first estimator for type detection
        if isinstance(model, (VotingClassifier, StackingClassifier)):
            return model  # Return as-is, will use KernelExplainer

        # Handle Pipeline
        if hasattr(model, "steps"):
            return model.steps[-1][1]

        return model

    def _get_predict_fn(self):
        """Get prediction function for KernelExplainer."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba
        return self.model.decision_function

    def _sample_background(self, data, max_samples=None):
        """Sample background data for KernelExplainer efficiency."""
        max_samples = max_samples or SHAP_CONFIG["max_samples"]

        if hasattr(data, "shape"):
            n_samples = data.shape[0]
        else:
            n_samples = len(data)

        if n_samples <= max_samples:
            return data

        indices = np.random.choice(n_samples, max_samples, replace=False)

        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return data[indices]

    def explain(self, X):
        """
        Compute SHAP values for input samples.

        Args:
            X: Input features (single sample or batch)

        Returns:
            shap.Explanation object with values and base_values
        """
        if hasattr(X, "values"):
            X = X.values

        # Convert sparse matrix to dense array
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        shap_values = self.explainer.shap_values(
            X,
            check_additivity=SHAP_CONFIG["check_additivity"],
        )

        return shap_values

    def explain_single(self, X, class_idx=None):
        """
        Get SHAP values for a single prediction.

        Args:
            X: Single input sample
            class_idx: Index of class to explain (default: predicted class)

        Returns:
            dict with shap_values (1D array) and base_value
        """
        # Convert sparse matrix to dense array
        if hasattr(X, "toarray"):
            X = X.toarray()

        if hasattr(X, "values"):
            X = X.values.reshape(1, -1)
        elif X.ndim == 1:
            X = X.reshape(1, -1)

        shap_values = self.explain(X)

        # Determine predicted class if not specified
        if class_idx is None:
            proba = self.model.predict_proba(X)[0]
            class_idx = np.argmax(proba)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # List of arrays, one per class: [array(n_samples, n_features), ...]
            values = shap_values[class_idx][0]
        elif shap_values.ndim == 3:
            # Shape: (n_samples, n_features, n_classes)
            values = shap_values[0, :, class_idx]
        elif shap_values.ndim == 2:
            # Could be (n_samples, n_features) or (n_features, n_classes) for single sample
            if shap_values.shape[0] == 1:
                # Single sample: (1, n_features)
                values = shap_values[0]
            elif shap_values.shape[1] == len(self.model.classes_):
                # Shape: (n_features, n_classes) - select class column
                values = shap_values[:, class_idx]
            else:
                values = shap_values[0]
        else:
            values = shap_values

        # Get base value
        if hasattr(self.explainer, "expected_value"):
            expected = self.explainer.expected_value
            if isinstance(expected, np.ndarray):
                base_value = expected[class_idx] if class_idx is not None else expected[0]
            else:
                base_value = expected
        else:
            base_value = 0.0

        return {
            "shap_values": values,
            "base_value": base_value,
            "class_idx": class_idx,
        }

    def get_feature_importance(self, X, feature_names, class_idx=None, top_k=10):
        """
        Get top contributing features for a prediction.

        Args:
            X: Input sample
            feature_names: List of feature names
            class_idx: Class index to explain
            top_k: Number of top features to return

        Returns:
            List of (feature_name, shap_value) tuples sorted by absolute importance
        """
        result = self.explain_single(X, class_idx)
        shap_values = result["shap_values"]

        # Pair features with values
        importance = list(zip(feature_names, shap_values))

        # Sort by absolute value
        importance.sort(key=lambda x: abs(x[1]), reverse=True)

        return importance[:top_k]
