"""
Feature grouper for semantic aggregation of TF-IDF features.

TF-IDF vectorization produces hundreds/thousands of sparse features.
This module aggregates SHAP values for individual tokens into meaningful
semantic groups (urgency, financial, account, technical keywords).
"""

import re
import numpy as np
from collections import defaultdict

from .config import SEMANTIC_GROUPS


class FeatureGrouper:
    """
    Aggregates TF-IDF token features into semantic groups.
    """

    def __init__(self, vectorizer=None, semantic_groups=None):
        """
        Initialize feature grouper.

        Args:
            vectorizer: Fitted TF-IDF vectorizer with vocabulary
            semantic_groups: Dict of group_name -> list of keywords (default: config groups)
        """
        self.vectorizer = vectorizer
        self.semantic_groups = semantic_groups or SEMANTIC_GROUPS
        self._token_to_groups = None
        self._feature_indices = None

        if vectorizer is not None:
            self._build_token_mapping()

    def _build_token_mapping(self):
        """Build mapping from tokens to semantic groups."""
        self._token_to_groups = defaultdict(list)
        self._feature_indices = {}

        vocab = self.vectorizer.vocabulary_
        feature_names = self.vectorizer.get_feature_names_out()

        # Map each token to its semantic groups
        for group_name, keywords in self.semantic_groups.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Find tokens containing this keyword
                for token, idx in vocab.items():
                    if keyword_lower in token.lower():
                        self._token_to_groups[idx].append(group_name)

        # Build feature indices for each group
        for group_name in self.semantic_groups:
            indices = []
            for idx, groups in self._token_to_groups.items():
                if group_name in groups:
                    indices.append(idx)
            self._feature_indices[group_name] = indices

    def set_vectorizer(self, vectorizer):
        """Set or update the TF-IDF vectorizer."""
        self.vectorizer = vectorizer
        self._build_token_mapping()

    def aggregate_shap_values(self, shap_values, feature_names, text_feature_start_idx):
        """
        Aggregate SHAP values for text features into semantic groups.

        Args:
            shap_values: Array of SHAP values for all features
            feature_names: List of all feature names
            text_feature_start_idx: Index where TF-IDF features begin

        Returns:
            Dict with:
                - semantic_groups: {group_name: aggregated_shap_value}
                - top_tokens: List of (token, shap_value) tuples
                - uncategorized: Sum of SHAP for tokens not in any group
        """
        text_shap_values = shap_values[text_feature_start_idx:]
        text_feature_names = feature_names[text_feature_start_idx:]

        # Aggregate by semantic group
        group_contributions = {}
        for group_name, indices in self._feature_indices.items():
            # Adjust indices relative to text features
            valid_indices = [i for i in indices if i < len(text_shap_values)]
            if valid_indices:
                group_contributions[group_name] = float(np.sum(text_shap_values[valid_indices]))
            else:
                group_contributions[group_name] = 0.0

        # Find top contributing tokens
        token_contributions = list(zip(text_feature_names, text_shap_values))
        token_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_tokens = [
            {"token": token, "contribution": float(value)}
            for token, value in token_contributions[:10]
            if abs(value) > 0.001
        ]

        # Calculate uncategorized contribution
        categorized_indices = set()
        for indices in self._feature_indices.values():
            categorized_indices.update(indices)

        uncategorized = 0.0
        for i, val in enumerate(text_shap_values):
            if i not in categorized_indices:
                uncategorized += val

        return {
            "semantic_groups": group_contributions,
            "top_tokens": top_tokens,
            "uncategorized": float(uncategorized),
        }

    def identify_dominant_theme(self, semantic_contributions):
        """
        Identify the dominant semantic theme from contributions.

        Args:
            semantic_contributions: Dict of group_name -> contribution value

        Returns:
            Tuple of (dominant_group_name, contribution_value) or (None, 0.0)
        """
        if not semantic_contributions:
            return None, 0.0

        # Filter to positive contributions
        positive = {k: v for k, v in semantic_contributions.items() if v > 0}

        if not positive:
            return None, 0.0

        dominant = max(positive.items(), key=lambda x: x[1])
        return dominant

    def extract_keywords_from_text(self, text):
        """
        Extract semantic keywords present in text.

        Args:
            text: Case summary text

        Returns:
            Dict of group_name -> list of matched keywords
        """
        text_lower = text.lower()
        matches = {}

        for group_name, keywords in self.semantic_groups.items():
            matched = []
            for keyword in keywords:
                # Word boundary matching
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    matched.append(keyword)
            if matched:
                matches[group_name] = matched

        return matches


class TabularFeatureProcessor:
    """
    Processes non-text (tabular) features for explanation.
    """

    def __init__(self, feature_columns, categorical_columns=None):
        """
        Initialize processor.

        Args:
            feature_columns: List of tabular feature column names
            categorical_columns: List of columns that are categorical
        """
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns or []

    def separate_features(self, shap_values, feature_names, all_feature_names):
        """
        Separate SHAP values into tabular and text features.

        Args:
            shap_values: Array of all SHAP values
            feature_names: Names of tabular features
            all_feature_names: Names of all features

        Returns:
            Dict with tabular and text start index
        """
        tabular_end = len(feature_names)

        return {
            "tabular_shap": shap_values[:tabular_end],
            "tabular_names": feature_names,
            "text_start_idx": tabular_end,
        }

    def rank_tabular_features(self, shap_values, feature_names, feature_values):
        """
        Rank tabular features by contribution.

        Args:
            shap_values: SHAP values for tabular features
            feature_names: Names of tabular features
            feature_values: Actual values of features

        Returns:
            List of dicts with feature info sorted by |contribution|
        """
        contributions = []
        for i, (name, shap_val) in enumerate(zip(feature_names, shap_values)):
            value = feature_values[i] if i < len(feature_values) else None
            contributions.append({
                "feature": name,
                "value": value,
                "contribution": float(shap_val),
            })

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return contributions
