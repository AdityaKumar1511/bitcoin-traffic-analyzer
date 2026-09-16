"""
SHAP-Backed Explainability Module for Investigative Lead Justification.

Extracts feature-level attributions and importance scores for flagged entities,
quantifying exactly which graph, behavioral, network, or temporal signals drove
the anomaly or composite risk determination.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelExplainer:
    """
    Computes local feature attributions using SHAP TreeExplainer / KernelExplainer
    with robust fallback to normalized feature z-score deviation.
    """

    def __init__(self, top_k: int = 5) -> None:
        """
        Initialize explainer.

        Args:
            top_k: Number of top contributing features to extract per alert.
        """
        self.top_k = top_k
        self.explainer: Optional[Any] = None
        self.feature_names: List[str] = []
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

    def fit(self, model: Any, background_data: pd.DataFrame) -> None:
        """
        Fit explainer with model and background distribution.
        """
        numeric_df = background_data.select_dtypes(include=[np.number]).fillna(0.0)
        self.feature_names = list(numeric_df.columns)
        if numeric_df.empty:
            return

        for col in self.feature_names:
            self.feature_means[col] = float(numeric_df[col].mean())
            std_val = float(numeric_df[col].std())
            self.feature_stds[col] = std_val if std_val > 1e-6 else 1.0

        try:
            if HAS_SHAP and isinstance(model, IsolationForest):
                self.explainer = shap.TreeExplainer(model, data=numeric_df.sample(min(100, len(numeric_df)), random_state=42))
        except Exception as err:
            logger.debug("SHAP TreeExplainer fallback: %s", err)
            self.explainer = None

    def explain_instance(
        self,
        instance_features: pd.Series,
    ) -> Dict[str, float]:
        """
        Explain a single entity instance.

        Returns:
            dict of {feature_name: contribution_score} sorted by absolute contribution descending.
        """
        contributions: Dict[str, float] = {}

        if self.explainer is not None:
            try:
                row_vals = instance_features[self.feature_names].values.reshape(1, -1)
                shap_vals = self.explainer.shap_values(row_vals)
                if isinstance(shap_vals, list):
                    vals = shap_vals[0][0]
                elif isinstance(shap_vals, np.ndarray):
                    vals = shap_vals[0]
                else:
                    vals = np.zeros(len(self.feature_names))

                for name, val in zip(self.feature_names, vals):
                    contributions[name] = float(val)
            except Exception as e:
                logger.debug("SHAP explanation error, fallback to z-score: %s", e)
                contributions = {}

        if not contributions:
            # Robust deviation fallback: (value - mean) / std for non-zero features
            for col, val in instance_features.items():
                if not isinstance(val, (int, float, np.number)) or np.isnan(val):
                    continue
                mean_v = self.feature_means.get(col, 0.0)
                std_v = self.feature_stds.get(col, 1.0)
                z = (float(val) - mean_v) / std_v
                if abs(z) > 0.1:
                    contributions[col] = round(z, 3)

        # Sort by absolute contribution descending and take top_k
        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_feats[: self.top_k])
