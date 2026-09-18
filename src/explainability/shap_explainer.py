"""
SHAP-Backed Explainability Module for Investigative Lead Justification.

Extracts feature-level attributions and importance scores for flagged entities,
quantifying exactly which graph, behavioral, network, or temporal signals drove
the anomaly or composite risk determination.

Supports:
    - TreeExplainer for IsolationForest, RandomForest, XGBoost (fast, exact SHAP values).
    - KernelExplainer fallback for opaque models (Autoencoder, PCA reconstruction).
    - Batch explanation generation for all flagged entities.
    - Global feature importance summaries for the dashboard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier

try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    HAS_XGBOOST = False

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Tree-based model types that support TreeExplainer
_TREE_MODEL_TYPES: Tuple[type, ...] = (IsolationForest, RandomForestClassifier, GradientBoostingClassifier)
if HAS_XGBOOST and XGBClassifier is not None:
    _TREE_MODEL_TYPES = _TREE_MODEL_TYPES + (XGBClassifier,)


class ModelExplainer:
    """
    Computes local feature attributions using SHAP TreeExplainer / KernelExplainer
    with robust fallback to normalized feature z-score deviation.

    Supports:
        - IsolationForest, RandomForest, XGBoost → SHAP TreeExplainer
        - Any callable predict function → SHAP KernelExplainer
        - Fallback to z-score deviation when SHAP is unavailable
    """

    def __init__(self, top_k: int = 5) -> None:
        """
        Initialize explainer.

        Args:
            top_k: Number of top contributing features to extract per alert.
        """
        self.top_k = top_k
        self.explainer: Optional[Any] = None
        self.explainer_type: str = "none"  # "tree", "kernel", "zscore"
        self.feature_names: List[str] = []
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}
        self._background_data: Optional[pd.DataFrame] = None
        self._global_shap_values: Optional[np.ndarray] = None

    def fit(
        self,
        model: Any,
        background_data: pd.DataFrame,
        predict_fn: Optional[Any] = None,
    ) -> None:
        """
        Fit explainer with model and background distribution.

        Args:
            model: The trained ML model (IsolationForest, RandomForest, XGBoost, or any object).
            background_data: Feature DataFrame used as the background distribution for SHAP.
            predict_fn: Optional callable for KernelExplainer fallback. Should accept
                a 2D numpy array and return a 1D array of scores.
        """
        numeric_df = background_data.select_dtypes(include=[np.number]).fillna(0.0)
        self.feature_names = list(numeric_df.columns)
        if numeric_df.empty:
            logger.warning("Empty background data — explainer will use z-score fallback.")
            return

        # Cache feature statistics for z-score fallback
        for col in self.feature_names:
            self.feature_means[col] = float(numeric_df[col].mean())
            std_val = float(numeric_df[col].std())
            self.feature_stds[col] = std_val if std_val > 1e-6 else 1.0

        # Sample background data for SHAP (limit for performance)
        n_background = min(100, len(numeric_df))
        bg_sample = numeric_df.sample(n=n_background, random_state=42)
        self._background_data = bg_sample

        if not HAS_SHAP:
            logger.info("SHAP not installed — falling back to z-score deviation explainer.")
            self.explainer_type = "zscore"
            return

        # Strategy 1: TreeExplainer for tree-based models
        if isinstance(model, _TREE_MODEL_TYPES):
            try:
                self.explainer = shap.TreeExplainer(model, data=bg_sample)
                self.explainer_type = "tree"
                logger.info("SHAP TreeExplainer initialized for %s.", type(model).__name__)
                return
            except Exception as err:
                logger.debug("TreeExplainer failed for %s: %s. Trying KernelExplainer.", type(model).__name__, err)

        # Strategy 2: KernelExplainer for any model with a predict function
        kernel_fn = predict_fn
        if kernel_fn is None:
            # Try common predict interfaces
            if hasattr(model, "decision_function"):
                kernel_fn = model.decision_function
            elif hasattr(model, "predict_proba"):
                kernel_fn = lambda x: model.predict_proba(x)[:, 1]
            elif hasattr(model, "predict"):
                kernel_fn = model.predict

        if kernel_fn is not None:
            try:
                self.explainer = shap.KernelExplainer(kernel_fn, bg_sample)
                self.explainer_type = "kernel"
                logger.info("SHAP KernelExplainer initialized as fallback.")
                return
            except Exception as err:
                logger.debug("KernelExplainer failed: %s. Using z-score fallback.", err)

        # Strategy 3: z-score fallback
        self.explainer_type = "zscore"
        logger.info("All SHAP explainers failed — using z-score deviation fallback.")

    def explain_instance(
        self,
        instance_features: pd.Series,
    ) -> Dict[str, float]:
        """
        Explain a single entity instance.

        Returns:
            dict of {feature_name: contribution_score} sorted by absolute contribution descending,
            limited to top_k entries.
        """
        contributions: Dict[str, float] = {}

        if self.explainer is not None and self.explainer_type in ("tree", "kernel"):
            try:
                # Align features to the expected columns
                aligned = instance_features.reindex(self.feature_names, fill_value=0.0)
                row_vals = aligned.values.astype(float).reshape(1, -1)

                if self.explainer_type == "kernel":
                    shap_vals = self.explainer.shap_values(row_vals, nsamples=50)
                else:
                    shap_vals = self.explainer.shap_values(row_vals)

                # Handle different SHAP output shapes
                if isinstance(shap_vals, list):
                    vals = shap_vals[0][0] if len(shap_vals[0].shape) > 1 else shap_vals[0]
                elif isinstance(shap_vals, np.ndarray):
                    vals = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
                else:
                    vals = np.zeros(len(self.feature_names))

                for name, val in zip(self.feature_names, vals):
                    contributions[name] = round(float(val), 4)
            except Exception as e:
                logger.debug("SHAP explanation error, fallback to z-score: %s", e)
                contributions = {}

        # Z-score fallback
        if not contributions:
            contributions = self._zscore_explain(instance_features)

        # Sort by absolute contribution descending and take top_k
        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_feats[: self.top_k])

    def explain_batch(
        self,
        features_df: pd.DataFrame,
        score_threshold: float = 0.3,
        anomaly_scores: Optional[pd.Series] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Explain multiple flagged entities in batch.

        Only explains entities whose anomaly score exceeds score_threshold,
        or all entities if anomaly_scores is not provided.

        Args:
            features_df: DataFrame indexed by entity_id with feature columns.
            score_threshold: Minimum anomaly score to warrant explanation.
            anomaly_scores: Optional Series indexed by entity_id with anomaly scores.

        Returns:
            dict: {entity_id: {feature_name: contribution_score, ...}, ...}
        """
        explanations: Dict[str, Dict[str, float]] = {}

        if features_df.empty:
            return explanations

        for entity_id in features_df.index:
            # Skip low-risk entities if scores are provided
            if anomaly_scores is not None:
                score = anomaly_scores.get(entity_id, 0.0)
                if score < score_threshold:
                    continue

            try:
                row = features_df.loc[entity_id]
                explanations[entity_id] = self.explain_instance(row)
            except Exception as e:
                logger.debug("Could not explain entity %s: %s", entity_id, e)
                explanations[entity_id] = {}

        logger.info("Generated SHAP explanations for %d entities.", len(explanations))
        return explanations

    def compute_global_feature_importance(
        self,
        features_df: pd.DataFrame,
        max_samples: int = 200,
    ) -> pd.DataFrame:
        """
        Compute global feature importance across the entire dataset using mean absolute
        SHAP values (or z-score deviations as fallback).

        Args:
            features_df: Full feature matrix.
            max_samples: Maximum number of samples to use for SHAP computation.

        Returns:
            pd.DataFrame with columns ['feature', 'mean_abs_importance', 'rank'],
            sorted by importance descending.
        """
        numeric_df = features_df.select_dtypes(include=[np.number]).fillna(0.0)
        if numeric_df.empty:
            return pd.DataFrame(columns=["feature", "mean_abs_importance", "rank"])

        sample_n = min(max_samples, len(numeric_df))
        sample_df = numeric_df.sample(n=sample_n, random_state=42)

        if self.explainer is not None and self.explainer_type in ("tree", "kernel"):
            try:
                aligned = sample_df.reindex(columns=self.feature_names, fill_value=0.0)
                X = aligned.values.astype(float)

                if self.explainer_type == "kernel":
                    shap_vals = self.explainer.shap_values(X, nsamples=50)
                else:
                    shap_vals = self.explainer.shap_values(X)

                if isinstance(shap_vals, list):
                    vals = shap_vals[0]
                else:
                    vals = shap_vals

                self._global_shap_values = vals
                mean_abs = np.mean(np.abs(vals), axis=0)

                importance_df = pd.DataFrame({
                    "feature": self.feature_names,
                    "mean_abs_importance": np.round(mean_abs, 4),
                })
                importance_df = importance_df.sort_values("mean_abs_importance", ascending=False).reset_index(drop=True)
                importance_df["rank"] = range(1, len(importance_df) + 1)
                return importance_df

            except Exception as e:
                logger.debug("Global SHAP importance failed: %s. Using variance-based fallback.", e)

        # Variance-based fallback
        feature_vars = numeric_df.var()
        total_var = feature_vars.sum()
        if total_var > 0:
            importance_vals = (feature_vars / total_var).round(4)
        else:
            importance_vals = pd.Series(0.0, index=numeric_df.columns)

        importance_df = pd.DataFrame({
            "feature": importance_vals.index,
            "mean_abs_importance": importance_vals.values,
        })
        importance_df = importance_df.sort_values("mean_abs_importance", ascending=False).reset_index(drop=True)
        importance_df["rank"] = range(1, len(importance_df) + 1)
        return importance_df

    def get_shap_summary_data(self) -> Optional[np.ndarray]:
        """
        Return the cached global SHAP values matrix for dashboard visualization
        (e.g., SHAP summary plots, beeswarm plots).

        Returns:
            numpy array of shape (n_samples, n_features) or None if not computed.
        """
        return self._global_shap_values

    def _zscore_explain(self, instance_features: pd.Series) -> Dict[str, float]:
        """
        Robust z-score deviation fallback when SHAP is unavailable.

        For each numeric feature, computes (value - mean) / std and returns features
        with |z| > 0.1 as non-trivial contributors.
        """
        contributions: Dict[str, float] = {}
        for col, val in instance_features.items():
            if not isinstance(val, (int, float, np.number)) or np.isnan(val):
                continue
            mean_v = self.feature_means.get(col, 0.0)
            std_v = self.feature_stds.get(col, 1.0)
            z = (float(val) - mean_v) / std_v
            if abs(z) > 0.1:
                contributions[col] = round(z, 3)
        return contributions
