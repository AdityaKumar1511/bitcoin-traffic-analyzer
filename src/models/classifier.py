"""
Supervised Risk Classifier Module for Bitcoin Entity Profiling.

Trains a supervised classifier (Random Forest / Gradient Boosting) on labeled or
ground-truth fraudulent patterns to predict risk probabilities for unseen entities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EntityRiskClassifier:
    """
    Supervised classifier for predicting suspicious entity probabilities.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Fit Random Forest classifier on labeled training features.
        """
        numeric_X = X.select_dtypes(include=[np.number]).fillna(0.0)
        self.feature_names = list(numeric_X.columns)

        if numeric_X.empty or len(y.unique()) < 2:
            logger.warning("Insufficient classes or features to train supervised classifier.")
            return {}

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.model.fit(numeric_X, y)

        preds = self.model.predict(numeric_X)
        metrics = {
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
        }
        logger.info("Supervised model trained: F1=%.4f, Precision=%.4f, Recall=%.4f", metrics["f1"], metrics["precision"], metrics["recall"])
        return metrics

    def predict_risk_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict risk probabilities [0.0, 1.0] for entities.
        """
        if self.model is None or X.empty:
            return pd.Series(0.0, index=X.index)

        numeric_X = X.select_dtypes(include=[np.number]).fillna(0.0)
        # Align features
        for col in self.feature_names:
            if col not in numeric_X.columns:
                numeric_X[col] = 0.0
        numeric_X = numeric_X[self.feature_names]

        probas = self.model.predict_proba(numeric_X)[:, 1]
        return pd.Series(np.round(probas, 4), index=X.index)

    def predict_risk_probabilities(self, X: pd.DataFrame) -> pd.Series:
        """Alias for predict_risk_proba."""
        return self.predict_risk_proba(X)
