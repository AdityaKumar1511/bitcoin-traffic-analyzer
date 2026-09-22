"""
Anomaly Detection Module for Bitcoin Transactions and Wallets.

Implements dual complementary anomaly detection models:
1. Isolation Forest: Tree-based ensemble isolating anomalous entities via short path lengths.
2. Reconstruction Error Detector: Subspace/PCA projection measuring reconstruction error from normal latent patterns.
3. Ensemble Anomaly Scorer: Fuses isolation and reconstruction metrics into a normalized [0.0, 1.0] score.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.features import extract_all_features
from src.graph.builder import build_graph
from src.graph.heuristics import apply_change_address_heuristic, apply_common_input_heuristic
from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class IsolationForestDetector:
    """
    Wrapper for scikit-learn IsolationForest with min-max normalized anomaly scoring.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42, n_estimators: int = 100) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            n_jobs=-1,
        )
        self.scaler = RobustScaler()
        self.fitted_features: List[str] = []

    def fit_predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit Isolation Forest and return (normalized_scores, is_anomaly_flags).

        Args:
            X: Numeric feature DataFrame.

        Returns:
            Tuple of (scores [0..1 where 1 is anomalous], boolean anomaly flags).
        """
        if X.empty:
            return np.array([]), np.array([], dtype=bool)

        self.fitted_features = list(X.columns)
        X_clean = X.select_dtypes(include=[np.number]).fillna(0.0)
        X_scaled = self.scaler.fit_transform(X_clean)

        self.model.fit(X_scaled)

        # Decision function: lower values mean more anomalous
        raw_scores = self.model.decision_function(X_scaled)
        inverted = -raw_scores

        min_val, max_val = inverted.min(), inverted.max()
        if max_val > min_val:
            normalized_scores = (inverted - min_val) / (max_val - min_val)
        else:
            normalized_scores = np.zeros_like(inverted)

        preds = self.model.predict(X_scaled)
        is_anomaly = preds == -1

        return normalized_scores, is_anomaly


class ReconstructionAnomalyDetector:
    """
    Subspace projection / Autoencoder anomaly detector using PCA latent space reconstruction.
    Wallets/transactions whose feature combinations cannot be reconstructed well from
    the primary principal components have high reconstruction error.
    """

    def __init__(self, variance_retained: float = 0.90, random_state: int = 42) -> None:
        self.variance_retained = variance_retained
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.pca: Optional[PCA] = None

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute reconstruction error for each row in X.

        Args:
            X: Numeric feature DataFrame.

        Returns:
            Normalized reconstruction error array in [0.0, 1.0].
        """
        if X.empty:
            return np.array([])

        X_clean = X.select_dtypes(include=[np.number]).fillna(0.0)
        if X_clean.shape[1] == 0:
            return np.zeros(len(X))

        X_scaled = self.scaler.fit_transform(X_clean)

        n_samples, n_features = X_scaled.shape
        n_components = min(n_samples - 1, n_features - 1, max(2, int(n_features * 0.5)))
        n_components = max(1, n_components)

        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_latent = self.pca.fit_transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_latent)

        # Mean squared reconstruction error per entity
        recon_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)

        min_val, max_val = recon_errors.min(), recon_errors.max()
        if max_val > min_val:
            norm_errors = (recon_errors - min_val) / (max_val - min_val)
        else:
            norm_errors = np.zeros_like(recon_errors)

        return norm_errors


def _find_top_deviant_features(
    X: pd.DataFrame,
    scores: np.ndarray,
    top_k: int = 3,
) -> List[List[str]]:
    """
    Find top contributing features for anomalies based on deviation from median.
    """
    if X.empty or len(scores) == 0:
        return []

    X_clean = X.select_dtypes(include=[np.number]).fillna(0.0)
    if X_clean.shape[1] == 0:
        return [[] for _ in range(len(X))]

    medians = X_clean.median()
    iqr = X_clean.quantile(0.75) - X_clean.quantile(0.25)
    iqr[iqr == 0] = 1.0

    z_scores = np.abs((X_clean - medians) / iqr)

    top_features_per_row: List[List[str]] = []
    col_names = np.array(X_clean.columns)

    for i in range(len(X)):
        if scores[i] >= 0.5:
            row_z = z_scores.iloc[i].values
            top_idx = np.argsort(row_z)[::-1][:top_k]
            top_features_per_row.append(col_names[top_idx].tolist())
        else:
            top_features_per_row.append([])

    return top_features_per_row


class AnomalyDetector:
    """
    Unified unsupervised multi-feature anomaly detector combining Isolation Forest
    and PCA reconstruction error into an ensemble risk score.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.wallet_scaler = RobustScaler()
        self.tx_scaler = RobustScaler()
        self.wallet_model: Optional[IsolationForest] = None
        self.tx_model: Optional[IsolationForest] = None
        self.wallet_feature_names: List[str] = []
        self.tx_feature_names: List[str] = []

    def fit_predict_wallets(self, wallet_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit dual anomaly engine on wallet feature matrix and compute continuous anomaly scores.

        Args:
            wallet_features_df: DataFrame indexed by wallet_id containing numerical features.

        Returns:
            pd.DataFrame with columns:
                anomaly_score, is_anomaly, iforest_score, reconstruction_score, top_deviant_features
        """
        if wallet_features_df.empty:
            return pd.DataFrame(
                columns=["anomaly_score", "is_anomaly", "iforest_score", "reconstruction_score", "top_deviant_features"]
            ).set_index(pd.Index([], name="wallet_id"))

        numeric_df = wallet_features_df.select_dtypes(include=[np.number]).fillna(0.0)
        self.wallet_feature_names = list(numeric_df.columns)

        if numeric_df.shape[1] == 0:
            logger.warning("No numeric features found for wallet anomaly detection.")
            scores = pd.Series(0.0, index=wallet_features_df.index)
            res = pd.DataFrame(
                {
                    "anomaly_score": scores,
                    "is_anomaly": False,
                    "iforest_score": scores,
                    "reconstruction_score": scores,
                    "top_deviant_features": [[] for _ in range(len(wallet_features_df))],
                },
                index=wallet_features_df.index,
            )
            res.index.name = "wallet_id"
            return res

        X_scaled = self.wallet_scaler.fit_transform(numeric_df)
        self.wallet_model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.wallet_model.fit(X_scaled)

        raw_scores = self.wallet_model.decision_function(X_scaled)
        preds = self.wallet_model.predict(X_scaled)

        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            norm_scores = (max_s - raw_scores) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(raw_scores)

        # Dual model: PCA reconstruction error
        n_comp = max(1, min(X_scaled.shape[0] - 1, X_scaled.shape[1] - 1, max(2, int(X_scaled.shape[1] * 0.5))))
        pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_latent = pca.fit_transform(X_scaled)
        X_rec = pca.inverse_transform(X_latent)
        rec_err = np.mean(np.square(X_scaled - X_rec), axis=1)
        min_r, max_r = rec_err.min(), rec_err.max()
        norm_rec = (rec_err - min_r) / (max_r - min_r) if max_r > min_r else np.zeros_like(rec_err)

        ensemble_scores = 0.65 * norm_scores + 0.35 * norm_rec
        deviant_features = _find_top_deviant_features(numeric_df, ensemble_scores, top_k=3)

        result_df = pd.DataFrame(
            {
                "anomaly_score": np.round(ensemble_scores, 4),
                "is_anomaly": preds == -1,
                "iforest_score": np.round(norm_scores, 4),
                "reconstruction_score": np.round(norm_rec, 4),
                "top_deviant_features": deviant_features,
            },
            index=wallet_features_df.index,
        )
        result_df.index.name = "wallet_id"
        logger.info(
            "Wallet anomaly detection completed: %d entities evaluated, %d flagged as anomalous.",
            len(result_df),
            int((preds == -1).sum()),
        )
        return result_df

    def fit_predict_transactions(self, tx_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit dual anomaly engine on transaction feature matrix and compute anomaly scores.

        Args:
            tx_features_df: DataFrame indexed by txid containing numerical features.

        Returns:
            pd.DataFrame with columns:
                anomaly_score, is_anomaly, iforest_score, reconstruction_score, top_deviant_features
        """
        if tx_features_df.empty:
            return pd.DataFrame(
                columns=["anomaly_score", "is_anomaly", "iforest_score", "reconstruction_score", "top_deviant_features"]
            ).set_index(pd.Index([], name="txid"))

        numeric_df = tx_features_df.select_dtypes(include=[np.number]).fillna(0.0)
        self.tx_feature_names = list(numeric_df.columns)

        if numeric_df.shape[1] == 0:
            scores = pd.Series(0.0, index=tx_features_df.index)
            res = pd.DataFrame(
                {
                    "anomaly_score": scores,
                    "is_anomaly": False,
                    "iforest_score": scores,
                    "reconstruction_score": scores,
                    "top_deviant_features": [[] for _ in range(len(tx_features_df))],
                },
                index=tx_features_df.index,
            )
            res.index.name = "txid"
            return res

        X_scaled = self.tx_scaler.fit_transform(numeric_df)
        self.tx_model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.tx_model.fit(X_scaled)

        raw_scores = self.tx_model.decision_function(X_scaled)
        preds = self.tx_model.predict(X_scaled)

        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            norm_scores = (max_s - raw_scores) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(raw_scores)

        # Dual model: PCA reconstruction error
        n_comp = max(1, min(X_scaled.shape[0] - 1, X_scaled.shape[1] - 1, max(2, int(X_scaled.shape[1] * 0.5))))
        pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_latent = pca.fit_transform(X_scaled)
        X_rec = pca.inverse_transform(X_latent)
        rec_err = np.mean(np.square(X_scaled - X_rec), axis=1)
        min_r, max_r = rec_err.min(), rec_err.max()
        norm_rec = (rec_err - min_r) / (max_r - min_r) if max_r > min_r else np.zeros_like(rec_err)

        ensemble_scores = 0.65 * norm_scores + 0.35 * norm_rec
        deviant_features = _find_top_deviant_features(numeric_df, ensemble_scores, top_k=3)

        result_df = pd.DataFrame(
            {
                "anomaly_score": np.round(ensemble_scores, 4),
                "is_anomaly": preds == -1,
                "iforest_score": np.round(norm_scores, 4),
                "reconstruction_score": np.round(norm_rec, 4),
                "top_deviant_features": deviant_features,
            },
            index=tx_features_df.index,
        )
        result_df.index.name = "txid"
        logger.info(
            "Transaction anomaly detection completed: %d evaluated, %d flagged.",
            len(result_df),
            int((preds == -1).sum()),
        )
        return result_df


def detect_wallet_anomalies(
    wallet_features_df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Convenience function to run dual ensemble anomaly detection on wallet features."""
    detector = AnomalyDetector(contamination=contamination, random_state=random_state)
    return detector.fit_predict_wallets(wallet_features_df)


def detect_transaction_anomalies(
    tx_features_df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Convenience function to run dual ensemble anomaly detection on transaction features."""
    detector = AnomalyDetector(contamination=contamination, random_state=random_state)
    return detector.fit_predict_transactions(tx_features_df)


def run_anomaly_detection(
    input_file: Union[str, Path] = "data/raw/synthetic_transactions.csv",
    contamination: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end runner for anomaly detection on raw transaction file.
    """
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    logger.info("Parsing transactions from %s", input_path)
    df = parse_file(input_path)

    logger.info("Building graph and applying heuristics...")
    graph = build_graph(df)
    apply_common_input_heuristic(graph, df)
    apply_change_address_heuristic(graph, df)

    logger.info("Extracting feature matrices...")
    feature_matrices = extract_all_features(graph, df)

    detector = AnomalyDetector(contamination=contamination)
    wallet_anomalies = detector.fit_predict_wallets(feature_matrices["wallets"])
    tx_anomalies = detector.fit_predict_transactions(feature_matrices["transactions"])

    return {
        "wallet_anomalies": wallet_anomalies,
        "transaction_anomalies": tx_anomalies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unsupervised anomaly detection.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/raw/synthetic_transactions.csv",
        help="Path to transaction file (.csv, .json, .xml)",
    )
    parser.add_argument(
        "--contamination",
        "-c",
        type=float,
        default="0.05",
        help="Anomaly contamination rate (default: 0.05)",
    )
    args = parser.parse_args()

    results = run_anomaly_detection(args.input, contamination=float(args.contamination))
    wallets_df = results["wallet_anomalies"]
    tx_df = results["transaction_anomalies"]

    print("\n" + "=" * 60)
    print("           ANOMALY DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total Wallets: {len(wallets_df):,} | Anomalous: {int(wallets_df['is_anomaly'].sum()):,}")
    print(f"Total Transactions: {len(tx_df):,} | Anomalous: {int(tx_df['is_anomaly'].sum()):,}")

    top_wallets = wallets_df.sort_values(by="anomaly_score", ascending=False).head(5)
    print("\n--- TOP 5 ANOMALOUS WALLETS ---")
    for w_id, row in top_wallets.iterrows():
        print(f"  Wallet: {w_id} | Score: {row['anomaly_score']:.4f} | Flagged: {row['is_anomaly']}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
