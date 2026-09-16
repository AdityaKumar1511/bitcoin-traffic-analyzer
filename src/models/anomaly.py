"""
Unsupervised Anomaly Detection Module for Bitcoin Transaction Network.

Uses Isolation Forest and robust statistical outlier methods to compute
continuous anomaly scores [0.0, 1.0] across engineered wallet, transaction,
and network features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.features import extract_all_features
from src.graph.builder import build_graph
from src.graph.heuristics import apply_change_address_heuristic, apply_common_input_heuristic
from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger
from src.utils.paths import get_project_root, load_config

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Unsupervised multi-feature anomaly detector for Bitcoin entities and transactions.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
    ) -> None:
        """
        Initialize anomaly detection model.

        Args:
            contamination: Expected proportion of outliers in the dataset.
            n_estimators: Number of isolation trees in the ensemble.
            random_state: Seed for reproducibility.
        """
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
        Fit Isolation Forest on wallet feature matrix and compute continuous anomaly scores.

        Args:
            wallet_features_df: DataFrame indexed by wallet_id containing numerical features.

        Returns:
            pd.DataFrame with columns:
                wallet_id, anomaly_score (0.0 - 1.0), is_anomaly (bool)
        """
        if wallet_features_df.empty:
            return pd.DataFrame(columns=["wallet_id", "anomaly_score", "is_anomaly"]).set_index("wallet_id")

        numeric_df = wallet_features_df.select_dtypes(include=[np.number]).fillna(0.0)
        self.wallet_feature_names = list(numeric_df.columns)

        if numeric_df.shape[1] == 0:
            logger.warning("No numeric features found for wallet anomaly detection.")
            scores = pd.Series(0.0, index=wallet_features_df.index)
            return pd.DataFrame({
                "anomaly_score": scores,
                "is_anomaly": False,
            }, index=wallet_features_df.index)

        X_scaled = self.wallet_scaler.fit_transform(numeric_df)
        self.wallet_model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.wallet_model.fit(X_scaled)

        # Decision function: lower values mean more anomalous
        raw_scores = self.wallet_model.decision_function(X_scaled)
        preds = self.wallet_model.predict(X_scaled)

        # Invert and scale to [0.0, 1.0] where 1.0 = highly anomalous
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            norm_scores = (max_s - raw_scores) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(raw_scores)

        result_df = pd.DataFrame(
            {
                "anomaly_score": np.round(norm_scores, 4),
                "is_anomaly": preds == -1,
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
        Fit Isolation Forest on transaction feature matrix and compute anomaly scores.

        Args:
            tx_features_df: DataFrame indexed by txid containing numerical features.

        Returns:
            pd.DataFrame with columns:
                txid, anomaly_score (0.0 - 1.0), is_anomaly (bool)
        """
        if tx_features_df.empty:
            return pd.DataFrame(columns=["txid", "anomaly_score", "is_anomaly"]).set_index("txid")

        numeric_df = tx_features_df.select_dtypes(include=[np.number]).fillna(0.0)
        self.tx_feature_names = list(numeric_df.columns)

        if numeric_df.shape[1] == 0:
            scores = pd.Series(0.0, index=tx_features_df.index)
            return pd.DataFrame({
                "anomaly_score": scores,
                "is_anomaly": False,
            }, index=tx_features_df.index)

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

        result_df = pd.DataFrame(
            {
                "anomaly_score": np.round(norm_scores, 4),
                "is_anomaly": preds == -1,
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
        default=0.05,
        help="Anomaly contamination rate (default: 0.05)",
    )
    args = parser.parse_args()

    results = run_anomaly_detection(args.input, contamination=args.contamination)
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
