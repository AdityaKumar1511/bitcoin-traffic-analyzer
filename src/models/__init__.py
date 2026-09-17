"""
AI/ML Models package for Bitcoin transaction and wallet analysis.

Exports:
    - AnomalyDetector
    - IsolationForestDetector
    - ReconstructionAnomalyDetector
    - detect_wallet_anomalies
    - detect_transaction_anomalies
    - run_anomaly_detection
    - EntityClusterer
    - cluster_entities_multimodal
    - TaintPropagator
    - propagate_taint
    - EntityRiskClassifier
    - detect_coinjoin_transactions
    - detect_peel_chains
    - detect_mixer_behavior
    - run_all_evasion_detectors
"""

from src.models.anomaly import (
    AnomalyDetector,
    IsolationForestDetector,
    ReconstructionAnomalyDetector,
    detect_transaction_anomalies,
    detect_wallet_anomalies,
    run_anomaly_detection,
)
from src.models.classifier import EntityRiskClassifier
from src.models.clustering import EntityClusterer, cluster_entities_multimodal
from src.models.evasion_detectors import (
    detect_coinjoin_transactions,
    detect_mixer_behavior,
    detect_peel_chains,
    run_all_evasion_detectors,
)
from src.models.taint_propagation import TaintPropagator, propagate_taint

__all__ = [
    "AnomalyDetector",
    "IsolationForestDetector",
    "ReconstructionAnomalyDetector",
    "detect_wallet_anomalies",
    "detect_transaction_anomalies",
    "run_anomaly_detection",
    "EntityClusterer",
    "cluster_entities_multimodal",
    "TaintPropagator",
    "propagate_taint",
    "EntityRiskClassifier",
    "detect_coinjoin_transactions",
    "detect_peel_chains",
    "detect_mixer_behavior",
    "run_all_evasion_detectors",
]
