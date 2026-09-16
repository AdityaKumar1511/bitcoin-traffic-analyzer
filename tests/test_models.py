"""
Unit tests for ML models, taint propagation, and clustering modules (src/models/).
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import networkx as nx

from src.models.anomaly import AnomalyDetector
from src.models.clustering import EntityClusterer
from src.models.taint_propagation import TaintPropagator
from src.models.classifier import EntityRiskClassifier
from src.correlation.network_blockchain import NetworkBlockchainCorrelator


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_features = pd.DataFrame(
            {
                "tx_count": [1, 5, 20, 2, 50],
                "fan_in_ratio": [0.1, 0.2, 0.95, 0.3, 0.99],
                "fan_out_ratio": [0.1, 0.3, 0.1, 0.2, 0.95],
                "total_btc_in": [0.5, 1.2, 50.0, 0.8, 120.0],
                "is_high_risk_asn": [0, 0, 1, 0, 1],
            },
            index=["w1", "w2", "w3_bad", "w4", "w5_bad"],
        )

    def test_anomaly_detector(self) -> None:
        detector = AnomalyDetector(contamination=0.4, random_state=42)
        results = detector.fit_predict_wallets(self.sample_features)
        
        self.assertEqual(len(results), 5)
        self.assertIn("anomaly_score", results.columns)
        self.assertIn("is_anomaly", results.columns)

        # Anomaly scores should be bounded in [0, 1]
        self.assertTrue((results["anomaly_score"] >= 0.0).all())
        self.assertTrue((results["anomaly_score"] <= 1.0).all())

        # w5_bad should have higher anomaly score than normal w1
        self.assertGreater(results.loc["w5_bad", "anomaly_score"], results.loc["w1", "anomaly_score"])

    def test_taint_propagation(self) -> None:
        graph = nx.MultiDiGraph()
        # Seed -> tx1 -> wA -> tx2 -> wB
        graph.add_node("seed_bad", node_type="wallet")
        graph.add_node("tx1", node_type="transaction")
        graph.add_node("wA", node_type="wallet")
        graph.add_node("tx2", node_type="transaction")
        graph.add_node("wB", node_type="wallet")

        graph.add_edge("seed_bad", "tx1", edge_type="input", amount=10.0)
        graph.add_edge("tx1", "wA", edge_type="output", amount=9.99)
        graph.add_edge("wA", "tx2", edge_type="input", amount=9.99)
        graph.add_edge("tx2", "wB", edge_type="output", amount=9.98)

        propagator = TaintPropagator(decay_factor=0.5, max_hops=3)
        taint_df = propagator.propagate(graph, {"seed_bad": 1.0})

        self.assertIn("seed_bad", taint_df.index)
        self.assertIn("wA", taint_df.index)
        self.assertIn("wB", taint_df.index)

        # Seed has 1.0 taint, wA has 0.5 (1 hop), wB has 0.25 (2 hops)
        self.assertEqual(taint_df.loc["seed_bad", "taint_score"], 1.0)
        self.assertEqual(taint_df.loc["seed_bad", "taint_hops"], 0)
        self.assertEqual(taint_df.loc["wA", "taint_score"], 0.5)
        self.assertEqual(taint_df.loc["wA", "taint_hops"], 1)
        self.assertEqual(taint_df.loc["wB", "taint_score"], 0.25)
        self.assertEqual(taint_df.loc["wB", "taint_hops"], 2)

    def test_network_correlation(self) -> None:
        sample_df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime("2026-09-01T10:00:00Z", utc=True),
                "src_ip": "198.51.100.99",
                "src_ip_asn": 48031,  # Bulletproof ASN
                "src_ip_country": "UA",
                "input_addresses": ["w_bp1"],
            },
            {
                "timestamp": pd.to_datetime("2026-09-01T10:01:00Z", utc=True),
                "src_ip": "198.51.100.99",
                "src_ip_asn": 48031,
                "src_ip_country": "UA",
                "input_addresses": ["w_bp1"],
            },
        ])

        correlator = NetworkBlockchainCorrelator()
        corr_df = correlator.correlate(sample_df)

        self.assertIn("w_bp1", corr_df.index)
        self.assertEqual(corr_df.loc["w_bp1", "asn_category"], "BULLETPROOF_HOSTING")
        self.assertEqual(corr_df.loc["w_bp1", "observation_count"], 2)
        self.assertGreaterEqual(corr_df.loc["w_bp1", "network_correlation_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
