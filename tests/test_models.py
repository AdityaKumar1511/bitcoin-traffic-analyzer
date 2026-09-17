"""
Unit tests for AI/ML Models, Network Correlation, Clustering, and Taint Propagation (src/models/ & src/correlation/).
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx

from src.graph.builder import build_graph
from src.graph.heuristics import apply_change_address_heuristic, apply_common_input_heuristic
from src.features import extract_all_features
from src.models.anomaly import (
    AnomalyDetector,
    IsolationForestDetector,
    ReconstructionAnomalyDetector,
    detect_transaction_anomalies,
    detect_wallet_anomalies,
)
from src.models.classifier import EntityRiskClassifier
from src.models.clustering import EntityClusterer, cluster_entities_multimodal
from src.correlation.network_blockchain import NetworkBlockchainCorrelator, correlate_network_blockchain
from src.models.taint_propagation import TaintPropagator, propagate_taint


class TestModelsAndCorrelation(unittest.TestCase):
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

        self.df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime("2026-09-01T10:00:00Z", utc=True),
                "src_ip": "198.51.100.1",
                "dst_ip": "198.51.100.2",
                "src_port": 50001,
                "dst_port": 8333,
                "txid": "tx1" + "0" * 61,
                "input_addresses": ["w_in1", "w_in2"],
                "output_addresses": ["w_out1", "w_out2"],
                "input_amounts": [5.0, 5.0],
                "output_amounts": [9.0, 0.999],
                "fee": 0.001,
                "script_type": "P2PKH",
                "src_ip_country": "US",
                "src_ip_asn": 13335,
            },
            {
                "timestamp": pd.to_datetime("2026-09-01T10:00:20Z", utc=True),
                "src_ip": "198.51.100.1",
                "dst_ip": "198.51.100.3",
                "src_port": 50002,
                "dst_port": 8333,
                "txid": "tx2" + "0" * 61,
                "input_addresses": ["w_out1"],
                "output_addresses": ["w_out3", "w_peel1"],
                "input_amounts": [9.0],
                "output_amounts": [8.0, 0.999],
                "fee": 0.001,
                "script_type": "P2WPKH",
                "src_ip_country": "US",
                "src_ip_asn": 13335,
            },
            {
                "timestamp": pd.to_datetime("2026-09-01T10:00:45Z", utc=True),
                "src_ip": "198.51.100.1",
                "dst_ip": "198.51.100.4",
                "src_port": 50003,
                "dst_port": 8333,
                "txid": "tx3" + "0" * 61,
                "input_addresses": ["w_out3"],
                "output_addresses": ["w_out4", "w_peel2"],
                "input_amounts": [8.0],
                "output_amounts": [7.0, 0.999],
                "fee": 0.001,
                "script_type": "P2WPKH",
                "src_ip_country": "US",
                "src_ip_asn": 13335,
            },
            {
                "timestamp": pd.to_datetime("2026-09-01T14:00:00Z", utc=True),
                "src_ip": "45.142.122.10",
                "dst_ip": "198.51.100.5",
                "src_port": 49152,
                "dst_port": 8333,
                "txid": "tx4" + "0" * 61,
                "input_addresses": ["w_anon"],
                "output_addresses": ["w_dest"],
                "input_amounts": [1.0],
                "output_amounts": [0.999],
                "fee": 0.001,
                "script_type": "P2SH",
                "src_ip_country": "UA",
                "src_ip_asn": 48031,  # Bulletproof hosting ASN
            },
        ])
        self.graph = build_graph(self.df)
        apply_common_input_heuristic(self.graph, self.df)
        apply_change_address_heuristic(self.graph, self.df)
        self.features = extract_all_features(self.graph, self.df)

    def test_anomaly_detector_class(self) -> None:
        detector = AnomalyDetector(contamination=0.4, random_state=42)
        results = detector.fit_predict_wallets(self.sample_features)

        self.assertEqual(len(results), 5)
        self.assertIn("anomaly_score", results.columns)
        self.assertIn("is_anomaly", results.columns)
        self.assertIn("top_deviant_features", results.columns)

        # Anomaly scores should be bounded in [0, 1]
        self.assertTrue((results["anomaly_score"] >= 0.0).all())
        self.assertTrue((results["anomaly_score"] <= 1.0).all())

        # w5_bad should have higher anomaly score than normal w1
        self.assertGreater(results.loc["w5_bad", "anomaly_score"], results.loc["w1", "anomaly_score"])

    def test_anomaly_detection_pipeline(self) -> None:
        wallet_anomalies = detect_wallet_anomalies(self.features["wallets"], contamination=0.15)
        self.assertIsInstance(wallet_anomalies, pd.DataFrame)
        self.assertIn("anomaly_score", wallet_anomalies.columns)
        self.assertIn("is_anomaly", wallet_anomalies.columns)
        self.assertIn("top_deviant_features", wallet_anomalies.columns)

        # Scores must be bounded between 0.0 and 1.0
        self.assertTrue((wallet_anomalies["anomaly_score"] >= 0.0).all())
        self.assertTrue((wallet_anomalies["anomaly_score"] <= 1.0).all())

        tx_anomalies = detect_transaction_anomalies(self.features["transactions"], contamination=0.25)
        self.assertIsInstance(tx_anomalies, pd.DataFrame)
        self.assertEqual(len(tx_anomalies), 4)

    def test_clustering_class(self) -> None:
        clusterer = EntityClusterer()
        cluster_df = clusterer.cluster_entities(self.graph, self.df)
        self.assertIsInstance(cluster_df, pd.DataFrame)
        self.assertIn("cluster_id", cluster_df.columns)
        self.assertIn("cluster_size", cluster_df.columns)
        self.assertIn("primary_ip", cluster_df.columns)
        self.assertIn("is_multi_wallet_entity", cluster_df.columns)

    def test_clustering_multimodal(self) -> None:
        cluster_df = cluster_entities_multimodal(self.graph, self.df)
        self.assertIsInstance(cluster_df, pd.DataFrame)
        self.assertIn("entity_cluster_id", cluster_df.columns)
        self.assertIn("cluster_size", cluster_df.columns)

        # w_in1 and w_in2 co-signed tx1 -> MUST be in the same entity cluster
        cid1 = cluster_df.loc["w_in1", "entity_cluster_id"]
        cid2 = cluster_df.loc["w_in2", "entity_cluster_id"]
        self.assertEqual(cid1, cid2)
        self.assertEqual(cluster_df.loc["w_in1", "clustering_method"], "common_input_ownership")
        self.assertEqual(cluster_df.loc["w_in1", "confidence_score"], 1.0)

    def test_taint_propagation_class(self) -> None:
        g = nx.MultiDiGraph()
        g.add_node("seed_bad", node_type="wallet")
        g.add_node("tx1", node_type="transaction")
        g.add_node("wA", node_type="wallet")
        g.add_node("tx2", node_type="transaction")
        g.add_node("wB", node_type="wallet")

        g.add_edge("seed_bad", "tx1", edge_type="input", amount=10.0)
        g.add_edge("tx1", "wA", edge_type="output", amount=9.99)
        g.add_edge("wA", "tx2", edge_type="input", amount=9.99)
        g.add_edge("tx2", "wB", edge_type="output", amount=9.98)

        propagator = TaintPropagator(decay_factor=0.5, max_hops=3)
        taint_df = propagator.propagate(g, {"seed_bad": 1.0})

        self.assertIn("seed_bad", taint_df.index)
        self.assertIn("wA", taint_df.index)
        self.assertIn("wB", taint_df.index)

        self.assertEqual(taint_df.loc["seed_bad", "taint_score"], 1.0)
        self.assertEqual(taint_df.loc["seed_bad", "taint_hops"], 0)
        self.assertEqual(taint_df.loc["wA", "taint_score"], 0.5)
        self.assertEqual(taint_df.loc["wA", "taint_hops"], 1)
        self.assertEqual(taint_df.loc["wB", "taint_score"], 0.25)
        self.assertEqual(taint_df.loc["wB", "taint_hops"], 2)

    def test_taint_propagation_pipeline(self) -> None:
        taint_df = propagate_taint(self.graph, seed_wallets={"w_in1": 1.0}, decay_factor=0.70, max_hops=3)
        self.assertIsInstance(taint_df, pd.DataFrame)
        self.assertIn("taint_score", taint_df.columns)
        self.assertIn("hops_from_seed", taint_df.columns)

        # Seed node must have taint = 1.0 and hops = 0
        self.assertEqual(taint_df.loc["w_in1", "taint_score"], 1.0)
        self.assertEqual(taint_df.loc["w_in1", "hops_from_seed"], 0)

        # w_out1 is 1 hop downstream from w_in1
        self.assertGreater(taint_df.loc["w_out1", "taint_score"], 0.0)
        self.assertEqual(taint_df.loc["w_out1", "hops_from_seed"], 1)

        # w_out3 is 2 hops downstream from w_in1
        self.assertGreater(taint_df.loc["w_out3", "taint_score"], 0.0)
        self.assertEqual(taint_df.loc["w_out3", "hops_from_seed"], 2)

        # Untainted wallet should have score 0
        self.assertEqual(taint_df.loc["w_anon", "taint_score"], 0.0)
        self.assertEqual(taint_df.loc["w_anon", "hops_from_seed"], -1)

    def test_network_correlation_class(self) -> None:
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

    def test_network_correlation_pipeline(self) -> None:
        corr_df = correlate_network_blockchain(self.graph, self.df)
        self.assertIsInstance(corr_df, pd.DataFrame)
        self.assertIn("network_correlation_score", corr_df.columns)
        self.assertIn("threat_category", corr_df.columns)
        self.assertIn("correlation_explanation", corr_df.columns)

        # w_anon was broadcasted from bulletproof ASN 48031
        self.assertIn("w_anon", corr_df.index)
        w_anon_row = corr_df.loc["w_anon"]
        self.assertGreater(w_anon_row["asn_threat_score"], 0.0)
        self.assertIn("Bulletproof Hosting", w_anon_row["correlation_explanation"])

        # 198.51.100.1 had 3 bursts within 45s (<60s gap)
        self.assertIn("w_out1", corr_df.index)
        w_out1_row = corr_df.loc["w_out1"]
        self.assertGreaterEqual(w_out1_row["burst_count"], 2)

    def test_entity_risk_classifier(self) -> None:
        clf = EntityRiskClassifier(n_estimators=10, random_state=42)
        y = pd.Series([0, 0, 1, 0, 1], index=self.sample_features.index)
        metrics = clf.fit(self.sample_features, y)
        self.assertIn("f1", metrics)
        preds = clf.predict_risk_probabilities(self.sample_features)
        self.assertEqual(len(preds), 5)


if __name__ == "__main__":
    unittest.main()
