"""
Unit tests for feature engineering modules (src/features/).
"""

import unittest
import pandas as pd
import numpy as np
import networkx as nx

from src.graph.builder import build_graph
from src.graph.heuristics import apply_common_input_heuristic, apply_change_address_heuristic
from src.features.wallet_features import compute_wallet_features
from src.features.transaction_features import compute_transaction_features
from src.features.network_features import compute_network_features, compute_wallet_network_features
from src.features.temporal_features import compute_temporal_features
from src.features import extract_all_features


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime("2026-09-01T10:00:00Z", utc=True),
                "src_ip": "198.51.100.1",
                "dst_ip": "198.51.100.2",
                "src_port": 50001,
                "dst_port": 8333,
                "txid": "tx1" + "0" * 61,
                "input_addresses": ["addrA", "addrB"],
                "output_addresses": ["addrC", "addrD"],
                "input_amounts": [2.0, 3.0],
                "output_amounts": [4.0, 0.999],
                "fee": 0.001,
                "script_type": "P2PKH",
                "src_ip_country": "US",
                "src_ip_asn": 13335,
            },
            {
                "timestamp": pd.to_datetime("2026-09-01T10:00:30Z", utc=True),  # 30s later (burst)
                "src_ip": "198.51.100.1",
                "dst_ip": "198.51.100.3",
                "src_port": 50002,
                "dst_port": 8333,
                "txid": "tx2" + "0" * 61,
                "input_addresses": ["addrC"],
                "output_addresses": ["addrE"],
                "input_amounts": [4.0],
                "output_amounts": [3.999],
                "fee": 0.001,
                "script_type": "P2WPKH",
                "src_ip_country": "US",
                "src_ip_asn": 13335,
            },
            {
                "timestamp": pd.to_datetime("2026-09-01T14:00:00Z", utc=True),
                "src_ip": "198.51.100.99",
                "dst_ip": "198.51.100.4",
                "src_port": 8333,
                "dst_port": 8333,
                "txid": "tx3" + "0" * 61,
                "input_addresses": ["addrF"],
                "output_addresses": ["addrG"],
                "input_amounts": [1.0],
                "output_amounts": [0.999],
                "fee": 0.001,
                "script_type": "P2SH",
                "src_ip_country": "RU",
                "src_ip_asn": 48031,  # In high-risk bulletproof ASN list
            },
        ])
        self.graph = build_graph(self.df)
        apply_common_input_heuristic(self.graph, self.df)
        apply_change_address_heuristic(self.graph, self.df)

    def test_wallet_features(self) -> None:
        wf = compute_wallet_features(self.graph, self.df)
        self.assertIsInstance(wf, pd.DataFrame)
        self.assertIn("addrA", wf.index)
        self.assertIn("addrC", wf.index)

        # Check expected columns
        expected_cols = [
            "in_degree", "out_degree", "tx_count", "total_btc_in", "total_btc_out",
            "net_flow", "fan_in_ratio", "fan_out_ratio", "distinct_ip_count",
            "cluster_size", "reuse_count"
        ]
        for col in expected_cols:
            self.assertIn(col, wf.columns)

        # addrA sent 2.0 BTC in tx1, received 0
        self.assertEqual(wf.loc["addrA", "total_btc_out"], 2.0)
        self.assertEqual(wf.loc["addrA", "total_btc_in"], 0.0)
        self.assertEqual(wf.loc["addrA", "net_flow"], -2.0)

        # addrC received 4.0 in tx1, sent 4.0 in tx2 -> net_flow 0.0
        self.assertEqual(wf.loc["addrC", "total_btc_in"], 4.0)
        self.assertEqual(wf.loc["addrC", "total_btc_out"], 4.0)
        self.assertEqual(wf.loc["addrC", "net_flow"], 0.0)

    def test_transaction_features(self) -> None:
        tf = compute_transaction_features(self.graph, self.df)
        self.assertIsInstance(tf, pd.DataFrame)
        self.assertEqual(len(tf), 3)

        tx1_id = "tx1" + "0" * 61
        tx3_id = "tx3" + "0" * 61
        self.assertIn(tx1_id, tf.index)

        # tx1 has 2 inputs, 2 outputs
        self.assertEqual(tf.loc[tx1_id, "num_inputs"], 2)
        self.assertEqual(tf.loc[tx1_id, "num_outputs"], 2)
        self.assertEqual(tf.loc[tx1_id, "total_input_value"], 5.0)
        self.assertEqual(tf.loc[tx1_id, "is_script_p2pkh"], 1)

        # tx3 was broadcast from high-risk ASN 48031
        self.assertEqual(tf.loc[tx3_id, "broadcast_ip_is_high_risk_asn"], 1)

    def test_network_features(self) -> None:
        ip_df = compute_network_features(self.graph, self.df)
        self.assertIsInstance(ip_df, pd.DataFrame)
        self.assertIn("198.51.100.1", ip_df.index)
        self.assertIn("198.51.100.99", ip_df.index)

        # 198.51.100.1 broadcasted 2 txs within 30s -> burst_count >= 1
        ip1 = ip_df.loc["198.51.100.1"]
        self.assertEqual(ip1["ip_tx_count"], 2)
        self.assertEqual(ip1["burst_count"], 1)
        self.assertTrue(ip1["uses_non_standard_port"])

        # 198.51.100.99 has high-risk ASN
        ip_high_risk = ip_df.loc["198.51.100.99"]
        self.assertTrue(ip_high_risk["is_high_risk_asn"])
        self.assertEqual(ip_high_risk["geo_risk_score"], 1.0)

        # Wallet network features mapping
        wnf = compute_wallet_network_features(self.graph, self.df)
        self.assertIsInstance(wnf, pd.DataFrame)
        self.assertIn("addrF", wnf.index)
        self.assertTrue(wnf.loc["addrF", "wallet_has_high_risk_ip"])

    def test_temporal_features(self) -> None:
        temp = compute_temporal_features(self.graph, self.df)
        self.assertIsInstance(temp, pd.DataFrame)
        self.assertIn("addrC", temp.index)

        # Check expected temporal columns
        for col in ["hour_entropy", "day_of_week_entropy", "burstiness", "mean_inter_tx_seconds"]:
            self.assertIn(col, temp.columns)

    def test_extract_all_features(self) -> None:
        all_feats = extract_all_features(self.graph, self.df)
        self.assertIn("wallets", all_feats)
        self.assertIn("transactions", all_feats)
        self.assertIn("ips", all_feats)

        wallets_df = all_feats["wallets"]
        tx_df = all_feats["transactions"]
        ip_df = all_feats["ips"]

        self.assertFalse(wallets_df.empty)
        self.assertFalse(tx_df.empty)
        self.assertFalse(ip_df.empty)
        self.assertFalse(wallets_df.isna().any().any())

    def test_empty_graph_handling(self) -> None:
        empty_graph = nx.MultiDiGraph()
        empty_df = pd.DataFrame()

        wf = compute_wallet_features(empty_graph, empty_df)
        self.assertTrue(wf.empty)

        tf = compute_transaction_features(empty_graph, empty_df)
        self.assertTrue(tf.empty)

        ip_df = compute_network_features(empty_graph, empty_df)
        self.assertTrue(ip_df.empty)


if __name__ == "__main__":
    unittest.main()
