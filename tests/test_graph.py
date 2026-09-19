"""
Unit tests for heterogeneous graph builder (src/graph/builder.py).
"""

from pathlib import Path
import unittest

import networkx as nx
import pandas as pd

from src.graph.builder import build_graph, get_graph_summary, get_nodes_by_type
from src.ingestion.parser import parse_csv


class TestGraphBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime("2026-09-01T12:00:00Z", utc=True),
                "src_ip": "198.51.100.1",
                "dst_ip": "198.51.100.2",
                "src_port": 12345,
                "dst_port": 8333,
                "txid": "tx1" + "0" * 61,
                "input_addresses": ["w_in1", "w_in2"],
                "output_addresses": ["w_out1"],
                "input_amounts": [1.5, 2.5],
                "output_amounts": [3.999],
                "fee": 0.001,
                "script_type": "P2PKH",
            },
            {
                "timestamp": pd.to_datetime("2026-09-01T13:00:00Z", utc=True),
                "src_ip": "198.51.100.1",  # Same broadcaster IP
                "dst_ip": "198.51.100.3",
                "src_port": 12346,
                "dst_port": 8333,
                "txid": "tx2" + "0" * 61,
                "input_addresses": ["w_out1"],  # w_out1 re-spent
                "output_addresses": ["w_out2"],
                "input_amounts": [3.999],
                "output_amounts": [3.998],
                "fee": 0.001,
                "script_type": "P2WPKH",
            },
        ])

    def test_build_graph_structure(self) -> None:
        graph = build_graph(self.sample_df)
        self.assertIsInstance(graph, nx.MultiDiGraph)

        summary = get_graph_summary(graph)
        self.assertEqual(summary["transaction_nodes"], 2)
        self.assertEqual(summary["ip_nodes"], 1)  # Only 1 unique src_ip
        self.assertEqual(summary["wallet_nodes"], 4)  # w_in1, w_in2, w_out1, w_out2

        self.assertEqual(summary["input_edges"], 3)  # 2 in tx1 + 1 in tx2
        self.assertEqual(summary["output_edges"], 2)  # 1 in tx1 + 1 in tx2
        self.assertEqual(summary["broadcast_edges"], 2)  # 1 per tx

        # Check node attributes
        tx1_id = "tx1" + "0" * 61
        tx1_node = graph.nodes[tx1_id]
        self.assertEqual(tx1_node["node_type"], "transaction")
        self.assertEqual(tx1_node["fee"], 0.001)
        self.assertEqual(tx1_node["script_type"], "P2PKH")

        ip_node = graph.nodes["198.51.100.1"]
        self.assertEqual(ip_node["node_type"], "ip")

        wallet_node = graph.nodes["w_in1"]
        self.assertEqual(wallet_node["node_type"], "wallet")

        # Check edge directions & attributes
        # Wallet -> Transaction (input)
        in_edges = graph.get_edge_data("w_in1", tx1_id)
        self.assertIsNotNone(in_edges)
        first_edge = list(in_edges.values())[0]
        self.assertEqual(first_edge["edge_type"], "input")
        self.assertEqual(first_edge["amount"], 1.5)

        # Transaction -> Wallet (output)
        out_edges = graph.get_edge_data(tx1_id, "w_out1")
        self.assertIsNotNone(out_edges)
        first_out = list(out_edges.values())[0]
        self.assertEqual(first_out["edge_type"], "output")
        self.assertEqual(first_out["amount"], 3.999)

        # Transaction -> IP (broadcast)
        b_edges = graph.get_edge_data(tx1_id, "198.51.100.1")
        self.assertIsNotNone(b_edges)
        first_b = list(b_edges.values())[0]
        self.assertEqual(first_b["edge_type"], "broadcast")
        self.assertEqual(first_b["src_port"], 12345)

    def test_get_nodes_by_type(self) -> None:
        graph = build_graph(self.sample_df)
        wallets = get_nodes_by_type(graph, "wallet")
        transactions = get_nodes_by_type(graph, "transaction")
        ips = get_nodes_by_type(graph, "ip")

        self.assertEqual(len(wallets), 4)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(len(ips), 1)

        self.assertIn("w_in1", wallets)
        self.assertIn("198.51.100.1", ips)
        self.assertIn("tx1" + "0" * 61, transactions)

    def test_build_graph_with_geoip_columns(self) -> None:
        df_geo = self.sample_df.copy()
        df_geo["src_ip_country"] = ["US", "US"]
        df_geo["src_ip_asn"] = [15169, 15169]

        graph = build_graph(df_geo)
        ip_node = graph.nodes["198.51.100.1"]
        self.assertEqual(ip_node["country"], "US")
        self.assertEqual(ip_node["asn"], 15169)

    def test_malformed_row_skipped(self) -> None:
        bad_df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime("2026-09-01T12:00:00Z", utc=True),
                "src_ip": "1.1.1.1",
                "dst_ip": "2.2.2.2",
                "src_port": 8333,
                "dst_port": 8333,
                "txid": "",  # Missing txid
                "input_addresses": ["addr1"],
                "output_addresses": ["addr2"],
                "input_amounts": [1.0],
                "output_amounts": [0.99],
                "fee": 0.01,
                "script_type": "P2PKH",
            }
        ])
        graph = build_graph(bad_df)
        self.assertEqual(graph.number_of_nodes(), 0)

    def test_synthetic_transactions_graph(self) -> None:
        csv_path = Path("data/raw/synthetic_transactions.csv")
        if not csv_path.exists():
            self.skipTest("synthetic_transactions.csv not found")

        df = parse_csv(csv_path)
        graph = build_graph(df)
        summary = get_graph_summary(graph)

        self.assertEqual(summary["transaction_nodes"], 2000)
        self.assertGreater(summary["wallet_nodes"], 5000)
        self.assertGreater(summary["ip_nodes"], 1000)
        self.assertEqual(summary["broadcast_edges"], 2000)


from src.graph.heuristics import (
    apply_change_address_heuristic,
    apply_common_input_heuristic,
    get_wallet_clusters,
)


class TestGraphHeuristics(unittest.TestCase):
    def setUp(self) -> None:
        self.tx1_id = "tx1" + "a" * 61
        self.tx2_id = "tx2" + "b" * 61
        self.df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime("2026-09-01T10:00:00Z", utc=True),
                "src_ip": "1.2.3.4",
                "dst_ip": "5.6.7.8",
                "src_port": 8333,
                "dst_port": 8333,
                "txid": self.tx1_id,
                "input_addresses": ["wA", "wB"],
                "output_addresses": ["wPay", "wChange"],
                "input_amounts": [1.0, 2.0],
                "output_amounts": [1.0, 1.999],  # 1.0 is round, 1.999 is non-round
                "fee": 0.001,
                "script_type": "P2PKH",
            },
            {
                "timestamp": pd.to_datetime("2026-09-01T11:00:00Z", utc=True),
                "src_ip": "1.2.3.4",
                "dst_ip": "5.6.7.8",
                "src_port": 8333,
                "dst_port": 8333,
                "txid": self.tx2_id,
                "input_addresses": ["wA", "wB", "wC"],  # wA and wB co-occur again, with wC
                "output_addresses": ["wOut"],
                "input_amounts": [0.5, 0.5, 1.0],
                "output_amounts": [1.999],
                "fee": 0.001,
                "script_type": "P2WPKH",
            },
        ])

    def test_apply_common_input_heuristic(self) -> None:
        graph = build_graph(self.df)
        apply_common_input_heuristic(graph, self.df)

        # In tx1: wA <-> wB edge added (count=1)
        # In tx2: wA <-> wB incremented (count=2), wA <-> wC (count=1), wB <-> wC (count=1)
        self.assertTrue(graph.has_edge("wA", "wB"))
        self.assertTrue(graph.has_edge("wB", "wA"))
        self.assertTrue(graph.has_edge("wA", "wC"))

        edge_ab = [d for _, _, d in graph.edges("wA", data=True) if d.get("edge_type") == "common_input_ownership" and _ == "wA"]
        # Find edge to wB
        ab_data = None
        for u, v, d in graph.edges(data=True):
            if u == "wA" and v == "wB" and d.get("edge_type") == "common_input_ownership":
                ab_data = d
                break
        self.assertIsNotNone(ab_data)
        self.assertEqual(ab_data["co_occurrence_count"], 2)

    def test_apply_change_address_heuristic(self) -> None:
        graph = build_graph(self.df)
        apply_change_address_heuristic(graph, self.df)

        # In tx1: wPay has round amount 1.0, wChange has non-round amount 1.999
        # wChange stands out as likely change
        change_edges = [
            d for u, v, d in graph.edges(data=True)
            if u == self.tx1_id and v == "wChange" and d.get("edge_type") == "output"
        ]
        self.assertEqual(len(change_edges), 1)
        self.assertTrue(change_edges[0].get("likely_change_address"))

        # wPay should NOT be marked
        pay_edges = [
            d for u, v, d in graph.edges(data=True)
            if u == self.tx1_id and v == "wPay" and d.get("edge_type") == "output"
        ]
        self.assertEqual(len(pay_edges), 1)
        self.assertFalse(pay_edges[0].get("likely_change_address", False))

    def test_get_wallet_clusters(self) -> None:
        graph = build_graph(self.df)
        apply_common_input_heuristic(graph, self.df)
        clusters = get_wallet_clusters(graph)

        # wA, wB, wC are connected into the same cluster
        self.assertEqual(clusters["wA"], clusters["wB"])
        self.assertEqual(clusters["wB"], clusters["wC"])

        # wPay and wChange are separate singleton clusters
        self.assertNotEqual(clusters["wA"], clusters["wPay"])
        self.assertNotEqual(clusters["wPay"], clusters["wChange"])


from src.graph.peel_chain import (
    annotate_graph_with_peel_chains,
    detect_peel_chains,
    is_peel_transaction,
)


class TestPeelChainDetection(unittest.TestCase):
    def test_is_peel_transaction(self) -> None:
        # Valid peel: 10.0 and 0.5 (ratio = 0.5/10.5 = 0.0476 <= 0.15)
        self.assertTrue(is_peel_transaction([10.0, 0.5]))
        self.assertTrue(is_peel_transaction({"output_amounts": [0.2, 5.0]}))

        # Balanced split: 5.0 and 5.0 (ratio = 0.5 > 0.15) -> False
        self.assertFalse(is_peel_transaction([5.0, 5.0]))

        # Invalid lengths (less than 2 outputs)
        self.assertFalse(is_peel_transaction([]))
        self.assertFalse(is_peel_transaction([10.0]))

        # Multi-output peel: dominant carrier with small peels
        self.assertTrue(is_peel_transaction([10.0, 1.0, 0.5]))
        # Multi-output non-peel: non-carrier ratio exceeds threshold
        self.assertFalse(is_peel_transaction([10.0, 3.0, 2.0]))

    def test_detect_peel_chains(self) -> None:
        # Create a synthetic 3-hop peel chain: w0 -> w1 -> w2 -> w3
        peel_rows = [
            {
                "timestamp": pd.to_datetime("2026-09-01T10:00:00Z", utc=True),
                "src_ip": "1.1.1.1",
                "dst_ip": "2.2.2.2",
                "src_port": 8333,
                "dst_port": 8333,
                "txid": "tx_p1" + "0" * 59,
                "input_addresses": ["w0"],
                "output_addresses": ["w1", "peel1"],
                "input_amounts": [10.0],
                "output_amounts": [9.5, 0.5],  # 0.5 peeled
                "fee": 0.0001,
                "script_type": "P2PKH",
            },
            {
                "timestamp": pd.to_datetime("2026-09-01T11:00:00Z", utc=True),
                "src_ip": "1.1.1.1",
                "dst_ip": "2.2.2.2",
                "src_port": 8333,
                "dst_port": 8333,
                "txid": "tx_p2" + "0" * 59,
                "input_addresses": ["w1"],
                "output_addresses": ["peel2", "w2"],
                "input_amounts": [9.5],
                "output_amounts": [0.4, 9.1],  # 0.4 peeled
                "fee": 0.0001,
                "script_type": "P2PKH",
            },
            {
                "timestamp": pd.to_datetime("2026-09-01T12:00:00Z", utc=True),
                "src_ip": "1.1.1.1",
                "dst_ip": "2.2.2.2",
                "src_port": 8333,
                "dst_port": 8333,
                "txid": "tx_p3" + "0" * 59,
                "input_addresses": ["w2"],
                "output_addresses": ["w3", "peel3"],
                "input_amounts": [9.1],
                "output_amounts": [8.8, 0.3],  # 0.3 peeled
                "fee": 0.0001,
                "script_type": "P2PKH",
            },
        ]
        df_peel = pd.DataFrame(peel_rows)
        graph = build_graph(df_peel)

        chains = detect_peel_chains(graph, df_peel, min_chain_length=3, peel_ratio_threshold=0.15)
        self.assertEqual(len(chains), 1)

        chain = chains[0]
        self.assertEqual(chain["total_chain_length"], 3)
        self.assertEqual(chain["chain_wallets"], ["w0", "w1", "w2", "w3"])
        self.assertEqual(len(chain["chain_txids"]), 3)
        self.assertAlmostEqual(chain["starting_amount"], 10.0)
        self.assertAlmostEqual(chain["ending_amount"], 8.8)
        self.assertAlmostEqual(chain["total_peeled"], 1.2)

        # Test min_chain_length filter: length 4 should return []
        chains_len4 = detect_peel_chains(graph, df_peel, min_chain_length=4)
        self.assertEqual(len(chains_len4), 0)

        # Test annotation
        annotate_graph_with_peel_chains(graph, chains)
        self.assertEqual(graph.nodes["w0"]["peel_chain_id"], 0)
        self.assertEqual(graph.nodes["w0"]["peel_chain_position"], 0)
        self.assertEqual(graph.nodes["w3"]["peel_chain_position"], 3)


if __name__ == "__main__":
    unittest.main()
