"""
Multi-Hop Taint Propagation Module for Forensic Risk Flow Analysis.

Implements the forensic poison/proportional haircut risk propagation model across the
heterogeneous Bitcoin transaction graph. Suspicion scores flow from seed
suspicious wallets along directed transaction pathways, decaying exponentially
with hop distance and scaled by the proportional output share.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from src.features import extract_all_features
from src.graph.builder import build_graph, get_nodes_by_type
from src.graph.heuristics import apply_change_address_heuristic, apply_common_input_heuristic
from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TaintPropagator:
    """
    Multi-hop taint propagation engine computing risk decay and provenance paths
    using the proportional haircut model.
    """

    def __init__(
        self,
        decay_factor: float = 0.65,
        max_hops: int = 4,
        min_taint_threshold: float = 0.01,
    ) -> None:
        """
        Initialize Taint Propagator.

        Args:
            decay_factor: Multiplier applied per graph hop (0.0 < decay <= 1.0).
            max_hops: Maximum number of transaction hops to traverse.
            min_taint_threshold: Minimum taint score to retain in records.
        """
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.min_taint_threshold = min_taint_threshold

    def propagate(
        self,
        graph: nx.MultiDiGraph,
        seed_risks: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Propagate taint from seed wallets forward across transaction links.

        Args:
            graph: Heterogeneous transaction MultiDiGraph.
            seed_risks: Mapping from seed wallet_id (str) to initial risk score (float in [0, 1]).

        Returns:
            pd.DataFrame indexed by wallet_id with columns:
                taint_score, taint_hops, hops_from_seed, taint_source,
                nearest_seed_wallet, taint_path, is_tainted
        """
        # Collect all wallet nodes in graph
        all_wallets = [
            node for node, data in graph.nodes(data=True) if data.get("node_type") == "wallet"
        ]
        if not all_wallets and graph.number_of_nodes() > 0:
            all_wallets = list(graph.nodes())

        if not seed_risks or graph.number_of_nodes() == 0:
            records = []
            for w in all_wallets:
                records.append({
                    "wallet_id": w,
                    "taint_score": 0.0,
                    "taint_hops": -1,
                    "hops_from_seed": -1,
                    "taint_source": "None",
                    "nearest_seed_wallet": "None",
                    "taint_path": "",
                    "is_tainted": False,
                })
            df = pd.DataFrame(records).set_index("wallet_id") if records else pd.DataFrame(
                columns=[
                    "taint_score",
                    "taint_hops",
                    "hops_from_seed",
                    "taint_source",
                    "nearest_seed_wallet",
                    "taint_path",
                    "is_tainted",
                ]
            ).set_index(pd.Index([], name="wallet_id"))
            return df

        # wallet_id -> (max_taint_score, min_hops, seed_source, path_str)
        taint_map: Dict[str, Tuple[float, int, str, str]] = {}

        # Initialize seeds
        for seed_wallet, score in seed_risks.items():
            if seed_wallet in graph:
                taint_map[seed_wallet] = (float(score), 0, seed_wallet, seed_wallet)

        # BFS Queue holds: (current_wallet, current_taint, hop_count, seed_source, path_list)
        queue = deque(
            [
                (w, score, 0, w, [w])
                for w, score in seed_risks.items()
                if w in graph and score > 0
            ]
        )

        visited_hops: Dict[Tuple[str, str], int] = defaultdict(lambda: 999)

        while queue:
            curr_w, curr_taint, hop, seed, path = queue.popleft()

            if hop >= self.max_hops:
                continue

            # In heterogeneous graph: Wallet -(input)-> Transaction -(output)-> Wallet
            for _, tx_node, edge_data in graph.out_edges(curr_w, data=True):
                if edge_data.get("edge_type") != "input":
                    continue

                # Find all outputs from this transaction to compute total output amount
                out_edges = [
                    (u, v, d) for u, v, d in graph.out_edges(tx_node, data=True)
                    if d.get("edge_type") == "output" and v != curr_w
                ]
                total_output_amt = sum(d.get("amount", 0.0) for _, _, d in out_edges)

                # From transaction, forward taint to destination wallets
                for _, next_w, out_data in out_edges:
                    dest_amt = out_data.get("amount", 0.0)
                    # Haircut model: fraction of output received
                    if total_output_amt > 0 and dest_amt > 0:
                        output_ratio = dest_amt / total_output_amt
                    else:
                        output_ratio = 1.0

                    next_taint = round(curr_taint * self.decay_factor * output_ratio, 4)
                    next_hop = hop + 1

                    if next_taint < self.min_taint_threshold:
                        continue

                    # Avoid redundant looping with greater or equal hops
                    if next_hop >= visited_hops[(next_w, seed)]:
                        continue
                    visited_hops[(next_w, seed)] = next_hop

                    next_path = path + [str(tx_node)[:8] + "...", next_w]
                    path_str = " -> ".join(next_path)

                    if next_w not in taint_map or next_taint > taint_map[next_w][0]:
                        taint_map[next_w] = (next_taint, next_hop, seed, path_str)

                    queue.append((next_w, next_taint, next_hop, seed, next_path))

        # Include all wallets in the graph
        records = []
        for w in all_wallets:
            if w in taint_map:
                score, hops, source, p_str = taint_map[w]
                records.append({
                    "wallet_id": w,
                    "taint_score": score,
                    "taint_hops": hops,
                    "hops_from_seed": hops,
                    "taint_source": source,
                    "nearest_seed_wallet": source,
                    "taint_path": p_str,
                    "is_tainted": score >= self.min_taint_threshold,
                })
            else:
                records.append({
                    "wallet_id": w,
                    "taint_score": 0.0,
                    "taint_hops": -1,
                    "hops_from_seed": -1,
                    "taint_source": "None",
                    "nearest_seed_wallet": "None",
                    "taint_path": "",
                    "is_tainted": False,
                })

        result_df = pd.DataFrame(records).set_index("wallet_id")
        total_tainted = int((result_df["taint_score"] >= self.min_taint_threshold).sum())
        logger.info(
            "Taint propagation complete: %d wallets tainted from %d seed sources.",
            total_tainted,
            len(seed_risks),
        )
        return result_df


def propagate_taint(
    graph: nx.MultiDiGraph,
    seed_wallets: Optional[Dict[str, float]] = None,
    decay_factor: float = 0.70,
    max_hops: int = 4,
    min_taint_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Functional wrapper for multi-hop taint propagation with optional auto-seeding.
    """
    if seed_wallets is None or len(seed_wallets) == 0:
        seed_wallets = {}
        for node, data in graph.nodes(data=True):
            if data.get("node_type") == "wallet":
                if data.get("peel_chain_position") == 0:
                    seed_wallets[node] = 1.0

    propagator = TaintPropagator(
        decay_factor=decay_factor,
        max_hops=max_hops,
        min_taint_threshold=min_taint_threshold,
    )
    return propagator.propagate(graph, seed_wallets)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Multi-Hop Taint Risk Propagation on Transaction Graph.")
    parser.add_argument("--data", "-d", type=Path, default=Path("data/raw/synthetic_transactions.csv"))
    parser.add_argument("--decay", type=float, default=0.65)
    parser.add_argument("--hops", type=int, default=4)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    print(f"[*] Ingesting transactions from: {args.data}")
    df = parse_file(args.data)
    graph = build_graph(df)

    taint_df = propagate_taint(graph, decay_factor=args.decay, max_hops=args.hops)

    print("\n" + "=" * 65)
    print("                 TAINT PROPAGATION SUMMARY")
    print("=" * 65)
    print(f"Total Wallets Evaluated:    {len(taint_df):,}")
    print(f"Total Wallets Tainted:      {int(taint_df['is_tainted'].sum()):,}")
    print("=" * 65)


if __name__ == "__main__":
    main()
