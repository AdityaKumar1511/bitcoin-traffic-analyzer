"""
Multi-Hop Taint Propagation Module for Forensic Risk Flow Analysis.

Implements the forensic poison/haircut risk propagation model across the
heterogeneous Bitcoin transaction graph. Suspicion scores flow from seed
suspicious wallets along directed transaction pathways, decaying exponentially
with hop distance.
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
    Multi-hop taint propagation engine computing risk decay and provenance paths.
    """

    def __init__(
        self,
        decay_factor: float = 0.6,
        max_hops: int = 4,
        min_taint_threshold: float = 0.05,
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
                taint_score (float), taint_hops (int), taint_source (str), taint_path (str)
        """
        if not seed_risks or graph.number_of_nodes() == 0:
            return pd.DataFrame(
                columns=["taint_score", "taint_hops", "taint_source", "taint_path"]
            ).set_index(pd.Index([], name="wallet_id"))

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

            # In the heterogeneous graph: Wallet -(input)-> Transaction -(output)-> Wallet
            # Find all outgoing transactions from curr_w
            for _, tx_node, edge_data in graph.out_edges(curr_w, data=True):
                if edge_data.get("edge_type") != "input":
                    continue

                # From the transaction, find all receiving wallets
                for _, next_w, out_data in graph.out_edges(tx_node, data=True):
                    if out_data.get("edge_type") != "output" or next_w == curr_w:
                        continue

                    # Calculate decayed taint
                    next_taint = round(curr_taint * self.decay_factor, 4)
                    next_hop = hop + 1

                    if next_taint < self.min_taint_threshold:
                        continue

                    # Avoid infinite looping on the same (wallet, seed) with greater or equal hops
                    if next_hop >= visited_hops[(next_w, seed)]:
                        continue
                    visited_hops[(next_w, seed)] = next_hop

                    next_path = path + [str(tx_node)[:8] + "...", next_w]
                    path_str = " -> ".join(next_path)

                    if next_w not in taint_map or next_taint > taint_map[next_w][0]:
                        taint_map[next_w] = (next_taint, next_hop, seed, path_str)

                    queue.append((next_w, next_taint, next_hop, seed, next_path))

        records = []
        for w, (score, hops, source, p_str) in taint_map.items():
            records.append({
                "wallet_id": w,
                "taint_score": score,
                "taint_hops": hops,
                "taint_source": source,
                "taint_path": p_str,
            })

        result_df = pd.DataFrame(records).set_index("wallet_id")
        logger.info(
            "Taint propagation complete: %d wallets tainted from %d seed sources.",
            len(result_df),
            len(seed_risks),
        )
        return result_df


def run_taint_propagation(
    input_file: Union[str, Path] = "data/raw/synthetic_transactions.csv",
    seed_wallet: Optional[str] = None,
) -> pd.DataFrame:
    """Run taint propagation on transaction dataset."""
    df = parse_file(Path(input_file))
    graph = build_graph(df)
    apply_common_input_heuristic(graph, df)
    apply_change_address_heuristic(graph, df)

    seeds: Dict[str, float] = {}
    if seed_wallet:
        seeds[seed_wallet] = 1.0
    else:
        # Check if ground truth or known ransomware seeds are present
        gt_path = Path(input_file).parent / "ground_truth.csv"
        if gt_path.is_file():
            gt_df = pd.read_csv(gt_path)
            bad_wallets = gt_df[
                (gt_df["entity_type"] == "wallet") & (gt_df["is_criminal"])
            ]["entity_id"].tolist()
            for w in bad_wallets[:5]:  # Take top 5 seeds
                seeds[w] = 1.0
        
        if not seeds:
            # Pick highest volume wallet as demo seed
            all_wallets = get_nodes_by_type(graph, "wallet")
            if all_wallets:
                seeds[all_wallets[0]] = 1.0

    propagator = TaintPropagator()
    return propagator.propagate(graph, seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-hop risk taint propagation.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/raw/synthetic_transactions.csv",
        help="Path to transaction file",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=str,
        default=None,
        help="Specific seed wallet address to trace taint from",
    )
    args = parser.parse_args()

    taint_df = run_taint_propagation(args.input, seed_wallet=args.seed)

    print("\n" + "=" * 65)
    print("           MULTI-HOP TAINT PROPAGATION SUMMARY")
    print("=" * 65)
    print(f"Total Tainted Wallets: {len(taint_df):,}")

    top_tainted = taint_df.sort_values(by=["taint_score", "taint_hops"], ascending=[False, True]).head(10)
    print("\n--- TOP TAINTED ENTITIES ---")
    for w_id, row in top_tainted.iterrows():
        print(
            f"  Wallet: {w_id}\n"
            f"    Score: {row['taint_score']:.4f} | Hops: {row['taint_hops']} | "
            f"Source: {row['taint_source']}\n"
            f"    Path: {row['taint_path']}"
        )

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
