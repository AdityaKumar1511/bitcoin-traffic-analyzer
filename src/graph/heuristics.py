"""
Blockchain Forensics Heuristics Module.

Derives forensic edges and attributes on the heterogeneous Bitcoin transaction graph:
1. Common-Input-Ownership Heuristic:
   Links wallet addresses co-signing inputs of the same transaction, establishing
   multi-input clustering.
2. Change Address Heuristic:
   Identifies candidate change addresses returning unspent change to the sender
   based on first-time appearance and non-round amount signatures.
3. Wallet Clustering:
   Computes connected components across common-input-ownership edges to assign
   cluster IDs to all wallets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import pandas as pd

from src.graph.builder import build_graph, get_nodes_by_type
from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Standard round Bitcoin denominations frequently used for intentional payments
ROUND_DENOMINATIONS = {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0}


def _is_round_amount(amount: float) -> bool:
    """Check if an amount appears to be an intentional round payment."""
    if amount in ROUND_DENOMINATIONS:
        return True
    # Check if amount has at most 2 decimal places (e.g. 1.5, 0.25, 12.0)
    if round(amount, 2) == amount:
        return True
    return False


def apply_common_input_heuristic(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
) -> nx.MultiDiGraph:
    """
    Apply the common-input-ownership heuristic to the transaction graph.

    For transactions with 2 or more input addresses, all input addresses are
    inferred to belong to the same entity. Adds derived bidirectional edges:
    (w1 <-> w2) with edge_type='common_input_ownership'. If an edge already
    exists between this pair, increments its 'co_occurrence_count'.

    Modifies the graph in place and returns it.

    Args:
        graph: Heterogeneous transaction MultiDiGraph.
        df: Normalized transaction DataFrame.

    Returns:
        The modified graph containing derived common_input_ownership edges.
    """
    if df.empty or graph.number_of_nodes() == 0:
        return graph

    edges_added = 0
    edges_incremented = 0

    for row in df.itertuples(index=True):
        txid = getattr(row, "txid", None)
        in_addrs = getattr(row, "input_addresses", [])

        if not isinstance(in_addrs, (list, tuple)) or len(in_addrs) < 2:
            continue

        # Extract unique valid wallet addresses present in the graph
        unique_wallets = list(dict.fromkeys(
            addr.strip() for addr in in_addrs
            if addr and isinstance(addr, str) and addr.strip() in graph
        ))

        if len(unique_wallets) < 2:
            continue

        # Connect every pair of co-input wallets
        for w1, w2 in itertools.combinations(unique_wallets, 2):
            for u, v in [(w1, w2), (w2, w1)]:
                # Check if common_input_ownership edge already exists
                existing_key = None
                if graph.has_edge(u, v):
                    for key, edge_data in graph[u][v].items():
                        if edge_data.get("edge_type") == "common_input_ownership":
                            existing_key = key
                            break

                if existing_key is not None:
                    graph[u][v][existing_key]["co_occurrence_count"] += 1
                    edges_incremented += 1
                else:
                    graph.add_edge(
                        u,
                        v,
                        edge_type="common_input_ownership",
                        txid=txid,
                        co_occurrence_count=1,
                    )
                    edges_added += 1

    logger.info(
        "Common-input heuristic applied: %d directed edges added, %d incremented.",
        edges_added,
        edges_incremented,
    )
    return graph


def apply_change_address_heuristic(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
) -> nx.MultiDiGraph:
    """
    Apply the change address identification heuristic to output edges.

    Identifies likely change addresses for transactions with 2+ outputs using:
    1. First-time-seen address: Output address has never appeared in prior transactions.
    2. Non-round amount: Amount has granular satoshi precision while other outputs
       have round payment amounts.

    When a clear candidate emerges, marks the transaction -> wallet output edge
    with 'likely_change_address' = True.

    Modifies the graph in place and returns it.

    Args:
        graph: Heterogeneous transaction MultiDiGraph.
        df: Normalized transaction DataFrame.

    Returns:
        The modified graph with marked output edges.
    """
    if df.empty or graph.number_of_nodes() == 0:
        return graph

    # Sort chronologically if timestamp is present to maintain prior-seen ordering
    if "timestamp" in df.columns:
        df_ordered = df.sort_values(by="timestamp", na_position="last")
    else:
        df_ordered = df

    seen_outputs: Set[str] = set()
    total_marked = 0

    for row in df_ordered.itertuples(index=True):
        txid = getattr(row, "txid", None)
        out_addrs = getattr(row, "output_addresses", [])
        out_amts = getattr(row, "output_amounts", [])

        if (
            not txid
            or not isinstance(out_addrs, (list, tuple))
            or not isinstance(out_amts, (list, tuple))
            or len(out_addrs) < 2
            or len(out_addrs) != len(out_amts)
        ):
            # Record seen outputs even for single-output transactions
            if isinstance(out_addrs, (list, tuple)):
                seen_outputs.update(str(a).strip() for a in out_addrs if a)
            continue

        clean_txid = str(txid).strip()
        candidates: List[Dict[str, Any]] = []

        # Analyze each output candidate
        for addr, amt in zip(out_addrs, out_amts):
            if not addr:
                continue
            clean_addr = str(addr).strip()
            try:
                amt_val = float(amt)
            except (ValueError, TypeError):
                amt_val = 0.0

            is_first_seen = clean_addr not in seen_outputs
            is_round = _is_round_amount(amt_val)

            candidates.append({
                "address": clean_addr,
                "amount": amt_val,
                "first_seen": is_first_seen,
                "is_round": is_round,
            })

        # Update running seen set for future transactions
        seen_outputs.update(c["address"] for c in candidates)

        if len(candidates) < 2:
            continue

        # Score candidates
        # Rule 1: First-time seen (+2 points, strongest signal)
        # Rule 2: Non-round leftover amount when others are round (+1 point)
        any_round = any(c["is_round"] for c in candidates)
        any_non_round = any(not c["is_round"] for c in candidates)
        has_round_contrast = any_round and any_non_round

        scored_candidates = []
        for c in candidates:
            score = 0
            if c["first_seen"]:
                score += 2
            if has_round_contrast and not c["is_round"]:
                score += 1
            scored_candidates.append((score, c["address"]))

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_score, top_addr = scored_candidates[0]
        second_score = scored_candidates[1][0]

        # Only mark if top candidate clearly stands out with sufficient confidence
        if top_score > second_score and top_score >= 2:
            # Mark the transaction -> wallet output edge
            if graph.has_edge(clean_txid, top_addr):
                for _, edge_data in graph[clean_txid][top_addr].items():
                    if edge_data.get("edge_type") == "output":
                        edge_data["likely_change_address"] = True
                        total_marked += 1

    logger.info("Change address heuristic applied: %d output edges marked.", total_marked)
    return graph


def get_wallet_clusters(graph: nx.MultiDiGraph) -> Dict[str, int]:
    """
    Cluster wallet nodes into co-ownership entities using common-input edges.

    Builds an undirected subgraph induced solely by 'common_input_ownership'
    edges, computes connected components, and assigns sequential cluster IDs
    (sorted descending by cluster size). Every wallet node in the graph receives
    a cluster ID (wallets without common-input links receive singleton IDs).

    Args:
        graph: Heterogeneous transaction MultiDiGraph.

    Returns:
        dict: Mapping wallet_address -> cluster_id (int).
    """
    wallet_nodes = [
        node for node, data in graph.nodes(data=True)
        if data.get("node_type") == "wallet"
    ]

    # Create an undirected graph for connected component resolution
    cio_graph = nx.Graph()
    cio_graph.add_nodes_from(wallet_nodes)

    for u, v, data in graph.edges(data=True):
        if data.get("edge_type") == "common_input_ownership":
            if u in cio_graph and v in cio_graph:
                cio_graph.add_edge(u, v)

    # Find connected components and sort by size descending
    components = sorted(nx.connected_components(cio_graph), key=len, reverse=True)

    wallet_to_cluster: Dict[str, int] = {}
    for cluster_id, comp in enumerate(components):
        for wallet in comp:
            wallet_to_cluster[wallet] = cluster_id

    logger.info(
        "Clustering complete: %d wallets partitioned into %d clusters.",
        len(wallet_nodes),
        len(components),
    )
    return wallet_to_cluster


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Apply blockchain heuristics and compute wallet clusters."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input transaction file (.csv, .json, or .xml).",
    )
    return parser


def main() -> None:
    """CLI entry point for running heuristics and clustering."""
    args = _build_parser().parse_args()

    print(f"[*] Parsing input transactions: {args.input}")
    df = parse_file(args.input)

    print(f"[*] Building graph from {len(df):,} transactions...")
    graph = build_graph(df)

    print("[*] Applying Common-Input-Ownership heuristic...")
    apply_common_input_heuristic(graph, df)

    print("[*] Applying Change Address heuristic...")
    apply_change_address_heuristic(graph, df)

    print("[*] Computing wallet clusters from connected components...")
    clusters = get_wallet_clusters(graph)

    # Compute metrics
    cio_edges = sum(
        1 for _, _, d in graph.edges(data=True)
        if d.get("edge_type") == "common_input_ownership"
    )

    change_wallets: Set[str] = set()
    for _, v, d in graph.edges(data=True):
        if d.get("edge_type") == "output" and d.get("likely_change_address") is True:
            change_wallets.add(v)

    cluster_counts = Counter(clusters.values())
    total_clusters = len(cluster_counts)
    largest_5 = cluster_counts.most_common(5)

    print("\n" + "=" * 55)
    print("           FORENSIC HEURISTICS SUMMARY")
    print("=" * 55)
    print(f"Common-Input Edges Added (Directed): {cio_edges:,}")
    print(f"Wallets Marked Likely Change:        {len(change_wallets):,}")
    print(f"Total Distinct Wallet Clusters:      {total_clusters:,}")
    print("-" * 55)
    print("Top 5 Largest Wallet Clusters:")
    for rank, (cid, size) in enumerate(largest_5, 1):
        print(f"  #{rank} Cluster ID: {cid:<4} | Wallets: {size:,}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
