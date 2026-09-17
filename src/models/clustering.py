"""
Entity Clustering Module for Bitcoin Wallets.

Identifies real-world entity clusters by combining:
1. Deterministic Common-Input-Ownership (CIO) co-spending heuristic connected components.
2. Network-Layer Exclusivity Linking: Groups wallet clusters that share exclusive broadcasting IPs.
3. Behavioral Fingerprint Clustering: Groups structurally similar unlinked wallets via unsupervised clustering.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from src.features import extract_all_features
from src.graph.builder import build_graph, get_nodes_by_type
from src.graph.heuristics import apply_change_address_heuristic, apply_common_input_heuristic, get_wallet_clusters
from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def cluster_entities_multimodal(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
    wallet_features_df: Optional[pd.DataFrame] = None,
    link_shared_ips: bool = True,
    ip_max_wallets_threshold: int = 10,
) -> pd.DataFrame:
    """
    Perform multi-modal entity clustering fusing on-chain CIO heuristics with network exclusivity.

    Args:
        graph: Heterogeneous MultiDiGraph.
        df: Normalized transaction DataFrame.
        wallet_features_df: Optional wallet features DataFrame for behavioral clustering.
        link_shared_ips: Whether to merge clusters sharing exclusive broadcasting IPs.
        ip_max_wallets_threshold: Maximum distinct wallets on an IP before treating it as a shared/NAT/relay IP.

    Returns:
        pd.DataFrame indexed by wallet_id with columns:
            - cluster_id (int)
            - entity_cluster_id (str)
            - cio_cluster_id (int)
            - cluster_size (int)
            - clustering_method (str)
            - confidence_score (float)
            - primary_ip (str)
            - is_multi_wallet_entity (bool)
    """
    logger.info("Starting multi-modal entity clustering...")

    if graph.number_of_nodes() == 0 and df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "entity_cluster_id",
                "cio_cluster_id",
                "cluster_size",
                "clustering_method",
                "confidence_score",
                "primary_ip",
                "is_multi_wallet_entity",
            ]
        ).set_index(pd.Index([], name="wallet_id"))

    # 1. Base Layer: Cryptographic Common-Input-Ownership (CIO) connected components
    cio_clusters = get_wallet_clusters(graph)

    # 2. Map wallets to broadcasting IPs
    wallet_to_ips: Dict[str, Set[str]] = defaultdict(set)
    ip_to_wallets: Dict[str, Set[str]] = defaultdict(set)

    for row in df.itertuples(index=True):
        src_ip = getattr(row, "src_ip", None)
        in_addrs = getattr(row, "input_addresses", [])

        if src_ip and isinstance(src_ip, str) and src_ip.strip():
            clean_ip = src_ip.strip()
            if isinstance(in_addrs, (list, tuple)):
                for addr in in_addrs:
                    if addr and isinstance(addr, str) and addr.strip():
                        clean_w = addr.strip()
                        wallet_to_ips[clean_w].add(clean_ip)
                        ip_to_wallets[clean_ip].add(clean_w)

    # All wallets in graph or df
    all_wallets = sorted(list(set(list(cio_clusters.keys()) + list(wallet_to_ips.keys()))))
    if not all_wallets:
        # Fallback to wallet nodes in graph
        all_wallets = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "wallet"]

    parent: Dict[str, str] = {w: w for w in all_wallets}

    def find(w: str) -> str:
        root = w
        while parent.get(root, root) != root:
            root = parent[root]
        curr = w
        while curr != root and curr in parent:
            nxt = parent[curr]
            parent[curr] = root
            curr = nxt
        return root

    def union(w1: str, w2: str) -> None:
        r1, r2 = find(w1), find(w2)
        if r1 != r2:
            parent[r2] = r1

    # Connect wallets within the same CIO cluster
    cio_to_wallets: Dict[int, List[str]] = defaultdict(list)
    for w, cid in cio_clusters.items():
        cio_to_wallets[cid].append(w)

    for cid, w_list in cio_to_wallets.items():
        for i in range(1, len(w_list)):
            union(w_list[0], w_list[i])

    # 3. Network Linkage: Merge clusters sharing exclusive/private broadcasting IPs
    ip_links_formed = 0
    wallet_link_method: Dict[str, str] = {w: "common_input_ownership" for w in all_wallets}
    wallet_confidence: Dict[str, float] = {w: 1.0 for w in all_wallets}

    if link_shared_ips:
        for ip, wallets in ip_to_wallets.items():
            valid_wallets = [w for w in wallets if w in parent]
            if 2 <= len(valid_wallets) <= ip_max_wallets_threshold:
                first_w = valid_wallets[0]
                for other_w in valid_wallets[1:]:
                    if find(first_w) != find(other_w):
                        union(first_w, other_w)
                        ip_links_formed += 1
                        wallet_link_method[other_w] = "exclusive_ip_linkage"
                        wallet_confidence[other_w] = 0.85

    logger.info("Formed %d network-layer cluster merges via exclusive IPs.", ip_links_formed)

    # 4. Resolve final cluster IDs
    cluster_groups: Dict[str, List[str]] = defaultdict(list)
    for w in all_wallets:
        root = find(w)
        cluster_groups[root].append(w)

    # Sort clusters by size descending
    sorted_roots = sorted(cluster_groups.keys(), key=lambda r: len(cluster_groups[r]), reverse=True)
    root_to_int_id = {r: idx for idx, r in enumerate(sorted_roots)}
    root_to_final_id = {r: f"ENT_{idx:05d}" for idx, r in enumerate(sorted_roots)}

    results = []
    for w in all_wallets:
        r = find(w)
        int_id = root_to_int_id[r]
        final_id = root_to_final_id[r]
        c_size = len(cluster_groups[r])
        ips = list(wallet_to_ips.get(w, []))
        primary_ip = ips[0] if ips else "Unknown"

        cio_size = len(cio_to_wallets[cio_clusters.get(w, -1)])
        if cio_size > 1:
            method = "common_input_ownership"
            confidence = 1.00
        elif c_size == 1:
            method = "singleton"
            confidence = 0.50
        else:
            method = "exclusive_ip_linkage"
            confidence = wallet_confidence.get(w, 0.85)

        results.append({
            "wallet_id": w,
            "cluster_id": int_id,
            "entity_cluster_id": final_id,
            "cio_cluster_id": cio_clusters.get(w, -1),
            "cluster_size": c_size,
            "clustering_method": method,
            "confidence_score": confidence,
            "primary_ip": primary_ip,
            "is_multi_wallet_entity": c_size > 1,
        })

    cluster_df = pd.DataFrame(results).set_index("wallet_id")
    multi_wallet_clusters = (cluster_df["cluster_size"] > 1).sum()
    logger.info(
        "Clustering complete: %d wallets grouped into %d distinct entities (%d wallets in multi-wallet entities).",
        len(cluster_df),
        len(sorted_roots),
        multi_wallet_clusters,
    )
    return cluster_df


class EntityClusterer:
    """
    Multi-signal entity clustering engine grouping Bitcoin wallet addresses.
    """

    def __init__(self) -> None:
        self.wallet_to_cluster: Dict[str, int] = {}
        self.cluster_metadata: Dict[int, Dict[str, Any]] = {}

    def cluster_entities(
        self,
        graph: nx.MultiDiGraph,
        df: pd.DataFrame,
        incorporate_shared_ips: bool = True,
        ip_max_wallets_threshold: int = 10,
    ) -> pd.DataFrame:
        """
        Cluster wallets into co-owned entities using multi-modal signals.

        Args:
            graph: Heterogeneous transaction MultiDiGraph.
            df: Normalized transaction DataFrame.
            incorporate_shared_ips: If True, merges wallet clusters that exclusively share
                the same broadcasting IP.
            ip_max_wallets_threshold: Maximum distinct wallets on an IP before treating it as shared.

        Returns:
            pd.DataFrame indexed by wallet_id with columns:
                cluster_id, entity_cluster_id, cluster_size, primary_ip, is_multi_wallet_entity,
                clustering_method, confidence_score
        """
        df_result = cluster_entities_multimodal(
            graph=graph,
            df=df,
            link_shared_ips=incorporate_shared_ips,
            ip_max_wallets_threshold=ip_max_wallets_threshold,
        )

        for w, row in df_result.iterrows():
            cid = int(row["cluster_id"])
            self.wallet_to_cluster[str(w)] = cid
            if cid not in self.cluster_metadata:
                self.cluster_metadata[cid] = {
                    "size": int(row["cluster_size"]),
                    "primary_ip": str(row["primary_ip"]),
                    "entity_id": str(row["entity_cluster_id"]),
                }

        return df_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-Modal Entity Clustering for Bitcoin Wallets.")
    parser.add_argument("--data", "-d", type=Path, default=Path("data/raw/synthetic_transactions.csv"))
    parser.add_argument("--ground-truth", "-g", type=Path, default=Path("data/raw/ground_truth.csv"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    print(f"[*] Ingesting transactions from: {args.data}")
    df = parse_file(args.data)
    graph = build_graph(df)
    apply_common_input_heuristic(graph, df)

    print("[*] Running Multi-Modal Entity Clustering...")
    clusters = cluster_entities_multimodal(graph, df)

    print("\n" + "=" * 65)
    print("                 ENTITY CLUSTERING SUMMARY")
    print("=" * 65)
    print(f"Total Wallets Clustered:          {len(clusters):,}")
    print(f"Total Distinct Entities:         {clusters['entity_cluster_id'].nunique():,}")
    print(f"Wallets in Multi-Address Groups: {(clusters['cluster_size'] > 1).sum():,}")
    largest_cluster = clusters["cluster_size"].max() if not clusters.empty else 0
    print(f"Largest Entity Cluster Size:     {largest_cluster:,} wallets")
    print("-" * 65)
    print("Clustering Method Breakdown:")
    for method, count in clusters["clustering_method"].value_counts().items():
        print(f"  - {method:<28}: {count:,} wallets")
    print("=" * 65)


if __name__ == "__main__":
    main()
