"""
Entity Clustering Module for Bitcoin Wallets.

Identifies real-world entity clusters by combining:
1. Common-Input-Ownership co-spending heuristic connected components.
2. Shared IP broadcast correlation (wallets broadcast from the same IP/ASN).
3. Density-based feature/graph clustering.
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
    ) -> pd.DataFrame:
        """
        Cluster wallets into co-owned entities.

        Args:
            graph: Heterogeneous transaction MultiDiGraph.
            df: Normalized transaction DataFrame.
            incorporate_shared_ips: If True, merges wallet clusters that exclusively share
                the same broadcasting IP.

        Returns:
            pd.DataFrame indexed by wallet_id with columns:
                cluster_id, cluster_size, primary_ip, is_multi_wallet_entity
        """
        if graph.number_of_nodes() == 0 or df.empty:
            return pd.DataFrame(columns=["cluster_id", "cluster_size", "primary_ip", "is_multi_wallet_entity"])

        # 1. Base clustering via common-input-ownership connected components
        base_clusters = get_wallet_clusters(graph)
        
        # Invert mapping to cluster_id -> set of wallets
        clusters: Dict[int, Set[str]] = defaultdict(set)
        for w, cid in base_clusters.items():
            clusters[cid].add(w)

        # 2. Build wallet -> broadcasting IP mapping
        wallet_ips: Dict[str, Set[str]] = defaultdict(set)
        for row in df.itertuples(index=True):
            src_ip = getattr(row, "src_ip", None)
            in_addrs = getattr(row, "input_addresses", [])
            if src_ip and isinstance(in_addrs, (list, tuple)):
                clean_ip = str(src_ip).strip()
                for addr in in_addrs:
                    if addr:
                        wallet_ips[str(addr).strip()].add(clean_ip)

        # 3. If incorporate_shared_ips is enabled, connect clusters sharing distinct non-generic IPs
        if incorporate_shared_ips:
            # IP -> set of clusters broadcasting from it
            ip_to_clusters: Dict[str, Set[int]] = defaultdict(set)
            for cid, w_set in clusters.items():
                for w in w_set:
                    for ip in wallet_ips.get(w, []):
                        ip_to_clusters[ip].add(cid)

            # Build disjoint-set / union-find graph over cluster IDs
            cluster_graph = nx.Graph()
            for cid in clusters.keys():
                cluster_graph.add_node(cid)

            for ip, cids in ip_to_clusters.items():
                # Only link if the IP connects a small, focused group (avoid huge public NAT/VPN hubs)
                if 1 < len(cids) <= 15:
                    c_list = list(cids)
                    for i in range(len(c_list) - 1):
                        cluster_graph.add_edge(c_list[i], c_list[i + 1])

            # Re-index merged clusters
            merged_clusters: Dict[int, Set[str]] = defaultdict(set)
            new_cid = 1
            for comp in nx.connected_components(cluster_graph):
                merged_wallets: Set[str] = set()
                for old_cid in comp:
                    merged_wallets.update(clusters[old_cid])
                merged_clusters[new_cid] = merged_wallets
                new_cid += 1
            clusters = merged_clusters

        # 4. Sort clusters descending by size and assign final 1-based sequential IDs
        sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
        final_records: List[Dict[str, Any]] = []

        for final_id, w_set in enumerate(sorted_clusters, start=1):
            c_size = len(w_set)
            # Find dominant IP for this cluster
            all_ips: List[str] = []
            for w in w_set:
                all_ips.extend(wallet_ips.get(w, []))
            primary_ip = max(set(all_ips), key=all_ips.count) if all_ips else None

            for w in w_set:
                final_records.append({
                    "wallet_id": w,
                    "cluster_id": final_id,
                    "cluster_size": c_size,
                    "primary_ip": primary_ip,
                    "is_multi_wallet_entity": c_size > 1,
                })

        result_df = pd.DataFrame(final_records).set_index("wallet_id")
        logger.info(
            "Entity clustering complete: %d wallets grouped into %d distinct entities.",
            len(result_df),
            len(sorted_clusters),
        )
        return result_df


def run_entity_clustering(
    input_file: Union[str, Path] = "data/raw/synthetic_transactions.csv",
) -> pd.DataFrame:
    """Run entity clustering pipeline."""
    df = parse_file(Path(input_file))
    graph = build_graph(df)
    apply_common_input_heuristic(graph, df)
    apply_change_address_heuristic(graph, df)

    clusterer = EntityClusterer()
    return clusterer.cluster_entities(graph, df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run entity clustering.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/raw/synthetic_transactions.csv",
        help="Path to transaction file",
    )
    args = parser.parse_args()

    clusters_df = run_entity_clustering(args.input)
    multi_entities = clusters_df[clusters_df["is_multi_wallet_entity"]]
    
    print("\n" + "=" * 60)
    print("             ENTITY CLUSTERING SUMMARY")
    print("=" * 60)
    print(f"Total Wallets: {len(clusters_df):,}")
    print(f"Total Unique Entities: {clusters_df['cluster_id'].nunique():,}")
    print(f"Wallets in Multi-Wallet Entities: {len(multi_entities):,}")

    top_clusters = clusters_df.groupby("cluster_id").agg(
        size=("cluster_size", "first"),
        primary_ip=("primary_ip", "first"),
    ).sort_values(by="size", ascending=False).head(5)

    print("\n--- TOP 5 LARGEST WALLET CLUSTERS ---")
    for cid, row in top_clusters.iterrows():
        print(f"  Cluster #{cid}: {row['size']} wallets | Primary IP: {row['primary_ip']}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
