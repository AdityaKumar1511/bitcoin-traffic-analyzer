"""
Features package for Bitcoin transaction network analysis.

Exports:
    - compute_wallet_features
    - compute_transaction_features
    - compute_network_features
    - compute_wallet_network_features
    - compute_temporal_features
    - extract_all_features (orchestrates and merges all feature matrices)
"""

from typing import Dict
import pandas as pd
import networkx as nx

from src.features.wallet_features import compute_wallet_features
from src.features.transaction_features import compute_transaction_features
from src.features.network_features import (
    compute_network_features,
    compute_wallet_network_features,
)
from src.features.temporal_features import compute_temporal_features
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_all_features(graph: nx.MultiDiGraph, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Extract and assemble unified feature matrices for wallets, transactions, and IPs.

    Args:
        graph: Heterogeneous transaction MultiDiGraph.
        df: Normalized transaction DataFrame.

    Returns:
        dict: {
            "wallets": Combined DataFrame of graph, network, and temporal features indexed by wallet_id.
            "transactions": DataFrame of transaction-level features indexed by txid.
            "ips": DataFrame of network features indexed by ip_address.
        }
    """
    logger.info("Extracting all feature matrices...")

    # 1. Transaction-level features
    tx_feats = compute_transaction_features(graph, df)

    # 2. IP network features
    ip_feats = compute_network_features(graph, df)

    # 3. Wallet graph features
    wallet_graph_feats = compute_wallet_features(graph, df)

    # 4. Wallet network features (mapped from broadcasting IPs)
    wallet_net_feats = compute_wallet_network_features(graph, df)

    # 5. Wallet temporal features
    wallet_temp_feats = compute_temporal_features(graph, df)

    # Combine wallet-level features
    # Start with wallet graph features as primary index
    combined_wallets = wallet_graph_feats.join(wallet_net_feats, how="left")
    combined_wallets = combined_wallets.join(wallet_temp_feats, how="left")
    combined_wallets = combined_wallets.fillna(0.0)

    # Ensure boolean flags are treated consistently
    bool_cols = [c for c in combined_wallets.columns if c.startswith("is_") or c.startswith("has_")]
    for col in bool_cols:
        combined_wallets[col] = combined_wallets[col].astype(int)

    logger.info(
        "Feature extraction complete: %d wallets (%d features), %d transactions (%d features), %d IPs (%d features).",
        len(combined_wallets),
        len(combined_wallets.columns),
        len(tx_feats),
        len(tx_feats.columns),
        len(ip_feats),
        len(ip_feats.columns),
    )

    return {
        "wallets": combined_wallets,
        "transactions": tx_feats,
        "ips": ip_feats,
    }


__all__ = [
    "compute_wallet_features",
    "compute_transaction_features",
    "compute_network_features",
    "compute_wallet_network_features",
    "compute_temporal_features",
    "extract_all_features",
]
