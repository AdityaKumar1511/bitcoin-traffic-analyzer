"""
Temporal pattern feature engineering for Bitcoin wallet behavior analysis.

This module computes temporal and behavioral pattern features for Bitcoin wallets based on their
transaction history. Features include entropy of transaction timing, burstiness, regularity,
and general activity metrics.
"""

import math
from typing import List
from collections import Counter
import pandas as pd
import numpy as np
import networkx as nx
import argparse

from src.utils.logging_config import get_logger
from src.graph.builder import get_nodes_by_type, build_graph
from src.ingestion.parser import parse_file

logger = get_logger(__name__)


def _shannon_entropy(counts: List[int]) -> float:
    """
    Calculate the Shannon entropy of a distribution.
    
    Args:
        counts: List of frequency counts.
        
    Returns:
        Shannon entropy in bits (log2).
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def _get_wallet_timestamps(graph: nx.MultiDiGraph, wallet_id: str) -> List[pd.Timestamp]:
    """
    Extract and sort all transaction timestamps for a given wallet.
    
    Args:
        graph: The transaction graph.
        wallet_id: The ID of the wallet node.
        
    Returns:
        Sorted list of pandas Timestamps.
    """
    tx_nodes = set()
    
    # Check successors for output edges (wallet -> tx)
    for neighbor in graph.successors(wallet_id):
        if graph.nodes[neighbor].get('node_type') == 'transaction':
            tx_nodes.add(neighbor)
            
    # Check predecessors for input edges (tx -> wallet)
    for neighbor in graph.predecessors(wallet_id):
        if graph.nodes[neighbor].get('node_type') == 'transaction':
            tx_nodes.add(neighbor)
            
    timestamps = []
    for tx in tx_nodes:
        ts = graph.nodes[tx].get('timestamp')
        if ts is not None:
            timestamps.append(pd.to_datetime(ts))
            
    return sorted(timestamps)


def compute_temporal_features(graph: nx.MultiDiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal/behavioral pattern features for every wallet in the graph.
    
    Args:
        graph: The transaction graph containing wallet and transaction nodes.
        df: The raw transaction dataframe.
        
    Returns:
        A pandas DataFrame indexed by wallet_id with computed temporal features.
    """
    logger.info("Computing temporal features for wallets")
    
    wallet_nodes = get_nodes_by_type(graph, 'wallet')
    logger.info(f"Found {len(wallet_nodes)} wallets for feature extraction")
    
    features_list = []
    max_hour_entropy = math.log2(24)
    
    for wallet_id in wallet_nodes:
        timestamps = _get_wallet_timestamps(graph, wallet_id)
        
        # Initialize default values
        feats = {
            'wallet_id': wallet_id,
            'hour_entropy': 0.0,
            'day_of_week_entropy': 0.0,
            'is_likely_automated': False,
            'burstiness': 0.0,
            'mean_inter_tx_seconds': 0.0,
            'min_inter_tx_seconds': 0.0,
            'max_inter_tx_seconds': 0.0,
            'tx_velocity_per_hour': 0.0,
            'tx_velocity_per_day': 0.0,
            'peak_hour': 0,
            'peak_day': 0,
            'active_days': 0,
            'weekend_tx_ratio': 0.0,
            'night_tx_ratio': 0.0,
            'longest_gap_hours': 0.0,
            'activity_regularity': 0.0
        }
        
        n_tx = len(timestamps)
        
        if n_tx > 0:
            # Extract time components
            hours = [ts.hour for ts in timestamps]
            days = [ts.dayofweek for ts in timestamps]
            dates = [ts.date() for ts in timestamps]
            
            # Entropies
            hour_counts = list(Counter(hours).values())
            day_counts = list(Counter(days).values())
            
            feats['hour_entropy'] = _shannon_entropy(hour_counts)
            feats['day_of_week_entropy'] = _shannon_entropy(day_counts)
            feats['is_likely_automated'] = feats['hour_entropy'] > 0.9 * max_hour_entropy
            
            # Peak components
            feats['peak_hour'] = Counter(hours).most_common(1)[0][0]
            feats['peak_day'] = Counter(days).most_common(1)[0][0]
            
            # Ratios
            weekend_txs = sum(1 for d in days if d >= 5)
            feats['weekend_tx_ratio'] = weekend_txs / n_tx
            
            night_txs = sum(1 for h in hours if 0 <= h < 6)
            feats['night_tx_ratio'] = night_txs / n_tx
            
            feats['active_days'] = len(set(dates))
            
            # Lifespan and velocities
            lifespan_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
            if lifespan_seconds > 0:
                feats['tx_velocity_per_hour'] = n_tx / (lifespan_seconds / 3600)
                feats['tx_velocity_per_day'] = n_tx / (lifespan_seconds / 86400)
            else:
                feats['tx_velocity_per_hour'] = float(n_tx)
                feats['tx_velocity_per_day'] = float(n_tx)
                
            # Inter-transaction features
            if n_tx >= 2:
                diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, n_tx)]
                diffs_array = np.array(diffs)
                
                feats['mean_inter_tx_seconds'] = float(np.mean(diffs_array))
                feats['min_inter_tx_seconds'] = float(np.min(diffs_array))
                feats['max_inter_tx_seconds'] = float(np.max(diffs_array))
                feats['longest_gap_hours'] = feats['max_inter_tx_seconds'] / 3600
                
                mean_diff = feats['mean_inter_tx_seconds']
                std_diff = float(np.std(diffs_array))
                
                if mean_diff > 0:
                    feats['burstiness'] = std_diff / mean_diff
                    feats['activity_regularity'] = max(0.0, min(1.0, 1.0 - (std_diff / mean_diff)))
                else:
                    feats['burstiness'] = 0.0
                    feats['activity_regularity'] = 1.0 
                    
        features_list.append(feats)
        
    result_df = pd.DataFrame(features_list)
    result_df.set_index('wallet_id', inplace=True)
    return result_df


def main():
    """CLI entry point for temporal feature extraction."""
    parser = argparse.ArgumentParser(description="Extract temporal features from Bitcoin transactions")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input transaction CSV")
    args = parser.parse_args()
    
    logger.info(f"Parsing input file: {args.input}")
    df = parse_file(args.input)
    
    logger.info("Building graph from transactions")
    graph = build_graph(df)
    
    logger.info("Computing temporal features")
    features_df = compute_temporal_features(graph, df)
    
    logger.info("Feature extraction complete.")
    
    bot_like = features_df[features_df['is_likely_automated']]
    top_bots = bot_like.sort_values(by='hour_entropy', ascending=False).head(10)
    
    print("\n=== Top 10 Most Bot-Like Wallets (by Hour Entropy) ===")
    if not top_bots.empty:
        print(top_bots[['hour_entropy', 'day_of_week_entropy', 'tx_velocity_per_day', 'activity_regularity']])
    else:
        print("No likely automated wallets found based on threshold.")


if __name__ == "__main__":
    main()
