"""
Wallet-level feature engineering module for Bitcoin transaction analysis.
"""
import argparse
import sys
import pandas as pd
import numpy as np
import networkx as nx

from typing import Dict, List, Any

from src.utils.logging_config import get_logger
from src.graph.builder import build_graph, get_nodes_by_type
from src.graph.heuristics import get_wallet_clusters, apply_common_input_heuristic
from src.ingestion.parser import parse_file

logger = get_logger(__name__)

def compute_wallet_features(graph: nx.MultiDiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes graph-based and transaction-based features for every wallet node in the graph.

    Args:
        graph (nx.MultiDiGraph): The networkx graph containing wallet, transaction, and ip nodes.
        df (pd.DataFrame): The transactions dataframe.

    Returns:
        pd.DataFrame: A pandas DataFrame indexed by wallet_id with computed features.
    """
    logger.info("Computing wallet features...")
    
    wallet_nodes = get_nodes_by_type(graph, 'wallet')
    
    try:
        wallet_clusters = get_wallet_clusters(graph)
        cluster_sizes = pd.Series(wallet_clusters).value_counts().to_dict()
    except Exception as e:
        logger.warning(f"Could not compute wallet clusters: {e}")
        wallet_clusters = {}
        cluster_sizes = {}

    features = []

    for wallet in wallet_nodes:
        # Predecessors of wallet are transactions sending TO this wallet
        # Successors of wallet are transactions receiving FROM this wallet
        
        in_txs = set()
        out_txs = set()
        
        total_btc_in = 0.0
        total_btc_out = 0.0
        
        in_amounts = []
        out_amounts = []
        
        timestamps = []
        ips = set()
        
        # In edges (tx -> wallet)
        for u, v, k, d in graph.in_edges(wallet, keys=True, data=True):
            if graph.nodes[u].get('node_type') == 'transaction':
                in_txs.add(u)
                amt = d.get('amount', 0.0)
                total_btc_in += amt
                in_amounts.append(amt)
                
                ts = graph.nodes[u].get('timestamp')
                if ts:
                    timestamps.append(ts)
                    
                # get ips for tx
                for _, tx_v, tx_k, tx_d in graph.out_edges(u, keys=True, data=True):
                    if tx_d.get('edge_type') == 'broadcast':
                        ips.add(tx_v)
                        
        # Out edges (wallet -> tx)
        for u, v, k, d in graph.out_edges(wallet, keys=True, data=True):
            if graph.nodes[v].get('node_type') == 'transaction':
                out_txs.add(v)
                amt = d.get('amount', 0.0)
                total_btc_out += amt
                out_amounts.append(amt)
                
                ts = graph.nodes[v].get('timestamp')
                if ts:
                    timestamps.append(ts)
                    
                # get ips for tx
                for _, tx_v, tx_k, tx_d in graph.out_edges(v, keys=True, data=True):
                    if tx_d.get('edge_type') == 'broadcast':
                        ips.add(tx_v)

        # distinct counterparty wallets sending TO this wallet
        # (senders to in_txs)
        in_counterparties = set()
        for tx in in_txs:
            for pred in graph.predecessors(tx):
                if graph.nodes[pred].get('node_type') == 'wallet':
                    in_counterparties.add(pred)
                    
        # distinct counterparty wallets receiving FROM this wallet
        # (receivers from out_txs)
        out_counterparties = set()
        for tx in out_txs:
            for succ in graph.successors(tx):
                if graph.nodes[succ].get('node_type') == 'wallet':
                    out_counterparties.add(succ)
                    
        in_degree = len(in_counterparties)
        out_degree = len(out_counterparties)
        tx_count = len(in_txs) + len(out_txs)
        net_flow = total_btc_in - total_btc_out
        
        all_amounts = in_amounts + out_amounts
        avg_tx_amount = np.mean(all_amounts) if all_amounts else 0.0
        std_tx_amount = np.std(all_amounts) if len(all_amounts) > 1 else 0.0
        
        total_degree = in_degree + out_degree
        fan_in_ratio = in_degree / total_degree if total_degree > 0 else 0.0
        fan_out_ratio = out_degree / total_degree if total_degree > 0 else 0.0
        
        distinct_ip_count = len(ips)
        countries = set()
        asns = set()
        
        for ip in ips:
            country = graph.nodes[ip].get('country')
            if country:
                countries.add(country)
            asn = graph.nodes[ip].get('asn')
            if asn:
                asns.add(asn)
                
        distinct_country_count = len(countries)
        distinct_asn_count = len(asns)
        
        cluster_id = wallet_clusters.get(wallet)
        cluster_size = cluster_sizes.get(cluster_id, 1) if cluster_id is not None else 1
        
        node_data = graph.nodes[wallet]
        is_in_peel_chain = 'peel_chain_id' in node_data
        peel_chain_depth = node_data.get('peel_chain_position', 0)
        
        if timestamps:
            min_ts = pd.to_datetime(min(timestamps))
            max_ts = pd.to_datetime(max(timestamps))
            address_lifespan_hours = (max_ts - min_ts).total_seconds() / 3600.0
        else:
            address_lifespan_hours = 0.0
            
        reuse_count = tx_count
        
        features.append({
            'wallet_id': wallet,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'tx_count': tx_count,
            'total_btc_in': total_btc_in,
            'total_btc_out': total_btc_out,
            'net_flow': net_flow,
            'avg_tx_amount': avg_tx_amount,
            'std_tx_amount': std_tx_amount,
            'fan_in_ratio': fan_in_ratio,
            'fan_out_ratio': fan_out_ratio,
            'distinct_ip_count': distinct_ip_count,
            'distinct_country_count': distinct_country_count,
            'distinct_asn_count': distinct_asn_count,
            'cluster_size': cluster_size,
            'is_in_peel_chain': is_in_peel_chain,
            'peel_chain_depth': peel_chain_depth,
            'address_lifespan_hours': address_lifespan_hours,
            'reuse_count': reuse_count
        })

    logger.info(f"Computed features for {len(features)} wallets.")
    
    if not features:
        return pd.DataFrame()
        
    features_df = pd.DataFrame(features)
    features_df.set_index('wallet_id', inplace=True)
    features_df.fillna(0, inplace=True)
    
    return features_df

def main():
    """
    Main CLI entry point for standalone testing.
    """
    parser = argparse.ArgumentParser(description="Compute wallet features.")
    parser.add_argument("file_path", type=str, help="Path to the transactions CSV file")
    args = parser.parse_args()

    try:
        logger.info(f"Loading data from {args.file_path}")
        df = parse_file(args.file_path)
        
        logger.info("Building graph...")
        graph = build_graph(df)
        
        logger.info("Applying common input heuristic...")
        apply_common_input_heuristic(graph, df)
        
        logger.info("Computing wallet features...")
        features_df = compute_wallet_features(graph, df)
        
        print("\n=== Wallet Features Summary ===")
        print(features_df.head())
        print("\nFeatures Summary Statistics:")
        print(features_df.describe())
        
    except Exception as e:
        logger.error(f"Error computing wallet features: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
