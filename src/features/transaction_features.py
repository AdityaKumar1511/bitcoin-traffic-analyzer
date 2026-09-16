"""
Transaction-level feature engineering module.

Computes various statistical, graph-based, and temporal features for transactions
to be used in machine learning models for detecting suspicious activities.
"""

import os
import yaml
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Set
from pathlib import Path

from src.utils.logging_config import get_logger
from src.graph.builder import get_nodes_by_type, build_graph
from src.ingestion.parser import parse_file

logger = get_logger(__name__)

def load_high_risk_asns(config_path: str = "config/high_risk_asns.yaml") -> Set[str]:
    """
    Load high-risk ASNs from a YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Set[str]: A set of high-risk ASN strings.
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
                if not data or not isinstance(data, dict):
                    return set()
                asns = set()
                for key, value in data.items():
                    if key == 'high_risk_countries':
                        continue
                    if isinstance(value, list):
                        for item in value:
                            asns.add(str(item))
                return asns
    except Exception as e:
        logger.warning(f"Failed to load high-risk ASNs from {config_path}: {e}")
    return set()

def compute_transaction_features(graph: nx.MultiDiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute transaction-level features for every transaction in the dataset.

    Args:
        graph (nx.MultiDiGraph): The transaction graph containing wallet, transaction, and ip nodes.
        df (pd.DataFrame): The transactions dataframe.

    Returns:
        pd.DataFrame: A dataframe containing the computed features indexed by txid.
    """
    logger.info("Computing transaction features...")
    
    high_risk_asns = load_high_risk_asns()
    
    # Try to import get_wallet_clusters, fallback to dummy if not available
    try:
        from src.graph.heuristics import get_wallet_clusters
        wallet_clusters = get_wallet_clusters(graph)
    except ImportError:
        logger.warning("get_wallet_clusters not found in src.graph.builder, falling back to dummy clusters.")
        wallet_clusters = {}

    features_list = []
    
    # Pre-calculate ip to wallet clusters mapping for broadcasting_ip_wallet_diversity
    ip_to_clusters: Dict[str, Set[Any]] = {}
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "ip":
            clusters = set()
            # Txs broadcasting to this IP
            for u, v, k, edge_data in graph.in_edges(node, data=True, keys=True):
                if graph.nodes[u].get("node_type") == "transaction":
                    tx_node = u
                    # Input wallets to this Tx
                    for w, t, ek, in_edge_data in graph.in_edges(tx_node, data=True, keys=True):
                        if graph.nodes[w].get("node_type") == "wallet":
                            cluster_id = wallet_clusters.get(w, w)  # Fallback to wallet id if no cluster
                            clusters.add(cluster_id)
            ip_to_clusters[node] = clusters

    # Pre-calculate address counts across all txs for input_address_reuse_max
    address_counts: Dict[str, int] = {}
    for idx, row in df.iterrows():
        inputs = row.get("input_addresses", [])
        if isinstance(inputs, str):
            inputs = [addr.strip() for addr in inputs.split(";") if addr.strip()]
        for addr in inputs:
            address_counts[addr] = address_counts.get(addr, 0) + 1
            
    for idx, row in df.iterrows():
        txid = row.get("txid")
        
        # Parse inputs
        inputs = row.get("input_addresses", [])
        if isinstance(inputs, str):
            inputs = [addr.strip() for addr in inputs.split(";") if addr.strip()]
            
        # Parse outputs
        outputs = row.get("output_addresses", [])
        if isinstance(outputs, str):
            outputs = [addr.strip() for addr in outputs.split(";") if addr.strip()]
            
        # Parse amounts
        input_amounts = row.get("input_amounts", [])
        if isinstance(input_amounts, str):
            input_amounts = [float(a) for a in input_amounts.split(";") if a.strip()]
            
        output_amounts = row.get("output_amounts", [])
        if isinstance(output_amounts, str):
            output_amounts = [float(a) for a in output_amounts.split(";") if a.strip()]
            
        num_inputs = len(inputs)
        num_outputs = len(outputs)
        total_input_value = sum(input_amounts) if input_amounts else 0.0
        total_output_value = sum(output_amounts) if output_amounts else 0.0
        
        fee = float(row.get("fee", 0.0))
        if pd.isna(fee):
            fee = 0.0
            
        fee_ratio = fee / total_input_value if total_input_value > 0 else 0.0
        
        avg_input_amount = np.mean(input_amounts) if input_amounts else 0.0
        avg_output_amount = np.mean(output_amounts) if output_amounts else 0.0
        max_output_amount = np.max(output_amounts) if output_amounts else 0.0
        min_output_amount = np.min(output_amounts) if output_amounts else 0.0
        output_amount_std = np.std(output_amounts) if len(output_amounts) > 1 else 0.0
        output_amount_range = max_output_amount - min_output_amount
        
        input_output_ratio = num_inputs / num_outputs if num_outputs > 0 else 0.0
        
        script_type = str(row.get("script_type", "")).upper()
        is_script_p2pkh = 1 if "P2PKH" in script_type else 0
        is_script_p2sh = 1 if "P2SH" in script_type else 0
        is_script_p2wpkh = 1 if "P2WPKH" in script_type else 0
        is_script_multisig = 1 if "MULTISIG" in script_type else 0
        
        # Graph-derived features
        broadcast_ip_is_high_risk_asn = 0
        has_change_output = 0
        broadcasting_ip_wallet_diversity = 0
        
        if graph.has_node(txid):
            for u, v, k, data in graph.out_edges(txid, data=True, keys=True):
                target_node = graph.nodes[v]
                if target_node.get("node_type") == "ip":
                    asn = str(target_node.get("asn", ""))
                    if asn in high_risk_asns:
                        broadcast_ip_is_high_risk_asn = 1
                    
                    broadcasting_ip_wallet_diversity = len(ip_to_clusters.get(v, set()))
                elif target_node.get("node_type") == "wallet":
                    if data.get("likely_change_address") == True:
                        has_change_output = 1

        # Temporal features
        timestamp = pd.to_datetime(row.get("timestamp"))
        if pd.notna(timestamp):
            broadcast_hour = timestamp.hour
            broadcast_day_of_week = timestamp.dayofweek
        else:
            broadcast_hour = 0
            broadcast_day_of_week = 0

        # is_round_total
        # Check if total_output_value has <=2 decimal places
        # Format to 8 decimals to avoid float precision issues, then check
        rounded_2 = round(total_output_value, 2)
        is_round_total = 1 if abs(total_output_value - rounded_2) < 1e-8 else 0
        
        # input_address_reuse_max
        input_address_reuse_max = max((address_counts.get(addr, 0) for addr in inputs), default=0)
        
        features_list.append({
            "txid": txid,
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "total_input_value": total_input_value,
            "total_output_value": total_output_value,
            "fee": fee,
            "fee_ratio": fee_ratio,
            "avg_input_amount": avg_input_amount,
            "avg_output_amount": avg_output_amount,
            "max_output_amount": max_output_amount,
            "min_output_amount": min_output_amount,
            "output_amount_std": output_amount_std,
            "output_amount_range": output_amount_range,
            "input_output_ratio": input_output_ratio,
            "is_script_p2pkh": is_script_p2pkh,
            "is_script_p2sh": is_script_p2sh,
            "is_script_p2wpkh": is_script_p2wpkh,
            "is_script_multisig": is_script_multisig,
            "broadcast_ip_is_high_risk_asn": broadcast_ip_is_high_risk_asn,
            "broadcast_hour": broadcast_hour,
            "broadcast_day_of_week": broadcast_day_of_week,
            "has_change_output": has_change_output,
            "is_round_total": is_round_total,
            "broadcasting_ip_wallet_diversity": broadcasting_ip_wallet_diversity,
            "input_address_reuse_max": input_address_reuse_max
        })
        
    features_df = pd.DataFrame(features_list)
    if not features_df.empty:
        features_df = features_df.set_index("txid")
        
    features_df = features_df.fillna(0.0)
    logger.info(f"Computed features for {len(features_df)} transactions.")
    return features_df

def main() -> None:
    """CLI entry point for standalone testing."""
    parser = argparse.ArgumentParser(description="Compute transaction features.")
    parser.add_argument("--data", type=str, required=True, help="Path to the transactions CSV file")
    args = parser.parse_args()

    df = parse_file(args.data)
    graph = build_graph(df)
    features_df = compute_transaction_features(graph, df)
    
    print(features_df.head())
    print(f"Total shape: {features_df.shape}")

if __name__ == "__main__":
    main()
