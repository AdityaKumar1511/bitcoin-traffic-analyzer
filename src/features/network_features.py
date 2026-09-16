"""
Network-layer feature engineering for Bitcoin transactions.
This module is the NTRO-specific differentiator. It fuses network-layer IP and timing data
with on-chain blockchain analysis to provide unique insights into transaction broadcasting
and potential deanonymization.
"""

import math
import yaml
from pathlib import Path
from typing import Set, List, Dict, Any
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime

from src.graph.builder import get_nodes_by_type, build_graph
from src.graph.heuristics import get_wallet_clusters, apply_common_input_heuristic
from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger
from src.utils.paths import get_project_root

logger = get_logger(__name__)

def _load_high_risk_countries() -> Set[str]:
    """Load high-risk country codes from config."""
    config_path = get_project_root() / "config" / "high_risk_asns.yaml"
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'high_risk_countries' in data:
                    return set(str(c) for c in data['high_risk_countries'])
    except Exception:
        pass
    return {'RU', 'KP', 'IR', 'CN', 'BY', 'SY', 'MM', 'VE'}  # fallback

def _load_high_risk_asns() -> Set[int]:
    """
    Load high-risk ASNs from config.
    
    Returns:
        Set[int]: Set of high-risk ASN numbers.
    """
    config_path = get_project_root() / "config" / "high_risk_asns.yaml"
    if not config_path.exists():
        logger.warning(f"High-risk ASNs config not found at {config_path}. Using empty set.")
        return set()
        
    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            if not data or not isinstance(data, dict):
                return set()
            asns = set()
            # Collect ASNs from all category lists (bulletproof_hosting, vpn_proxy, etc.)
            for key, value in data.items():
                if key == 'high_risk_countries':
                    continue  # Skip country list
                if isinstance(value, list):
                    for item in value:
                        try:
                            asns.add(int(item))
                        except (ValueError, TypeError):
                            pass
            return asns
    except Exception as e:
        logger.error(f"Error loading high-risk ASNs config: {e}")
        
    return set()

def _shannon_entropy(counts: List[int]) -> float:
    """
    Compute Shannon entropy for a distribution of counts.
    
    Args:
        counts (List[int]): Frequency counts.
        
    Returns:
        float: Shannon entropy.
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

def compute_network_features(graph: nx.MultiDiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute network-layer features for every IP address node in the graph.
    
    Args:
        graph (nx.MultiDiGraph): The transaction/network graph.
        df (pd.DataFrame): Raw transaction DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame containing features indexed by ip_address.
    """
    logger.info("Computing network features for IP nodes...")
    
    high_risk_asns = _load_high_risk_asns()
    high_risk_countries = _load_high_risk_countries()
    ip_nodes = get_nodes_by_type(graph, 'ip')
    
    # Precompute wallet clusters for quick lookup
    wallet_clusters = get_wallet_clusters(graph)
    
    features = []
    
    for ip in ip_nodes:
        ip_data = graph.nodes[ip]
        country = ip_data.get('country', 'Unknown')
        asn = int(ip_data.get('asn', 0))
        is_high_risk_asn = asn in high_risk_asns
        
        # Get all transactions broadcast by this IP
        # Edges are tx -> ip with type 'broadcast'
        tx_nodes = []
        timestamps = []
        src_ports = []
        
        for u, v, key, data in graph.in_edges(ip, data=True, keys=True):
            if data.get('edge_type') == 'broadcast':
                tx_nodes.append(u)
                if 'timestamp' in data:
                    timestamps.append(data['timestamp'])
                if 'src_port' in data:
                    src_ports.append(data['src_port'])
                    
        ip_tx_count = len(tx_nodes)
        
        # Determine wallets that created these transactions
        # Edges are wallet -> tx with type 'input'
        wallets = set()
        for tx in tx_nodes:
            for w, _, key, data in graph.in_edges(tx, data=True, keys=True):
                if data.get('edge_type') == 'input':
                    wallets.add(w)
                    
        ip_wallet_diversity = len(wallets)
        
        clusters = {wallet_clusters.get(w, w) for w in wallets}
        ip_cluster_diversity = len(clusters)
        
        # Timing features
        # Convert timestamps to pd.Timestamp for consistency and sort
        parsed_timestamps = []
        for ts in timestamps:
            try:
                parsed_timestamps.append(pd.to_datetime(ts))
            except Exception:
                pass
        sorted_timestamps = sorted(parsed_timestamps)
        
        time_deltas_seconds = []
        for i in range(1, len(sorted_timestamps)):
            delta = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds()
            time_deltas_seconds.append(delta)
            
        if time_deltas_seconds:
            timing_tightness_mean = float(np.mean(time_deltas_seconds))
            timing_tightness_std = float(np.std(time_deltas_seconds))
            timing_tightness_min = float(np.min(time_deltas_seconds))
        else:
            timing_tightness_mean = 0.0
            timing_tightness_std = 0.0
            timing_tightness_min = 0.0
            
        burst_count = sum(1 for d in time_deltas_seconds if d <= 60)
        
        # Port features
        distinct_ports = set(src_ports)
        port_diversity = len(distinct_ports)
        uses_non_standard_port = any(p != 8333 for p in distinct_ports)
        
        # Active hours features
        hours_distribution = [0] * 24
        for ts in sorted_timestamps:
            try:
                hours_distribution[ts.hour] += 1
            except Exception:
                pass
                
        active_hours_count = sum(1 for h in hours_distribution if h > 0)
        active_hours_entropy = _shannon_entropy(hours_distribution)
        
        # Risk score
        geo_risk_score = 0.0
        if is_high_risk_asn:
            geo_risk_score = 1.0
        elif country in high_risk_countries:
            geo_risk_score = 0.5
            
        features.append({
            'ip_address': ip,
            'ip_tx_count': ip_tx_count,
            'ip_wallet_diversity': ip_wallet_diversity,
            'ip_cluster_diversity': ip_cluster_diversity,
            'ip_country': country,
            'ip_asn': asn,
            'is_high_risk_asn': is_high_risk_asn,
            'timing_tightness_mean': timing_tightness_mean,
            'timing_tightness_std': timing_tightness_std,
            'timing_tightness_min': timing_tightness_min,
            'burst_count': burst_count,
            'port_diversity': port_diversity,
            'uses_non_standard_port': uses_non_standard_port,
            'active_hours_count': active_hours_count,
            'active_hours_entropy': active_hours_entropy,
            'geo_risk_score': geo_risk_score
        })
        
    res_df = pd.DataFrame(features)
    if not res_df.empty:
        res_df.set_index('ip_address', inplace=True)
    logger.info(f"Computed network features for {len(res_df)} IPs.")
    return res_df

def compute_wallet_network_features(graph: nx.MultiDiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """
    Map network features back to wallets.
    
    Args:
        graph (nx.MultiDiGraph): The transaction/network graph.
        df (pd.DataFrame): Raw transaction DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame containing wallet network features indexed by wallet_id.
    """
    logger.info("Computing wallet network features...")
    ip_features_df = compute_network_features(graph, df)
    
    wallet_nodes = get_nodes_by_type(graph, 'wallet')
    
    wallet_features = []
    
    for wallet in wallet_nodes:
        # Find all IPs that broadcast this wallet's transactions
        # wallet -> tx ('input'), tx -> ip ('broadcast')
        associated_ips = set()
        
        for _, tx, key, data in graph.out_edges(wallet, data=True, keys=True):
            if data.get('edge_type') == 'input':
                # find ip this tx was broadcast to
                for _, ip, ip_key, ip_data in graph.out_edges(tx, data=True, keys=True):
                    if ip_data.get('edge_type') == 'broadcast':
                        associated_ips.add(ip)
                        
        if not associated_ips:
            continue
            
        # Get features for these IPs from the ip_features_df
        # Only query IPs that exist in the dataframe index
        valid_ips = [ip for ip in associated_ips if ip in ip_features_df.index]
        if not valid_ips:
            continue
            
        ip_data = ip_features_df.loc[valid_ips]
        
        wallet_distinct_ips = len(valid_ips)
        wallet_ip_country_diversity = ip_data['ip_country'].nunique()
        wallet_ip_asn_diversity = ip_data['ip_asn'].nunique()
        wallet_max_ip_cluster_diversity = ip_data['ip_cluster_diversity'].max()
        wallet_has_high_risk_ip = ip_data['is_high_risk_asn'].any()
        
        wallet_avg_timing_tightness = ip_data['timing_tightness_mean'].mean()
        wallet_min_burst_interval = ip_data['timing_tightness_min'].min()
        
        wallet_features.append({
            'wallet_id': wallet,
            'wallet_distinct_ips': wallet_distinct_ips,
            'wallet_ip_country_diversity': wallet_ip_country_diversity,
            'wallet_ip_asn_diversity': wallet_ip_asn_diversity,
            'wallet_max_ip_cluster_diversity': wallet_max_ip_cluster_diversity,
            'wallet_has_high_risk_ip': bool(wallet_has_high_risk_ip),
            'wallet_avg_timing_tightness': wallet_avg_timing_tightness,
            'wallet_min_burst_interval': wallet_min_burst_interval
        })
        
    res_df = pd.DataFrame(wallet_features)
    if not res_df.empty:
        res_df.set_index('wallet_id', inplace=True)
        
    logger.info(f"Computed network features for {len(res_df)} wallets.")
    return res_df

def main() -> None:
    """
    CLI entry point for standalone testing.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Compute network features for Bitcoin transactions")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV data")
    args = parser.parse_args()
    
    logger.info(f"Loading data from {args.input}")
    df = parse_file(args.input)
    graph = build_graph(df)
    apply_common_input_heuristic(graph, df)
    
    ip_df = compute_network_features(graph, df)
    wallet_df = compute_wallet_network_features(graph, df)
    
    print("IP Network Features:")
    print(ip_df.head())
    
    print("\nWallet Network Features:")
    print(wallet_df.head())

if __name__ == "__main__":
    main()
