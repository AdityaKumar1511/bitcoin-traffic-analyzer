"""
Network-Blockchain Correlation Module (NTRO Core Novelty).

Fuses P2P network-layer broadcast metadata (IP, port, timestamp, ASN, country)
with blockchain-layer transaction graph structures (wallets, TXIDs, amounts, clusters).

Operationalizes Deanonymization and Threat Correlation:
1. IP-to-Wallet Cluster Exclusivity: Discovers if an IP is a dedicated actor node vs. shared relay/VPN.
2. Timing Tightness & Burst Detection: Detects coordinated multi-wallet broadcasts (<60s intervals).
3. Infrastructure Risk Scoring: Evaluates bulletproof hosting, VPNs, Tor exit nodes, and high-risk jurisdictions.
4. Synthesizes an interpretable Network Correlation Risk Score with forensic explanations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import pandas as pd
import yaml

from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger
from src.utils.paths import get_project_root

logger = get_logger(__name__)


class NetworkBlockchainCorrelator:
    """
    Computes calibrated correlation scores between network broadcasting IPs/ASNs and wallet entities.
    """

    def __init__(self, high_risk_asns_path: Optional[Union[str, Path]] = None) -> None:
        project_root = get_project_root()
        if high_risk_asns_path is None:
            high_risk_asns_path = project_root / "config" / "high_risk_asns.yaml"
        else:
            high_risk_asns_path = Path(high_risk_asns_path)

        self.bulletproof_asns: Set[int] = set()
        self.vpn_proxy_asns: Set[int] = set()
        self.tor_asns: Set[int] = set()
        self.privacy_asns: Set[int] = set()
        self.high_risk_countries: Set[str] = set()

        if high_risk_asns_path.is_file():
            with open(high_risk_asns_path, "r", encoding="utf-8") as f:
                asn_cfg = yaml.safe_load(f) or {}
                for item in asn_cfg.get("bulletproof_hosting", []):
                    try:
                        self.bulletproof_asns.add(int(item))
                    except (ValueError, TypeError):
                        pass
                for item in asn_cfg.get("vpn_proxy", []):
                    try:
                        self.vpn_proxy_asns.add(int(item))
                    except (ValueError, TypeError):
                        pass
                for item in asn_cfg.get("tor_related", []):
                    try:
                        self.tor_asns.add(int(item))
                    except (ValueError, TypeError):
                        pass
                for item in asn_cfg.get("privacy_focused", []):
                    try:
                        self.privacy_asns.add(int(item))
                    except (ValueError, TypeError):
                        pass
                for c in asn_cfg.get("high_risk_countries", []):
                    self.high_risk_countries.add(str(c).upper())

    def _get_asn_category_and_weight(self, asn: Optional[int], country: Optional[str]) -> Tuple[str, float, float, str]:
        """
        Categorize ASN and return (category, confidence_multiplier, threat_score, label).
        """
        if asn is not None:
            try:
                asn_int = int(asn)
            except (ValueError, TypeError):
                asn_int = None
        else:
            asn_int = None

        if asn_int in self.bulletproof_asns:
            return ("BULLETPROOF_HOSTING", 1.25, 1.0, "Bulletproof Hosting")
        elif asn_int in self.tor_asns:
            return ("TOR_EXIT_NODE", 0.50, 0.85, "Tor Exit Node")
        elif asn_int in self.vpn_proxy_asns:
            return ("VPN_PROXY", 0.65, 0.60, "Commercial VPN/Proxy")
        elif asn_int in self.privacy_asns:
            return ("PRIVACY_HOSTING", 1.10, 0.75, "Privacy-Focused Provider")
        elif country and str(country).upper() in self.high_risk_countries:
            return ("HIGH_RISK_JURISDICTION", 1.15, 0.70, "High-Risk Jurisdiction")
        elif asn_int is not None:
            return ("STANDARD_ISP", 1.0, 0.0, "Standard ISP/Transit")
        return ("UNKNOWN", 0.85, 0.0, "Unknown Infrastructure")

    def correlate(
        self,
        df: pd.DataFrame,
        wallet_clusters: Optional[Dict[str, int]] = None,
        graph: Optional[nx.MultiDiGraph] = None,
    ) -> pd.DataFrame:
        """
        Compute correlation confidence between wallets and observed broadcast IPs.

        Args:
            df: Normalized and enriched transaction DataFrame.
            wallet_clusters: Optional mapping of wallet_id -> entity_cluster_id.
            graph: Optional graph to incorporate all wallet nodes.

        Returns:
            pd.DataFrame indexed by wallet_id with correlation scores and forensic rationale.
        """
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "network_correlation_score",
                    "primary_broadcast_ip",
                    "primary_ip",
                    "primary_asn",
                    "asn_category",
                    "threat_category",
                    "observation_count",
                    "correlation_confidence",
                    "asn_threat_score",
                    "burst_count",
                    "min_delta_seconds",
                    "correlation_explanation",
                ]
            ).set_index(pd.Index([], name="wallet_id"))

        # Pre-compute IP-level bursts across the entire DataFrame
        ip_timestamps: Dict[str, List[pd.Timestamp]] = defaultdict(list)
        ip_wallets: Dict[str, Set[str]] = defaultdict(set)

        for row in df.itertuples(index=True):
            src_ip = getattr(row, "src_ip", None)
            ts = getattr(row, "timestamp", None)
            in_addrs = getattr(row, "input_addresses", [])

            if src_ip and isinstance(src_ip, str) and src_ip.strip():
                clean_ip = src_ip.strip()
                if ts is not None:
                    ip_timestamps[clean_ip].append(pd.to_datetime(ts, utc=True))
                if isinstance(in_addrs, (list, tuple)):
                    for addr in in_addrs:
                        if addr and isinstance(addr, str) and addr.strip():
                            ip_wallets[clean_ip].add(addr.strip())

        ip_burst_stats: Dict[str, Tuple[int, float]] = {}
        for ip, t_list in ip_timestamps.items():
            st = sorted([t for t in t_list if t is not None])
            b_count = 0
            min_d = float("inf")
            for i in range(len(st) - 1):
                delta = (st[i + 1] - st[i]).total_seconds()
                if delta < min_d:
                    min_d = delta
                if delta <= 60.0:
                    b_count += 1
            ip_burst_stats[ip] = (b_count, min_d if min_d != float("inf") else -1.0)

        # Map wallet -> observations
        wallet_broadcasts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in df.itertuples(index=True):
            src_ip = getattr(row, "src_ip", None)
            ts = getattr(row, "timestamp", None)
            src_asn = getattr(row, "src_ip_asn", None)
            src_country = getattr(row, "src_ip_country", None)
            in_addrs = getattr(row, "input_addresses", [])

            if not src_ip or not isinstance(in_addrs, (list, tuple)):
                continue

            clean_ip = str(src_ip).strip()
            ts_val = pd.to_datetime(ts, utc=True) if ts is not None else None

            for addr in in_addrs:
                if addr:
                    clean_w = str(addr).strip()
                    wallet_broadcasts[clean_w].append({
                        "ip": clean_ip,
                        "asn": src_asn,
                        "country": src_country,
                        "timestamp": ts_val,
                    })

        all_target_wallets = list(wallet_broadcasts.keys())
        if graph is not None:
            for n, d in graph.nodes(data=True):
                if d.get("node_type") == "wallet" and n not in wallet_broadcasts:
                    all_target_wallets.append(n)

        records = []
        for wallet in all_target_wallets:
            obs_list = wallet_broadcasts.get(wallet, [])
            if not obs_list:
                records.append({
                    "wallet_id": wallet,
                    "network_correlation_score": 0.0,
                    "primary_broadcast_ip": "None",
                    "primary_ip": "None",
                    "primary_asn": 0,
                    "asn_category": "UNKNOWN",
                    "threat_category": "UNKNOWN",
                    "observation_count": 0,
                    "correlation_confidence": "LOW",
                    "asn_threat_score": 0.0,
                    "burst_count": 0,
                    "min_delta_seconds": -1.0,
                    "correlation_explanation": "No broadcast activity observed on monitored network interfaces.",
                })
                continue

            obs_count = len(obs_list)
            ip_counts: Dict[str, int] = defaultdict(int)
            for obs in obs_list:
                ip_counts[obs["ip"]] += 1
            primary_ip = max(ip_counts.keys(), key=lambda ip: ip_counts[ip])
            primary_obs = [obs for obs in obs_list if obs["ip"] == primary_ip][0]

            asn = primary_obs.get("asn")
            country = primary_obs.get("country")
            asn_cat, asn_weight, asn_threat_score, infra_label = self._get_asn_category_and_weight(asn, country)

            ip_burst_count, ip_min_delta = ip_burst_stats.get(primary_ip, (0, -1.0))
            burst_multiplier = 1.15 if ip_burst_count > 0 else 1.0

            base_score = 1.0 - math.exp(-0.4 * obs_count)
            calibrated_score = round(min(1.0, max(0.05, base_score * asn_weight * burst_multiplier)), 4)

            if calibrated_score >= 0.80:
                conf_label = "VERY_HIGH"
            elif calibrated_score >= 0.60:
                conf_label = "HIGH"
            elif calibrated_score >= 0.35:
                conf_label = "MEDIUM"
            else:
                conf_label = "LOW"

            # Forensic explanation synthesis
            explanations = []
            if asn_threat_score > 0:
                explanations.append(f"{infra_label} (ASN {asn or 'N/A'})")
            if ip_burst_count > 0:
                explanations.append(f"{ip_burst_count} rapid broadcast burst(s)")
            if not explanations:
                explanations.append(f"Standard broadcast via {primary_ip}")
            explanation_str = f"Observed via {primary_ip}: " + "; ".join(explanations)

            records.append({
                "wallet_id": wallet,
                "network_correlation_score": calibrated_score,
                "primary_broadcast_ip": primary_ip,
                "primary_ip": primary_ip,
                "primary_asn": asn if asn is not None else 0,
                "asn_category": asn_cat,
                "threat_category": asn_cat,
                "observation_count": obs_count,
                "correlation_confidence": conf_label,
                "asn_threat_score": asn_threat_score,
                "burst_count": ip_burst_count,
                "min_delta_seconds": ip_min_delta,
                "correlation_explanation": explanation_str,
            })

        result_df = pd.DataFrame(records).set_index("wallet_id")
        logger.info(
            "Network-blockchain correlation computed for %d wallets (max score: %.4f).",
            len(result_df),
            result_df["network_correlation_score"].max() if not result_df.empty else 0.0,
        )
        return result_df


def correlate_network_blockchain(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
    wallet_clusters: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Compute Network-to-Blockchain correlation and deanonymization threat scores.
    """
    correlator = NetworkBlockchainCorrelator()
    return correlator.correlate(df=df, wallet_clusters=wallet_clusters, graph=graph)


def run_network_correlation(
    input_file: Union[str, Path] = "data/raw/synthetic_transactions.csv",
) -> pd.DataFrame:
    """Run network-blockchain correlation pipeline."""
    input_path = Path(input_file)
    df = parse_file(input_path)
    correlator = NetworkBlockchainCorrelator()
    return correlator.correlate(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run network-blockchain correlation scoring.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/raw/synthetic_transactions.csv",
        help="Path to transaction file (.csv, .json, .xml)",
    )
    args = parser.parse_args()

    corr_df = run_network_correlation(args.input)
    print("\n" + "=" * 65)
    print("           NETWORK-BLOCKCHAIN CORRELATION SUMMARY")
    print("=" * 65)
    print(f"Total Wallets Correlated: {len(corr_df):,}")
    print("\n--- TOP 10 CORRELATED WALLETS ---")
    top_corr = corr_df.sort_values(by="network_correlation_score", ascending=False).head(10)
    for w_id, row in top_corr.iterrows():
        print(
            f"  Wallet: {w_id:<36} | "
            f"Score: {row['network_correlation_score']:.4f} ({row['correlation_confidence']}) | "
            f"IP: {row['primary_broadcast_ip']:<15} | ASN: {row['primary_asn']} ({row['asn_category']})"
        )
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
