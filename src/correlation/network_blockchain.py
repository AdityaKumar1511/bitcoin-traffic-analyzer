"""
Network-Blockchain Correlation Engine.

Quantifies the correlation confidence tying network broadcast observations (IP/ASN/timing)
to on-chain wallet entities and transaction clusters. Calibrates scores by discounting
known VPN/Tor/Cloud exit nodes and up-weighting persistent, bulletproof, or burst co-locations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from src.features import extract_all_features
from src.graph.builder import build_graph, get_nodes_by_type
from src.graph.heuristics import apply_change_address_heuristic, apply_common_input_heuristic, get_wallet_clusters
from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger
from src.utils.paths import get_project_root, load_config

logger = get_logger(__name__)


class NetworkBlockchainCorrelator:
    """
    Computes calibrated correlation scores between network broadcasting IPs/ASNs and wallet entities.
    """

    def __init__(self, high_risk_asns_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize correlator and load ASN taxonomy.
        """
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
                self.bulletproof_asns = set(asn_cfg.get("bulletproof_hosting", []))
                self.vpn_proxy_asns = set(asn_cfg.get("vpn_proxy", []))
                self.tor_asns = set(asn_cfg.get("tor_related", []))
                self.privacy_asns = set(asn_cfg.get("privacy_focused", []))
                self.high_risk_countries = set(asn_cfg.get("high_risk_countries", []))

    def _get_asn_category_and_weight(self, asn: Optional[int], country: Optional[str]) -> Tuple[str, float]:
        """
        Categorize ASN and return correlation confidence multiplier.

        Discount shared infrastructure (VPN/Tor/Cloud) because 1 IP != 1 Actor.
        Boost or maintain strong correlation for bulletproof hosting or dedicated residential IPs.
        """
        if asn is not None:
            try:
                asn_int = int(asn)
            except (ValueError, TypeError):
                asn_int = None
        else:
            asn_int = None

        if asn_int in self.bulletproof_asns:
            return ("BULLETPROOF_HOSTING", 1.25)
        elif asn_int in self.tor_asns:
            # Tor exit nodes are heavily shared; high anonymity risk but lower identity linkage confidence
            return ("TOR_EXIT_NODE", 0.50)
        elif asn_int in self.vpn_proxy_asns:
            # VPN/Proxy shared pool
            return ("VPN_PROXY", 0.65)
        elif asn_int in self.privacy_asns:
            return ("PRIVACY_HOSTING", 1.10)
        elif country and str(country).upper() in self.high_risk_countries:
            return ("HIGH_RISK_JURISDICTION", 1.15)
        elif asn_int is not None:
            return ("STANDARD_ISP", 1.0)
        return ("UNKNOWN", 0.85)

    def correlate(
        self,
        df: pd.DataFrame,
        wallet_clusters: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """
        Compute correlation confidence between wallets and observed broadcast IPs.

        Args:
            df: Normalized and enriched transaction DataFrame.
            wallet_clusters: Optional mapping of wallet_id -> entity_cluster_id.

        Returns:
            pd.DataFrame indexed by wallet_id with columns:
                network_correlation_score, primary_broadcast_ip, primary_asn,
                asn_category, observation_count, correlation_confidence
        """
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "network_correlation_score",
                    "primary_broadcast_ip",
                    "primary_asn",
                    "asn_category",
                    "observation_count",
                    "correlation_confidence",
                ]
            ).set_index(pd.Index([], name="wallet_id"))

        # Map wallet -> list of observed (ip, asn, country, timestamp)
        wallet_broadcasts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for row in df.itertuples(index=True):
            src_ip = getattr(row, "src_ip", None)
            src_asn = getattr(row, "src_ip_asn", None)
            src_country = getattr(row, "src_ip_country", None)
            ts = getattr(row, "timestamp", None)
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

        records = []
        for wallet, obs_list in wallet_broadcasts.items():
            obs_count = len(obs_list)
            # Find primary (most frequent) IP
            ip_counts = defaultdict(int)
            for obs in obs_list:
                ip_counts[obs["ip"]] += 1
            primary_ip = max(ip_counts.keys(), key=lambda ip: ip_counts[ip])
            primary_obs = [obs for obs in obs_list if obs["ip"] == primary_ip][0]

            asn = primary_obs.get("asn")
            country = primary_obs.get("country")
            asn_cat, asn_weight = self._get_asn_category_and_weight(asn, country)

            # Check for temporal burstiness (multiple txs within 5 minutes)
            timestamps = sorted([obs["timestamp"] for obs in obs_list if obs["timestamp"] is not None])
            burst_count = 0
            for i in range(len(timestamps) - 1):
                if (timestamps[i + 1] - timestamps[i]).total_seconds() <= 300:
                    burst_count += 1
            burst_multiplier = 1.15 if burst_count > 0 else 1.0

            # Calibrated correlation formula:
            # Scales asymptotically with observation count, adjusted by ASN weight and burstiness
            base_score = 1.0 - math.exp(-0.4 * obs_count)
            calibrated_score = round(min(1.0, max(0.05, base_score * asn_weight * burst_multiplier)), 4)

            # Qualitative confidence label
            if calibrated_score >= 0.80:
                conf_label = "VERY_HIGH"
            elif calibrated_score >= 0.60:
                conf_label = "HIGH"
            elif calibrated_score >= 0.35:
                conf_label = "MEDIUM"
            else:
                conf_label = "LOW"

            records.append({
                "wallet_id": wallet,
                "network_correlation_score": calibrated_score,
                "primary_broadcast_ip": primary_ip,
                "primary_asn": asn,
                "asn_category": asn_cat,
                "observation_count": obs_count,
                "correlation_confidence": conf_label,
            })

        result_df = pd.DataFrame(records).set_index("wallet_id")
        logger.info(
            "Network-blockchain correlation computed for %d wallets (max score: %.4f).",
            len(result_df),
            result_df["network_correlation_score"].max() if not result_df.empty else 0.0,
        )
        return result_df


def run_network_correlation(
    input_file: Union[str, Path] = "data/raw/synthetic_transactions.csv",
) -> pd.DataFrame:
    """Run network-blockchain correlation pipeline."""
    df = parse_file(Path(input_file))
    correlator = NetworkBlockchainCorrelator()
    return correlator.correlate(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run network-blockchain correlation scoring.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/raw/synthetic_transactions.csv",
        help="Path to transaction file",
    )
    args = parser.parse_args()

    corr_df = run_network_correlation(args.input)
    print("\n" + "=" * 65)
    print("      NETWORK-BLOCKCHAIN CORRELATION SCORING SUMMARY")
    print("=" * 65)
    print(f"Total Evaluated Wallets: {len(corr_df):,}")

    top_corr = corr_df.sort_values(by="network_correlation_score", ascending=False).head(10)
    print("\n--- TOP 10 CORRELATED WALLET-IP ENTITIES ---")
    for w_id, row in top_corr.iterrows():
        print(
            f"  Wallet: {w_id}\n"
            f"    Score: {row['network_correlation_score']:.4f} ({row['correlation_confidence']}) | "
            f"IP: {row['primary_broadcast_ip']} | ASN: {row['primary_asn']} [{row['asn_category']}] | "
            f"Obs: {row['observation_count']}"
        )

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
