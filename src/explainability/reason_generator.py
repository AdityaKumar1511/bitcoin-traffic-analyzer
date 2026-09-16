"""
Investigator-Ready Reason Generator for Flagged Alerts.

Generates human-readable, evidence-grounded plain-English summaries and
detailed justifications for flagged Bitcoin entities, transactions, and IPs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ReasonGenerator:
    """
    Generates plain-language investigative justifications for forensic alerts.
    """

    @staticmethod
    def generate_reasons(
        entity_id: str,
        entity_type: str,
        risk_score: float,
        features: Optional[Dict[str, Any]] = None,
        top_shap_features: Optional[Dict[str, float]] = None,
        evasion_flags: Optional[List[str]] = None,
        network_data: Optional[Dict[str, Any]] = None,
        taint_data: Optional[Dict[str, Any]] = None,
        cluster_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Generate concise primary reason and comprehensive detailed justification.

        Returns:
            dict: {
                "primary_reason": str (one-line summary),
                "detailed_justification": str (multi-point forensic breakdown)
            }
        """
        features = features or {}
        top_shap_features = top_shap_features or {}
        evasion_flags = evasion_flags or []
        network_data = network_data or {}
        taint_data = taint_data or {}
        cluster_info = cluster_info or {}

        reasons_list: List[str] = []
        evidence_points: List[str] = []

        # 1. Evasion Patterns
        for flag in evasion_flags:
            if "coinjoin" in flag.lower():
                reasons_list.append("participated in CoinJoin mixing transaction")
                evidence_points.append("Detected CoinJoin-style equal denomination output splitting.")
            elif "peel" in flag.lower():
                reasons_list.append("active carrier in ransomware peel-chain forwarding")
                evidence_points.append("Identified in sequential peel chain peeling small change outputs.")
            elif "mixer" in flag.lower():
                reasons_list.append("exhibited mixer pass-through hub throughput")
                evidence_points.append("High pass-through counterparty mixing within rolling time window.")

        # 2. Graph & Behavioral Features
        fan_in = features.get("fan_in_ratio", 0.0)
        fan_out = features.get("fan_out_ratio", 0.0)
        reuse_cnt = features.get("reuse_count", 0)
        cluster_sz = cluster_info.get("cluster_size", 1)

        if fan_in >= 0.75:
            reasons_list.append(f"high fan-in aggregation ratio ({fan_in:.2f})")
            evidence_points.append(f"Fan-in ratio of {fan_in:.2f} indicates consolidation of multi-source funds.")
        if fan_out >= 0.75:
            reasons_list.append(f"high fan-out dispersion ratio ({fan_out:.2f})")
            evidence_points.append(f"Fan-out ratio of {fan_out:.2f} indicates multi-target fund dispersion.")
        if cluster_sz > 1:
            evidence_points.append(f"Member of co-spending wallet cluster containing {cluster_sz} addresses.")

        # 3. Network & GeoIP Features
        ip = network_data.get("primary_broadcast_ip")
        asn = network_data.get("primary_asn")
        asn_cat = network_data.get("asn_category", "")
        obs_count = network_data.get("observation_count", 0)

        if asn_cat == "BULLETPROOF_HOSTING":
            reasons_list.append(f"broadcast from known bulletproof ASN ({asn})")
            evidence_points.append(f"Origin IP {ip} resides in high-risk bulletproof hosting ASN {asn}.")
        elif asn_cat in ("TOR_EXIT_NODE", "VPN_PROXY"):
            evidence_points.append(f"Broadcasted via anonymization service / {asn_cat.replace('_', ' ').title()} (ASN {asn}).")
        
        if obs_count > 1 and ip:
            evidence_points.append(f"Observed in {obs_count} independent transaction broadcasts from IP {ip}.")

        # 4. Multi-hop Taint Propagation
        taint_score = taint_data.get("taint_score", 0.0)
        taint_hops = taint_data.get("taint_hops", 0)
        taint_src = taint_data.get("taint_source")
        taint_path = taint_data.get("taint_path", "")

        if taint_score > 0 and taint_src and taint_src != entity_id:
            reasons_list.append(f"{taint_hops}-hop risk taint ({taint_score:.2f}) inherited from seed {taint_src[:8]}...")
            evidence_points.append(
                f"Multi-hop risk taint of {taint_score:.2f} traced across {taint_hops} hops from seed wallet {taint_src} (Flow path: {taint_path})."
            )

        # 5. SHAP Features
        if top_shap_features:
            shap_strs = [f"{k} ({v:+.2f})" for k, v in list(top_shap_features.items())[:3]]
            evidence_points.append(f"Top model attribution drivers: {', '.join(shap_strs)}.")

        if not reasons_list:
            reasons_list.append(f"statistical outlier with anomaly risk score {risk_score:.2f}")

        primary_reason = f"Flagged due to {', '.join(reasons_list[:3])} (Risk Score: {risk_score:.2f})."
        
        detail_lines = [
            f"**Investigative Alert Summary for {entity_type.upper()} `{entity_id}`**",
            f"- **Composite Risk Score:** {risk_score:.4f}",
            "- **Identified Forensic Signals:**",
        ]
        for ep in evidence_points:
            detail_lines.append(f"  • {ep}")

        detailed_justification = "\n".join(detail_lines)

        return {
            "primary_reason": primary_reason,
            "detailed_justification": detailed_justification,
        }
