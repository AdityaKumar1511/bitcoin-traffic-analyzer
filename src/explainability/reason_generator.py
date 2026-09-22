"""
Investigator-Ready Reason Generator for Flagged Alerts.

Generates human-readable, evidence-grounded plain-English summaries and
detailed justifications for flagged Bitcoin entities, transactions, and IPs.

Converts raw SHAP attribution scores, evasion pattern flags, network metadata,
taint propagation results, and behavioral features into court-presentable,
investigator-actionable natural-language explanations with severity-scaled wording.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Human-readable feature name mappings for SHAP output
_FEATURE_LABELS: Dict[str, str] = {
    "in_degree": "incoming counterparty count",
    "out_degree": "outgoing counterparty count",
    "total_btc_in": "total BTC received",
    "total_btc_out": "total BTC sent",
    "net_flow": "net BTC flow balance",
    "tx_count": "transaction count",
    "avg_amount": "average transaction amount",
    "std_amount": "transaction amount variability",
    "fan_in_ratio": "fan-in aggregation ratio",
    "fan_out_ratio": "fan-out dispersion ratio",
    "distinct_ip_count": "distinct broadcasting IP count",
    "distinct_country_count": "distinct country count",
    "distinct_asn_count": "distinct ASN count",
    "cluster_size": "co-spending cluster size",
    "peel_chain_participation": "peel chain involvement",
    "peel_chain_depth": "peel chain depth",
    "address_lifespan_hours": "address active lifespan",
    "reuse_count": "address reuse frequency",
    "hour_entropy": "time-of-day distribution entropy",
    "burstiness": "inter-transaction burstiness",
    "velocity": "transaction velocity",
    "ip_wallet_diversity": "IP-to-wallet diversity",
    "timing_tightness": "broadcast timing tightness",
    "port_anomaly": "non-standard port usage",
    "asn_risk_score": "ASN infrastructure risk score",
    "geo_concentration": "geographic concentration entropy",
    "num_inputs": "transaction input count",
    "num_outputs": "transaction output count",
    "total_value": "transaction total value",
    "fee": "miner fee",
    "fee_per_input": "fee per input",
    "is_high_risk_asn": "high-risk ASN flag",
    "is_coinjoin_candidate": "CoinJoin candidate flag",
    "is_peel_step": "peel chain step flag",
    "iforest_score": "Isolation Forest anomaly score",
    "reconstruction_score": "reconstruction error score",
}

# Severity-scaled adjective templates
_SEVERITY_ADJECTIVES = {
    "CRITICAL": ("extremely", "exceptionally", "highly alarming"),
    "HIGH": ("significantly", "notably", "substantially"),
    "MEDIUM": ("moderately", "measurably", "noticeably"),
    "LOW": ("slightly", "marginally", "mildly"),
}


def _humanize_feature_name(feature: str) -> str:
    """Convert a snake_case feature name to a readable label."""
    return _FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _severity_level(score: float) -> str:
    """Map [0, 1] score to severity label."""
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.50:
        return "HIGH"
    elif score >= 0.30:
        return "MEDIUM"
    return "LOW"


def _severity_adverb(score: float) -> str:
    """Return a severity-appropriate adverb."""
    level = _severity_level(score)
    adverbs = _SEVERITY_ADJECTIVES.get(level, ("moderately",))
    return adverbs[0]


class ReasonGenerator:
    """
    Generates plain-language investigative justifications for forensic alerts.

    Converts raw ML outputs into structured, evidence-grounded investigator briefings:
    - One-line primary reason (for alert table)
    - Multi-point forensic justification (for detail view / audit trail)
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

        Args:
            entity_id: Wallet/TXID/IP identifier.
            entity_type: "wallet", "txid", or "ip".
            risk_score: Final composite risk score [0.0, 1.0].
            features: Dict of computed features for this entity.
            top_shap_features: Dict of {feature_name: signed_shap_value} from ModelExplainer.
            evasion_flags: List of matched evasion pattern names.
            network_data: Dict with network correlation metadata.
            taint_data: Dict with taint propagation metadata.
            cluster_info: Dict with entity cluster metadata.

        Returns:
            dict: {
                "primary_reason": str (one-line summary for alert table),
                "detailed_justification": str (multi-point forensic breakdown for detail view)
            }
        """
        features = features or {}
        top_shap_features = top_shap_features or {}
        evasion_flags = evasion_flags or []
        network_data = network_data or {}
        taint_data = taint_data or {}
        cluster_info = cluster_info or {}

        severity = _severity_level(risk_score)
        adverb = _severity_adverb(risk_score)

        reasons_list: List[str] = []
        evidence_points: List[str] = []

        # ──────────────────────────────────────────────
        # 1. Evasion Patterns (highest priority signals)
        # ──────────────────────────────────────────────
        for flag in evasion_flags:
            flag_lower = flag.lower()
            if "coinjoin" in flag_lower:
                reasons_list.append("participated in CoinJoin mixing transaction")
                evidence_points.append(
                    "🔴 **CoinJoin Mixing Detected:** Transaction exhibits high fan-in/fan-out "
                    "with standardized equal-denomination output splitting, consistent with "
                    "coordinated multi-party mixing to break transaction traceability."
                )
            elif "peel" in flag_lower:
                reasons_list.append("active carrier in ransomware peel-chain forwarding")
                evidence_points.append(
                    "🔴 **Peel Chain Involvement:** Identified as a forwarding node in a sequential "
                    "peel chain — repeatedly receiving a bulk amount, peeling off a small portion, "
                    "and forwarding the remainder to a new address. Consistent with ransomware "
                    "exfiltration or systematic money laundering."
                )
            elif "mixer" in flag_lower:
                reasons_list.append("exhibited mixer pass-through hub throughput")
                evidence_points.append(
                    "🔴 **Mixer Hub Behavior:** Wallet demonstrates high-throughput pass-through activity — "
                    "receiving from multiple distinct sources and redistributing to multiple distinct "
                    "destinations within a compressed time window, matching known mixing service patterns."
                )

        # ──────────────────────────────────────────────
        # 2. SHAP Feature Attributions (ML-driven signals)
        # ──────────────────────────────────────────────
        if top_shap_features:
            shap_evidence_items: List[str] = []
            for feat_name, shap_val in list(top_shap_features.items())[:5]:
                human_name = _humanize_feature_name(feat_name)
                direction = "increased" if shap_val > 0 else "decreased"
                raw_val = features.get(feat_name)
                val_str = f" (value: {raw_val:.4f})" if isinstance(raw_val, (int, float)) else ""
                shap_evidence_items.append(
                    f"{human_name}{val_str} [{shap_val:+.3f} → {direction} risk]"
                )

            if shap_evidence_items:
                evidence_points.append(
                    "📊 **Top Model Attribution Drivers (SHAP):**\n"
                    + "\n".join(f"    • {item}" for item in shap_evidence_items)
                )

                # Add top SHAP feature to primary reason if no evasion patterns
                if not reasons_list and shap_evidence_items:
                    top_feat_name = list(top_shap_features.keys())[0]
                    reasons_list.append(
                        f"{adverb} anomalous {_humanize_feature_name(top_feat_name)}"
                    )

        # ──────────────────────────────────────────────
        # 3. Graph & Behavioral Features
        # ──────────────────────────────────────────────
        fan_in = features.get("fan_in_ratio", 0.0)
        fan_out = features.get("fan_out_ratio", 0.0)
        reuse_cnt = features.get("reuse_count", 0)
        cluster_sz = cluster_info.get("cluster_size", 1)
        tx_count = features.get("tx_count", 0)
        addr_lifespan = features.get("address_lifespan_hours", 0)

        if isinstance(fan_in, (int, float)) and fan_in >= 0.75:
            reasons_list.append(f"high fan-in aggregation ratio ({fan_in:.2f})")
            evidence_points.append(
                f"📈 **Fund Consolidation Signal:** Fan-in ratio of {fan_in:.2f} indicates "
                f"this wallet is {adverb} aggregating funds from multiple source wallets, "
                f"consistent with collection/consolidation behavior."
            )
        if isinstance(fan_out, (int, float)) and fan_out >= 0.75:
            reasons_list.append(f"high fan-out dispersion ratio ({fan_out:.2f})")
            evidence_points.append(
                f"📉 **Fund Dispersion Signal:** Fan-out ratio of {fan_out:.2f} indicates "
                f"this wallet is {adverb} distributing funds across multiple destination wallets, "
                f"consistent with layering or distribution to mules."
            )

        # Address reuse anomaly
        if isinstance(reuse_cnt, (int, float)) and reuse_cnt >= 5:
            evidence_points.append(
                f"🔁 **Unusual Address Reuse:** Address reused {int(reuse_cnt)} times — "
                f"significantly above typical Bitcoin privacy practices (1-2 uses per address). "
                f"May indicate an automated service or intentional re-aggregation point."
            )

        # Cluster membership
        if isinstance(cluster_sz, (int, float)) and cluster_sz > 1:
            cluster_id = cluster_info.get("cluster_id", "N/A")
            evidence_points.append(
                f"🔗 **Co-Spending Cluster Membership:** Member of entity cluster "
                f"#{cluster_id} containing {int(cluster_sz)} addresses linked via "
                f"cryptographic common-input-ownership heuristic."
            )

        # High transaction velocity with short lifespan (disposable wallet)
        if isinstance(tx_count, (int, float)) and isinstance(addr_lifespan, (int, float)):
            if tx_count >= 10 and 0 < addr_lifespan < 24:
                evidence_points.append(
                    f"⚡ **Disposable Wallet Pattern:** {int(tx_count)} transactions within "
                    f"{addr_lifespan:.1f} hours lifespan suggests a short-lived, high-velocity "
                    f"wallet used for rapid fund movement."
                )

        # ──────────────────────────────────────────────
        # 4. Temporal Anomalies
        # ──────────────────────────────────────────────
        hour_entropy = features.get("hour_entropy", None)
        burstiness = features.get("burstiness", None)
        velocity = features.get("velocity", None)

        if isinstance(hour_entropy, (int, float)) and hour_entropy > 3.0:
            evidence_points.append(
                f"🕐 **Unnatural Activity Distribution:** Time-of-day entropy of {hour_entropy:.2f} "
                f"(near-maximum for 24-hour cycle) indicates uniformly distributed activity — "
                f"inconsistent with human behavior, suggesting automated/scripted operation."
            )
        if isinstance(burstiness, (int, float)) and burstiness > 2.0:
            evidence_points.append(
                f"💥 **Transaction Burstiness:** Coefficient of variation of {burstiness:.2f} "
                f"in inter-transaction timing indicates highly irregular burst patterns — "
                f"alternating between dormancy and intense activity."
            )
        if isinstance(velocity, (int, float)) and velocity > 5.0:
            evidence_points.append(
                f"🚀 **High Transaction Velocity:** {velocity:.1f} transactions/hour — "
                f"well above typical user behavior, suggesting automated fund movement."
            )

        # ──────────────────────────────────────────────
        # 5. Network & GeoIP Forensics
        # ──────────────────────────────────────────────
        ip = network_data.get("primary_broadcast_ip")
        asn = network_data.get("primary_asn")
        asn_cat = network_data.get("asn_category", "")
        obs_count = network_data.get("observation_count", 0)

        if asn_cat == "BULLETPROOF_HOSTING":
            reasons_list.append(f"broadcast from known bulletproof ASN ({asn})")
            evidence_points.append(
                f"🛡️ **Bulletproof Hosting Infrastructure:** Origin IP {ip} resides in "
                f"ASN {asn}, identified as a bulletproof hosting provider known for "
                f"harboring criminal infrastructure and ignoring abuse complaints."
            )
        elif asn_cat == "TOR_EXIT_NODE":
            evidence_points.append(
                f"🧅 **Tor Exit Node Broadcast:** Transaction broadcast via Tor exit node "
                f"(ASN {asn}), indicating deliberate anonymization of broadcasting infrastructure."
            )
        elif asn_cat == "VPN_PROXY":
            evidence_points.append(
                f"🔒 **VPN/Proxy Broadcast:** Transaction broadcast via commercial VPN/proxy "
                f"service (ASN {asn}), reducing confidence in IP-to-entity attribution."
            )
        elif asn_cat == "PRIVACY_HOSTING":
            evidence_points.append(
                f"🏴 **Privacy-Focused Hosting:** Broadcast from privacy-focused hosting provider "
                f"(ASN {asn}), commonly associated with privacy-conscious or evasive actors."
            )
        elif asn_cat == "HIGH_RISK_JURISDICTION":
            evidence_points.append(
                "🌐 **High-Risk Jurisdiction:** Broadcast originated from a jurisdiction "
                "flagged as high-risk for financial crime or with limited cooperation treaties."
            )

        if isinstance(obs_count, (int, float)) and obs_count > 1 and ip:
            evidence_points.append(
                f"📡 **Repeated Broadcast Correlation:** Observed in {int(obs_count)} independent "
                f"transaction broadcasts from IP {ip}, strengthening IP-to-entity attribution confidence."
            )

        # IP diversity anomaly
        distinct_ips = features.get("distinct_ip_count", 0)
        if isinstance(distinct_ips, (int, float)) and distinct_ips >= 5:
            evidence_points.append(
                f"🌍 **Multi-IP Broadcasting:** Transactions broadcast from {int(distinct_ips)} "
                f"distinct IP addresses, suggesting geographically distributed infrastructure, "
                f"VPN rotation, or operational security countermeasures."
            )

        # ──────────────────────────────────────────────
        # 6. Multi-Hop Taint Propagation
        # ──────────────────────────────────────────────
        taint_score = taint_data.get("taint_score", 0.0)
        taint_hops = taint_data.get("taint_hops", 0)
        taint_src = taint_data.get("taint_source")
        taint_path = taint_data.get("taint_path", "")

        if isinstance(taint_score, (int, float)) and taint_score > 0 and taint_src and taint_src != entity_id:
            taint_severity = "direct" if taint_hops == 1 else f"{taint_hops}-hop indirect"
            reasons_list.append(
                f"{taint_severity} risk taint ({taint_score:.2f}) from seed {str(taint_src)[:8]}..."
            )
            evidence_points.append(
                f"☣️ **Multi-Hop Taint Propagation:** Risk taint score of {taint_score:.2f} "
                f"inherited across {taint_hops} transaction hops from known-suspicious seed wallet "
                f"`{taint_src}`. Taint flow path: `{taint_path}`."
            )

        # ──────────────────────────────────────────────
        # Assemble final output
        # ──────────────────────────────────────────────
        if not reasons_list:
            reasons_list.append(f"statistical outlier with {adverb} elevated anomaly risk score {risk_score:.2f}")

        # Primary reason: concise one-liner for alert table
        primary_reason = f"Flagged due to {', '.join(reasons_list[:3])} (Risk Score: {risk_score:.2f})."

        # Detailed justification: structured forensic breakdown for detail view
        detail_lines = [
            f"### Investigative Alert — {entity_type.upper()} `{entity_id}`",
            f"**Composite Risk Score:** {risk_score:.4f} ({severity})",
            "",
            "**Identified Forensic Signals:**",
        ]
        for ep in evidence_points:
            detail_lines.append(f"- {ep}")

        if not evidence_points:
            detail_lines.append(
                "- Statistical deviation detected across multiple behavioral dimensions. "
                "Manual review recommended."
            )

        detailed_justification = "\n".join(detail_lines)

        return {
            "primary_reason": primary_reason,
            "detailed_justification": detailed_justification,
        }

    @staticmethod
    def generate_cluster_explanation(
        cluster_id: int,
        cluster_wallets: List[str],
        method: str,
        confidence: float,
        primary_ip: Optional[str] = None,
    ) -> str:
        """
        Generate a plain-English explanation of why wallets were grouped into a cluster.

        Args:
            cluster_id: Cluster identifier.
            cluster_wallets: List of wallet addresses in the cluster.
            method: Clustering method used ("common_input_ownership", "exclusive_ip_linkage", etc).
            confidence: Clustering confidence score [0, 1].
            primary_ip: Primary broadcasting IP if available.

        Returns:
            Human-readable cluster explanation string.
        """
        n_wallets = len(cluster_wallets)

        if method == "common_input_ownership":
            explanation = (
                f"Entity Cluster #{cluster_id}: {n_wallets} wallet addresses grouped via "
                f"cryptographic common-input-ownership (CIO) heuristic — these addresses "
                f"were used as co-inputs in the same transaction(s), proving they are controlled "
                f"by the same private key holder. Confidence: {confidence:.0%}."
            )
        elif method == "exclusive_ip_linkage":
            ip_str = f" (primary IP: {primary_ip})" if primary_ip else ""
            explanation = (
                f"Entity Cluster #{cluster_id}: {n_wallets} wallet addresses grouped via "
                f"exclusive IP linkage — all addresses exclusively broadcast transactions "
                f"from the same IP address{ip_str}, suggesting a single operator node. "
                f"Confidence: {confidence:.0%}."
            )
        elif method == "embedding_similarity":
            explanation = (
                f"Entity Cluster #{cluster_id}: {n_wallets} wallet addresses grouped via "
                f"graph embedding similarity — structural behavior patterns in the transaction "
                f"graph are statistically indistinguishable, suggesting same-actor control. "
                f"Confidence: {confidence:.0%}."
            )
        elif method == "singleton":
            explanation = (
                f"Entity Cluster #{cluster_id}: Singleton — no co-spending or IP linkage "
                f"detected for this address. May be a one-time-use address."
            )
        else:
            explanation = (
                f"Entity Cluster #{cluster_id}: {n_wallets} addresses grouped via "
                f"method '{method}'. Confidence: {confidence:.0%}."
            )

        return explanation
