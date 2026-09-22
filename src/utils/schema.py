"""
Schema definitions, type annotations, and column constants for Bitcoin Traffic Analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Core Transaction schema columns expected from parsers
RAW_TRANSACTION_COLUMNS: List[str] = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "txid",
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
    "fee",
    "script_type",
]

# Enriched Transaction schema columns
ENRICHED_TRANSACTION_COLUMNS: List[str] = RAW_TRANSACTION_COLUMNS + [
    "src_ip_country",
    "src_ip_asn",
]

# Node types in heterogeneous transaction graph
NODE_TYPE_WALLET = "wallet"
NODE_TYPE_TRANSACTION = "transaction"
NODE_TYPE_IP = "ip"

# Edge types in heterogeneous transaction graph
EDGE_TYPE_INPUT = "input"
EDGE_TYPE_OUTPUT = "output"
EDGE_TYPE_BROADCAST = "broadcast"
EDGE_TYPE_COMMON_INPUT = "common_input_ownership"

# Pattern types for detection and ground truth
PATTERN_RANSOMWARE_PEELCHAIN = "ransomware_peelchain"
PATTERN_SAME_ACTOR_CLUSTER = "same_actor_cluster"
PATTERN_COINJOIN_MIXING = "coinjoin_mixing"
PATTERN_MIXER_HUB = "mixer_hub"
PATTERN_ANOMALOUS_OUTLIER = "anomalous_outlier"
PATTERN_HIGH_RISK_NETWORK = "high_risk_network"


@dataclass
class InvestigativeAlert:
    """Represents a prioritized investigative alert for a flagged entity."""

    entity_id: str
    entity_type: str  # "wallet", "txid", "ip"
    composite_risk_score: float
    risk_level: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    pattern_types: List[str] = field(default_factory=list)
    primary_reason: str = ""
    detailed_justification: str = ""
    top_contributing_features: Dict[str, float] = field(default_factory=dict)
    associated_ips: List[str] = field(default_factory=list)
    associated_asns: List[int] = field(default_factory=list)
    associated_txids: List[str] = field(default_factory=list)
    taint_source: Optional[str] = None
    taint_hops: int = 0
    taint_score: float = 0.0
    entity_cluster_id: Optional[int] = None
    entity_cluster_size: int = 1
    analyst_status: str = "PENDING"  # "PENDING", "CONFIRMED", "FALSE_POSITIVE"
    analyst_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "composite_risk_score": round(self.composite_risk_score, 4),
            "risk_level": self.risk_level,
            "pattern_types": self.pattern_types,
            "primary_reason": self.primary_reason,
            "detailed_justification": self.detailed_justification,
            "top_contributing_features": self.top_contributing_features,
            "associated_ips": self.associated_ips,
            "associated_asns": self.associated_asns,
            "associated_txids": self.associated_txids,
            "taint_source": self.taint_source,
            "taint_hops": self.taint_hops,
            "taint_score": round(self.taint_score, 4),
            "entity_cluster_id": self.entity_cluster_id,
            "entity_cluster_size": self.entity_cluster_size,
            "analyst_status": self.analyst_status,
            "analyst_notes": self.analyst_notes,
        }


def compute_risk_level(score: float) -> str:
    """Map continuous score [0, 1] to risk category."""
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.50:
        return "HIGH"
    elif score >= 0.30:
        return "MEDIUM"
    return "LOW"
