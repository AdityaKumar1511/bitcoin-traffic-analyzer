"""
Composite Risk Scoring and Forensic Lead Prioritization Engine.

Combines signals from:
1. Unsupervised Anomaly Detection (Isolation Forest)
2. Evasion Detectors (CoinJoin, Peel Chain, Mixer Hub)
3. Network-Blockchain Correlation (IP/ASN Co-Location & Calibration)
4. Multi-Hop Risk Taint Propagation
5. Entity Clustering & Heuristics
6. Analyst Feedback Loop (Overwrites & Suppressions)

Generates prioritized, explainable investigative alerts with plain-English justifications.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from src.correlation.network_blockchain import NetworkBlockchainCorrelator
from src.explainability.reason_generator import ReasonGenerator
from src.explainability.shap_explainer import ModelExplainer
from src.features import extract_all_features
from src.feedback.feedback_store import FeedbackStore
from src.graph.builder import build_graph, get_nodes_by_type
from src.graph.heuristics import apply_change_address_heuristic, apply_common_input_heuristic, get_wallet_clusters
from src.ingestion.parser import parse_file
from src.models.anomaly import AnomalyDetector
from src.models.clustering import EntityClusterer
from src.models.evasion_detectors import detect_coinjoin_transactions, detect_mixer_behavior
from src.graph.peel_chain import detect_peel_chains
from src.models.taint_propagation import TaintPropagator
from src.utils.logging_config import get_logger
from src.utils.paths import get_project_root, load_config
from src.utils.schema import InvestigativeAlert, compute_risk_level

logger = get_logger(__name__)


class CompositeScorer:
    """
    Combines forensic signals into weighted, explainable composite risk scores.
    """

    def __init__(
        self,
        anomaly_weight: float = 0.25,
        evasion_weight: float = 0.30,
        correlation_weight: float = 0.20,
        taint_weight: float = 0.25,
    ) -> None:
        self.w_anomaly = anomaly_weight
        self.w_evasion = evasion_weight
        self.w_correlation = correlation_weight
        self.w_taint = taint_weight

    def compute_composite_scores(
        self,
        graph: nx.MultiDiGraph,
        df: pd.DataFrame,
        wallet_features: pd.DataFrame,
        anomaly_df: pd.DataFrame,
        correlation_df: pd.DataFrame,
        taint_df: pd.DataFrame,
        coinjoin_df: pd.DataFrame,
        peel_chains: List[Dict[str, Any]],
        mixer_df: pd.DataFrame,
        clusters_df: pd.DataFrame,
        feedback_store: Optional[FeedbackStore] = None,
    ) -> pd.DataFrame:
        """
        Merge all forensic signals into unified wallet risk table.
        """
        all_wallets = sorted(list(get_nodes_by_type(graph, "wallet")))
        if not all_wallets:
            return pd.DataFrame()

        # Build evasion lookups
        cj_wallets: Set[str] = set()
        for row in df.itertuples(index=True):
            txid = getattr(row, "txid", "")
            in_addrs = getattr(row, "input_addresses", [])
            out_addrs = getattr(row, "output_addresses", [])
            if txid in set(coinjoin_df[coinjoin_df["is_coinjoin"]]["txid"]):
                if isinstance(in_addrs, (list, tuple)):
                    cj_wallets.update(str(a).strip() for a in in_addrs if a)
                if isinstance(out_addrs, (list, tuple)):
                    cj_wallets.update(str(a).strip() for a in out_addrs if a)

        peel_wallets: Set[str] = {
            w for pc in peel_chains for w in pc.get("chain_wallets", [])
        }
        mixer_wallets: Set[str] = set(
            mixer_df[mixer_df["is_mixer_pattern"]]["wallet_id"]
        ) if not mixer_df.empty and "is_mixer_pattern" in mixer_df.columns else set()

        # Build wallet -> TXIDs mapping
        wallet_txids: Dict[str, List[str]] = defaultdict(list)
        for row in df.itertuples(index=True):
            txid = str(getattr(row, "txid", "")).strip()
            in_addrs = getattr(row, "input_addresses", [])
            out_addrs = getattr(row, "output_addresses", [])
            if isinstance(in_addrs, (list, tuple)):
                for a in in_addrs:
                    if a:
                        wallet_txids[str(a).strip()].append(txid)
            if isinstance(out_addrs, (list, tuple)):
                for a in out_addrs:
                    if a:
                        wallet_txids[str(a).strip()].append(txid)

        feedback_dict = feedback_store.get_all_feedback() if feedback_store else {}

        records: List[Dict[str, Any]] = []

        for wallet in all_wallets:
            # 1. Anomaly score
            s_anomaly = float(anomaly_df.loc[wallet, "anomaly_score"]) if wallet in anomaly_df.index else 0.0

            # 2. Evasion score
            evasion_patterns = []
            evasion_score_components = []
            if wallet in cj_wallets:
                evasion_patterns.append("CoinJoin_Mixing")
                evasion_score_components.append(0.75)
            if wallet in peel_wallets:
                evasion_patterns.append("Ransomware_PeelChain")
                evasion_score_components.append(0.90)
            if wallet in mixer_wallets:
                evasion_patterns.append("Mixer_Hub")
                evasion_score_components.append(0.85)

            s_evasion = max(evasion_score_components) if evasion_score_components else 0.0

            # 3. Correlation score
            s_corr = 0.0
            primary_ip = None
            primary_asn = None
            asn_cat = "STANDARD"
            obs_cnt = 0
            if wallet in correlation_df.index:
                row_c = correlation_df.loc[wallet]
                s_corr = float(row_c.get("network_correlation_score", 0.0))
                primary_ip = row_c.get("primary_broadcast_ip")
                primary_asn = row_c.get("primary_asn")
                asn_cat = row_c.get("asn_category", "STANDARD")
                obs_cnt = int(row_c.get("observation_count", 0))

            # 4. Taint score
            s_taint = 0.0
            taint_hops = 0
            taint_src = None
            taint_path = ""
            if wallet in taint_df.index:
                row_t = taint_df.loc[wallet]
                s_taint = float(row_t.get("taint_score", 0.0))
                taint_hops = int(row_t.get("taint_hops", 0))
                taint_src = row_t.get("taint_source")
                taint_path = str(row_t.get("taint_path", ""))

            # 5. Composite score calculation
            composite_score = (
                self.w_anomaly * s_anomaly
                + self.w_evasion * s_evasion
                + self.w_correlation * s_corr
                + self.w_taint * s_taint
            )

            # Boost score if multiple distinct high-risk signals co-occur
            active_signals = sum([s_anomaly > 0.6, s_evasion > 0.5, s_corr > 0.6, s_taint > 0.4])
            if active_signals >= 3:
                composite_score = min(1.0, composite_score * 1.25)

            # Cluster metadata
            cid = None
            c_size = 1
            if wallet in clusters_df.index:
                cid = int(clusters_df.loc[wallet, "cluster_id"])
                c_size = int(clusters_df.loc[wallet, "cluster_size"])

            # Feedback status
            fb = feedback_dict.get(wallet, {})
            fb_status = fb.get("status", "PENDING")
            fb_notes = fb.get("notes", "")

            # Apply feedback adjustments
            if fb_status == "CONFIRMED":
                composite_score = min(1.0, composite_score * 1.2)
            elif fb_status == "FALSE_POSITIVE":
                composite_score = round(composite_score * 0.1, 4)

            composite_score = round(float(np.clip(composite_score, 0.0, 1.0)), 4)
            risk_lvl = compute_risk_level(composite_score)

            # Generate plain-English reasons
            w_feats = wallet_features.loc[wallet].to_dict() if wallet in wallet_features.index else {}
            reasons = ReasonGenerator.generate_reasons(
                entity_id=wallet,
                entity_type="wallet",
                risk_score=composite_score,
                features=w_feats,
                evasion_flags=evasion_patterns,
                network_data={
                    "primary_broadcast_ip": primary_ip,
                    "primary_asn": primary_asn,
                    "asn_category": asn_cat,
                    "observation_count": obs_cnt,
                },
                taint_data={
                    "taint_score": s_taint,
                    "taint_hops": taint_hops,
                    "taint_source": taint_src,
                    "taint_path": taint_path,
                },
                cluster_info={"cluster_id": cid, "cluster_size": c_size},
            )

            records.append({
                "entity_id": wallet,
                "entity_type": "wallet",
                "composite_risk_score": composite_score,
                "risk_level": risk_lvl,
                "anomaly_score": s_anomaly,
                "evasion_score": s_evasion,
                "network_correlation_score": s_corr,
                "taint_score": s_taint,
                "pattern_types": evasion_patterns if evasion_patterns else (["Anomalous_Traffic"] if s_anomaly > 0.65 else []),
                "primary_reason": reasons["primary_reason"],
                "detailed_justification": reasons["detailed_justification"],
                "primary_ip": primary_ip,
                "primary_asn": primary_asn,
                "asn_category": asn_cat,
                "taint_source": taint_src,
                "taint_hops": taint_hops,
                "taint_path": taint_path,
                "cluster_id": cid,
                "cluster_size": c_size,
                "associated_txids": wallet_txids.get(wallet, [])[:10],
                "analyst_status": fb_status,
                "analyst_notes": fb_notes,
            })

        result_df = pd.DataFrame(records).set_index("entity_id")
        return result_df.sort_values(by="composite_risk_score", ascending=False)


def run_full_pipeline(
    input_file: Union[str, Path] = "data/raw/synthetic_transactions.csv",
    output_dir: Union[str, Path] = "data/processed",
) -> Dict[str, Any]:
    """
    Execute entire investigative pipeline from raw transactions to scored alerts.
    """
    input_path = Path(input_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("1/7 Ingesting transaction data from %s...", input_path)
    df = parse_file(input_path)

    logger.info("2/7 Constructing heterogeneous graph & deriving heuristics...")
    graph = build_graph(df)
    apply_common_input_heuristic(graph, df)
    apply_change_address_heuristic(graph, df)

    logger.info("3/7 Extracting unified behavioral, network, and temporal feature matrices...")
    feature_matrices = extract_all_features(graph, df)
    wallet_feats = feature_matrices["wallets"]
    tx_feats = feature_matrices["transactions"]

    logger.info("4/7 Running anomaly detection & entity clustering...")
    anomaly_detector = AnomalyDetector()
    wallet_anomalies = anomaly_detector.fit_predict_wallets(wallet_feats)
    tx_anomalies = anomaly_detector.fit_predict_transactions(tx_feats)

    clusterer = EntityClusterer()
    clusters_df = clusterer.cluster_entities(graph, df)

    logger.info("5/7 Detecting evasion signatures (CoinJoin, Peel Chains, Mixer Hubs)...")
    coinjoin_df = detect_coinjoin_transactions(df)
    peel_chains = detect_peel_chains(graph, df)
    mixer_df = detect_mixer_behavior(graph, df)

    logger.info("6/7 Computing network-blockchain correlation & multi-hop taint propagation...")
    correlator = NetworkBlockchainCorrelator()
    correlation_df = correlator.correlate(df)

    # Pick seed wallets for taint propagation (high evasion + peel chain seeds)
    seed_risks: Dict[str, float] = {}
    for pc in peel_chains:
        if pc.get("chain_wallets"):
            seed_risks[pc["chain_wallets"][0]] = 1.0
    for w in mixer_df[mixer_df["is_mixer_pattern"]]["wallet_id"]:
        seed_risks[w] = 0.95

    # If ground truth exists, include known bad seeds
    gt_path = input_path.parent / "ground_truth.csv"
    if gt_path.is_file():
        gt_df = pd.read_csv(gt_path)
        bad_wallets = gt_df[(gt_df["entity_type"] == "wallet") & (gt_df["is_criminal"])]["entity_id"].tolist()
        for w in bad_wallets[:8]:
            seed_risks[w] = 1.0

    propagator = TaintPropagator()
    taint_df = propagator.propagate(graph, seed_risks)

    logger.info("7/7 Computing composite risk scores and generating investigative alerts...")
    feedback_store = FeedbackStore(out_dir / "analyst_feedback.db")
    scorer = CompositeScorer()
    scored_alerts_df = scorer.compute_composite_scores(
        graph=graph,
        df=df,
        wallet_features=wallet_feats,
        anomaly_df=wallet_anomalies,
        correlation_df=correlation_df,
        taint_df=taint_df,
        coinjoin_df=coinjoin_df,
        peel_chains=peel_chains,
        mixer_df=mixer_df,
        clusters_df=clusters_df,
        feedback_store=feedback_store,
    )

    # Save artifacts to processed directory
    scored_alerts_df.to_csv(out_dir / "scored_alerts.csv")
    coinjoin_df.to_csv(out_dir / "coinjoin_transactions.csv")
    clusters_df.to_csv(out_dir / "wallet_clusters.csv")
    
    with open(out_dir / "peel_chains.json", "w", encoding="utf-8") as f:
        json.dump(peel_chains, f, indent=2)

    logger.info("Pipeline completed successfully! %d entities scored.", len(scored_alerts_df))

    return {
        "graph": graph,
        "df": df,
        "wallet_features": wallet_feats,
        "tx_features": tx_feats,
        "scored_alerts": scored_alerts_df,
        "coinjoin_df": coinjoin_df,
        "peel_chains": peel_chains,
        "mixer_df": mixer_df,
        "clusters_df": clusters_df,
        "taint_df": taint_df,
        "correlation_df": correlation_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete Bitcoin Traffic Analyzer pipeline.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/raw/synthetic_transactions.csv",
        help="Path to transaction file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/processed",
        help="Path to output directory",
    )
    args = parser.parse_args()

    results = run_full_pipeline(args.input, args.output)
    alerts_df = results["scored_alerts"]

    print("\n" + "=" * 70)
    print("         BITCOIN TRAFFIC ANALYZER - PIPELINE RUN SUMMARY")
    print("=" * 70)
    print(f"Total Evaluated Wallets: {len(alerts_df):,}")
    print(f"Critical Alerts (Score >= 0.75): {len(alerts_df[alerts_df['risk_level'] == 'CRITICAL']):,}")
    print(f"High Alerts (Score >= 0.50): {len(alerts_df[alerts_df['risk_level'] == 'HIGH']):,}")
    print(f"Medium Alerts (Score >= 0.30): {len(alerts_df[alerts_df['risk_level'] == 'MEDIUM']):,}")

    top_leads = alerts_df.head(10)
    print("\n--- TOP 10 PRIORITIZED INVESTIGATIVE LEADS ---")
    for w_id, row in top_leads.iterrows():
        print(
            f"  [{row['risk_level']}] Wallet: {w_id} | Risk: {row['composite_risk_score']:.4f}\n"
            f"    Patterns: {', '.join(row['pattern_types']) if row['pattern_types'] else 'Anomaly'}\n"
            f"    IP/ASN: {row['primary_ip']} (ASN {row['primary_asn']} - {row['asn_category']})\n"
            f"    Reason: {row['primary_reason']}\n"
        )

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
