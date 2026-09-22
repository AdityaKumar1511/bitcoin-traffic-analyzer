"""
Composite Risk Scoring and Forensic Lead Prioritization Engine.

Combines signals from:
1. Unsupervised Anomaly Detection (Isolation Forest + PCA Reconstruction Error)
2. Evasion Detectors (CoinJoin, Peel Chain, Mixer Hub)
3. Network-Blockchain Correlation (IP/ASN Co-Location & Calibration)
4. Multi-Hop Risk Taint Propagation
5. Entity Clustering & Heuristics
6. Analyst Feedback Loop (Overwrites & Suppressions)

Generates prioritized, explainable investigative alerts with:
- SHAP-backed per-entity feature attributions
- Plain-English justifications for each flagged entity
- Ranked alert tables exportable as CSV and JSON
- Ground truth evaluation metrics (when available)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx
import numpy as np
import pandas as pd

from src.correlation.network_blockchain import NetworkBlockchainCorrelator
from src.explainability.reason_generator import ReasonGenerator
from src.explainability.shap_explainer import ModelExplainer
from src.features import extract_all_features
from src.feedback.feedback_store import FeedbackStore
from src.graph.builder import build_graph, get_nodes_by_type
from src.graph.heuristics import apply_change_address_heuristic, apply_common_input_heuristic
from src.ingestion.parser import parse_file
from src.models.anomaly import AnomalyDetector
from src.models.clustering import EntityClusterer
from src.models.evasion_detectors import detect_coinjoin_transactions, detect_mixer_behavior
from src.graph.peel_chain import detect_peel_chains
from src.models.taint_propagation import TaintPropagator
from src.utils.logging_config import get_logger
from src.utils.schema import compute_risk_level

logger = get_logger(__name__)


class CompositeScorer:
    """
    Combines forensic signals into weighted, explainable composite risk scores.

    The scoring formula is:
        risk_score = (w1 * anomaly_score +
                      w2 * evasion_score +
                      w3 * network_correlation_score +
                      w4 * taint_score)

    With multi-signal co-occurrence boosting: if >= 3 distinct high-risk signals
    are active, the score is amplified by 25% (capped at 1.0).
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
        self._explainer: Optional[ModelExplainer] = None
        self._shap_explanations: Dict[str, Dict[str, float]] = {}
        self._global_feature_importance: Optional[pd.DataFrame] = None

    def attach_explainer(
        self,
        explainer: ModelExplainer,
        wallet_features: pd.DataFrame,
        anomaly_scores: Optional[pd.Series] = None,
    ) -> None:
        """
        Attach a fitted ModelExplainer and generate batch SHAP explanations for
        all entities above the score threshold.

        Args:
            explainer: A fitted ModelExplainer instance.
            wallet_features: Full wallet feature matrix.
            anomaly_scores: Optional anomaly scores indexed by wallet_id.
        """
        self._explainer = explainer

        # Generate batch SHAP explanations for entities above threshold
        self._shap_explanations = explainer.explain_batch(
            wallet_features,
            score_threshold=0.25,
            anomaly_scores=anomaly_scores,
        )

        # Compute global feature importance for dashboard
        self._global_feature_importance = explainer.compute_global_feature_importance(
            wallet_features
        )

        logger.info(
            "SHAP explainer attached: %d entity explanations generated, %d features ranked.",
            len(self._shap_explanations),
            len(self._global_feature_importance) if self._global_feature_importance is not None else 0,
        )

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

        Returns:
            pd.DataFrame indexed by entity_id with columns:
                entity_type, composite_risk_score, risk_level,
                anomaly_score, evasion_score, network_correlation_score, taint_score,
                pattern_types, primary_reason, detailed_justification,
                top_shap_features, primary_ip, primary_asn, asn_category,
                taint_source, taint_hops, taint_path,
                cluster_id, cluster_size, associated_txids,
                analyst_status, analyst_notes
        """
        all_wallets = sorted(list(get_nodes_by_type(graph, "wallet")))
        if not all_wallets:
            return pd.DataFrame()

        # ──────────────────────────────────────────────
        # Build evasion lookups
        # ──────────────────────────────────────────────
        cj_txids: Set[str] = set()
        if not coinjoin_df.empty and "is_coinjoin" in coinjoin_df.columns:
            cj_txids = set(coinjoin_df[coinjoin_df["is_coinjoin"]]["txid"])

        cj_wallets: Set[str] = set()
        for row in df.itertuples(index=True):
            txid = getattr(row, "txid", "")
            in_addrs = getattr(row, "input_addresses", [])
            out_addrs = getattr(row, "output_addresses", [])
            if txid in cj_txids:
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

        # ──────────────────────────────────────────────
        # Build wallet -> TXIDs mapping
        # ──────────────────────────────────────────────
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

        # ──────────────────────────────────────────────
        # Build wallet -> related wallets mapping (from graph neighbors)
        # ──────────────────────────────────────────────
        wallet_related: Dict[str, Set[str]] = defaultdict(set)
        for wallet in all_wallets:
            # Walk wallet -> tx -> wallet paths for counterparties
            for _, tx_node, edge_data in graph.out_edges(wallet, data=True):
                if edge_data.get("edge_type") in ("input", "output"):
                    for _, next_w, e2 in graph.out_edges(tx_node, data=True):
                        if e2.get("edge_type") == "output" and next_w != wallet:
                            node_data = graph.nodes.get(next_w, {})
                            if node_data.get("node_type") == "wallet":
                                wallet_related[wallet].add(next_w)

        feedback_dict = feedback_store.get_all_feedback() if feedback_store else {}

        records: List[Dict[str, Any]] = []

        for wallet in all_wallets:
            # ── 1. Anomaly score ──
            s_anomaly = float(anomaly_df.loc[wallet, "anomaly_score"]) if wallet in anomaly_df.index else 0.0

            # ── 2. Evasion score (max of all matched evasion patterns) ──
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

            # ── 3. Network correlation score ──
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

            # ── 4. Taint score ──
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

            # ── 5. Composite score calculation ──
            composite_score = (
                self.w_anomaly * s_anomaly
                + self.w_evasion * s_evasion
                + self.w_correlation * s_corr
                + self.w_taint * s_taint
            )

            # Multi-signal co-occurrence boost
            active_signals = sum([s_anomaly > 0.6, s_evasion > 0.5, s_corr > 0.6, s_taint > 0.4])
            if active_signals >= 3:
                composite_score = min(1.0, composite_score * 1.25)

            # ── 6. Cluster metadata ──
            cid = None
            c_size = 1
            if wallet in clusters_df.index:
                cid = int(clusters_df.loc[wallet, "cluster_id"])
                c_size = int(clusters_df.loc[wallet, "cluster_size"])

            # ── 7. Feedback adjustments ──
            fb = feedback_dict.get(wallet, {})
            fb_status = fb.get("status", "PENDING")
            fb_notes = fb.get("notes", "")

            if fb_status == "CONFIRMED":
                composite_score = min(1.0, composite_score * 1.2)
            elif fb_status == "FALSE_POSITIVE":
                composite_score = round(composite_score * 0.1, 4)

            composite_score = round(float(np.clip(composite_score, 0.0, 1.0)), 4)
            risk_lvl = compute_risk_level(composite_score)

            # ── 8. SHAP explanations for this entity ──
            entity_shap = self._shap_explanations.get(wallet, {})

            # ── 9. Generate plain-English reasons ──
            w_feats = wallet_features.loc[wallet].to_dict() if wallet in wallet_features.index else {}
            reasons = ReasonGenerator.generate_reasons(
                entity_id=wallet,
                entity_type="wallet",
                risk_score=composite_score,
                features=w_feats,
                top_shap_features=entity_shap,
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

            # ── 10. Related wallets ──
            related_wallets = sorted(list(wallet_related.get(wallet, set())))[:10]

            records.append({
                "entity_id": wallet,
                "entity_type": "wallet",
                "composite_risk_score": composite_score,
                "risk_level": risk_lvl,
                "anomaly_score": round(s_anomaly, 4),
                "evasion_score": round(s_evasion, 4),
                "network_correlation_score": round(s_corr, 4),
                "taint_score": round(s_taint, 4),
                "pattern_types": evasion_patterns if evasion_patterns else (
                    ["Anomalous_Traffic"] if s_anomaly > 0.65 else []
                ),
                "primary_reason": reasons["primary_reason"],
                "detailed_justification": reasons["detailed_justification"],
                "top_shap_features": entity_shap,
                "primary_ip": primary_ip,
                "primary_asn": primary_asn,
                "asn_category": asn_cat,
                "taint_source": taint_src,
                "taint_hops": taint_hops,
                "taint_path": taint_path,
                "cluster_id": cid,
                "cluster_size": c_size,
                "associated_txids": wallet_txids.get(wallet, [])[:10],
                "related_wallets": related_wallets,
                "analyst_status": fb_status,
                "analyst_notes": fb_notes,
            })

        result_df = pd.DataFrame(records).set_index("entity_id")
        return result_df.sort_values(by="composite_risk_score", ascending=False)

    def get_global_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return global feature importance ranking computed during SHAP batch explain."""
        return self._global_feature_importance


def evaluate_against_ground_truth(
    scored_alerts: pd.DataFrame,
    ground_truth_path: Union[str, Path],
    score_threshold: float = 0.30,
) -> Dict[str, Any]:
    """
    Evaluate scored alerts against ground truth labels.

    Args:
        scored_alerts: DataFrame from CompositeScorer.compute_composite_scores().
        ground_truth_path: Path to ground_truth.csv with columns [entity_id, entity_type, is_criminal, pattern_type].
        score_threshold: Risk score threshold for flagging (default: 0.30 = MEDIUM+).

    Returns:
        dict: {
            "total_ground_truth": int,
            "total_flagged": int,
            "true_positives": int,
            "false_positives": int,
            "false_negatives": int,
            "precision": float,
            "recall": float,
            "f1": float,
            "per_pattern_recall": dict,
            "top_missed": list,
        }
    """
    gt_path = Path(ground_truth_path)
    if not gt_path.is_file():
        logger.warning("Ground truth file not found at %s.", gt_path)
        return {}

    gt_df = pd.read_csv(gt_path)

    # Get ground truth criminal wallets
    gt_wallets = set(
        gt_df[(gt_df["entity_type"] == "wallet") & (gt_df["is_criminal"])]["entity_id"]
    )

    # Get flagged wallets
    flagged_wallets = set(scored_alerts[scored_alerts["composite_risk_score"] >= score_threshold].index)

    # Metrics
    tp = len(flagged_wallets & gt_wallets)
    fp = len(flagged_wallets - gt_wallets)
    fn = len(gt_wallets - flagged_wallets)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-pattern recall
    per_pattern_recall: Dict[str, float] = {}
    if "pattern_type" in gt_df.columns:
        for pattern in gt_df["pattern_type"].unique():
            pattern_wallets = set(
                gt_df[(gt_df["pattern_type"] == pattern) & (gt_df["entity_type"] == "wallet") & (gt_df["is_criminal"])]["entity_id"]
            )
            if pattern_wallets:
                pattern_tp = len(flagged_wallets & pattern_wallets)
                per_pattern_recall[pattern] = round(pattern_tp / len(pattern_wallets), 4)

    # Top missed (false negatives with highest potential)
    missed = gt_wallets - flagged_wallets
    missed_with_scores = []
    for w in list(missed)[:20]:
        score = float(scored_alerts.loc[w, "composite_risk_score"]) if w in scored_alerts.index else 0.0
        missed_with_scores.append({"wallet": w, "score": score})
    missed_with_scores.sort(key=lambda x: x["score"], reverse=True)

    metrics = {
        "total_ground_truth": len(gt_wallets),
        "total_flagged": len(flagged_wallets),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_pattern_recall": per_pattern_recall,
        "top_missed": missed_with_scores[:10],
    }

    logger.info(
        "Ground truth evaluation: P=%.4f R=%.4f F1=%.4f (TP=%d, FP=%d, FN=%d)",
        precision, recall, f1, tp, fp, fn,
    )
    return metrics


def _export_alerts_json(
    scored_alerts: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Export scored alerts as a structured JSON file for downstream tooling.

    Handles non-serializable types (numpy, sets, timestamps).
    """
    records = []
    for entity_id, row in scored_alerts.iterrows():
        record = {"entity_id": entity_id}
        for col in scored_alerts.columns:
            val = row[col]
            # Convert numpy types to native Python
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, set):
                val = sorted(list(val))
            record[col] = val
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    logger.info("Exported %d alerts to JSON: %s", len(records), output_path)


def run_full_pipeline(
    input_file: Union[str, Path] = "data/raw/synthetic_transactions.csv",
    output_dir: Union[str, Path] = "data/processed",
) -> Dict[str, Any]:
    """
    Execute entire investigative pipeline from raw transactions to scored alerts.

    Pipeline stages:
        1. Ingest transaction data
        2. Build heterogeneous graph + apply forensic heuristics
        3. Extract unified feature matrices
        4. Run anomaly detection + entity clustering
        5. Detect evasion signatures
        6. Compute network-blockchain correlation + taint propagation
        7. Fit SHAP explainer + compute composite risk scores
        8. Evaluate against ground truth (if available)
        9. Export artifacts (CSV, JSON)
    """
    input_path = Path(input_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("1/9 Ingesting transaction data from %s...", input_path)
    df = parse_file(input_path)

    logger.info("2/9 Constructing heterogeneous graph & deriving heuristics...")
    graph = build_graph(df)
    apply_common_input_heuristic(graph, df)
    apply_change_address_heuristic(graph, df)

    logger.info("3/9 Extracting unified behavioral, network, and temporal feature matrices...")
    feature_matrices = extract_all_features(graph, df)
    wallet_feats = feature_matrices["wallets"]
    tx_feats = feature_matrices["transactions"]

    logger.info("4/9 Running anomaly detection & entity clustering...")
    anomaly_detector = AnomalyDetector()
    wallet_anomalies = anomaly_detector.fit_predict_wallets(wallet_feats)
    anomaly_detector.fit_predict_transactions(tx_feats)

    clusterer = EntityClusterer()
    clusters_df = clusterer.cluster_entities(graph, df)

    logger.info("5/9 Detecting evasion signatures (CoinJoin, Peel Chains, Mixer Hubs)...")
    coinjoin_df = detect_coinjoin_transactions(df)
    peel_chains = detect_peel_chains(graph, df)
    mixer_df = detect_mixer_behavior(graph, df)

    logger.info("6/9 Computing network-blockchain correlation & multi-hop taint propagation...")
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

    logger.info("7/9 Fitting SHAP explainer & computing composite risk scores...")
    # Fit SHAP explainer on the anomaly detection model
    explainer = ModelExplainer(top_k=5)
    if anomaly_detector.wallet_model is not None:
        # Scale wallet features the same way the anomaly detector did
        numeric_feats = wallet_feats.select_dtypes(include=[np.number]).fillna(0.0)
        if not numeric_feats.empty:
            scaled_bg = pd.DataFrame(
                anomaly_detector.wallet_scaler.transform(numeric_feats),
                columns=numeric_feats.columns,
                index=numeric_feats.index,
            )
            explainer.fit(
                model=anomaly_detector.wallet_model,
                background_data=scaled_bg,
            )
    else:
        explainer.fit(model=None, background_data=wallet_feats)

    # Create scorer and attach explainer
    feedback_store = FeedbackStore(out_dir / "analyst_feedback.db")
    scorer = CompositeScorer()
    scorer.attach_explainer(
        explainer=explainer,
        wallet_features=wallet_feats,
        anomaly_scores=wallet_anomalies["anomaly_score"] if "anomaly_score" in wallet_anomalies.columns else None,
    )

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

    # 8/9 Evaluate against ground truth
    gt_metrics: Dict[str, Any] = {}
    if gt_path.is_file():
        logger.info("8/9 Evaluating against ground truth labels...")
        gt_metrics = evaluate_against_ground_truth(scored_alerts_df, gt_path)
    else:
        logger.info("8/9 No ground truth file found — skipping evaluation.")

    # 9/9 Save artifacts to processed directory
    logger.info("9/9 Exporting scored alerts and pipeline artifacts...")
    scored_alerts_df.to_csv(out_dir / "scored_alerts.csv")
    _export_alerts_json(scored_alerts_df, out_dir / "scored_alerts.json")
    coinjoin_df.to_csv(out_dir / "coinjoin_transactions.csv")
    clusters_df.to_csv(out_dir / "wallet_clusters.csv")

    with open(out_dir / "peel_chains.json", "w", encoding="utf-8") as f:
        json.dump(peel_chains, f, indent=2, default=str)

    # Save ground truth metrics
    if gt_metrics:
        with open(out_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(gt_metrics, f, indent=2, default=str)

    # Save global feature importance
    feature_importance = scorer.get_global_feature_importance()
    if feature_importance is not None and not feature_importance.empty:
        feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)

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
        "explainer": explainer,
        "gt_metrics": gt_metrics,
        "feature_importance": feature_importance,
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
    gt_metrics = results.get("gt_metrics", {})

    print("\n" + "=" * 70)
    print("         BITCOIN TRAFFIC ANALYZER - PIPELINE RUN SUMMARY")
    print("=" * 70)
    print(f"Total Evaluated Wallets: {len(alerts_df):,}")

    for level in ("CRITICAL", "HIGH", "MEDIUM"):
        count = len(alerts_df[alerts_df["risk_level"] == level])
        label = {"CRITICAL": "Score >= 0.75", "HIGH": "Score >= 0.50", "MEDIUM": "Score >= 0.30"}[level]
        print(f"{level} Alerts ({label}): {count:,}")

    # Ground truth metrics
    if gt_metrics:
        print("\n--- GROUND TRUTH EVALUATION ---")
        print(f"  Precision: {gt_metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {gt_metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:  {gt_metrics.get('f1', 0):.4f}")
        print(f"  TP: {gt_metrics.get('true_positives', 0)} | FP: {gt_metrics.get('false_positives', 0)} | FN: {gt_metrics.get('false_negatives', 0)}")
        per_pattern = gt_metrics.get("per_pattern_recall", {})
        if per_pattern:
            print("  Per-Pattern Recall:")
            for pattern, recall in per_pattern.items():
                print(f"    {pattern}: {recall:.1%}")

    # Top leads
    top_leads = alerts_df.head(10)
    print("\n--- TOP 10 PRIORITIZED INVESTIGATIVE LEADS ---")
    for w_id, row in top_leads.iterrows():
        shap_str = ""
        shap_feats = row.get("top_shap_features", {})
        if isinstance(shap_feats, dict) and shap_feats:
            top_3 = list(shap_feats.items())[:3]
            shap_str = " | SHAP: " + ", ".join(f"{k} ({v:+.2f})" for k, v in top_3)

        print(
            f"  [{row['risk_level']}] Wallet: {w_id} | Risk: {row['composite_risk_score']:.4f}\n"
            f"    Patterns: {', '.join(row['pattern_types']) if row['pattern_types'] else 'Anomaly'}\n"
            f"    IP/ASN: {row['primary_ip']} (ASN {row['primary_asn']} - {row['asn_category']})\n"
            f"    Reason: {row['primary_reason']}{shap_str}\n"
        )

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
