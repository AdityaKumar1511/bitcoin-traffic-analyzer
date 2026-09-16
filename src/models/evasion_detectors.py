"""
Evasion and Money Laundering Detection Module.

Detects advanced blockchain obfuscation and laundering patterns:
1. CoinJoin-Style Mixing:
   High-fan-in, high-fan-out transactions where participants receive standardized,
   equal-value output denominations.
2. Peel Chains:
   Sequential forwarding where a carrier wallet repeatedly passes on the majority
   balance while peeling off small portions (reusing src.graph.peel_chain).
3. Mixer Hub Behavior:
   Wallets exhibiting high-throughput pass-through activity (receiving from >= N sources
   and sending to >= M destinations within a tight rolling time window).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from src.graph.builder import build_graph
from src.graph.peel_chain import detect_peel_chains
from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger
from src.utils.paths import load_config

logger = get_logger(__name__)


def detect_coinjoin_transactions(
    df: pd.DataFrame,
    min_inputs: int = 5,
    min_outputs: int = 5,
    amount_similarity_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Detect CoinJoin-style mixing transactions based on input/output fan-out and equal denominations.

    Identifies transactions with >= min_inputs and >= min_outputs where a majority (>= 60%)
    of the output amounts are within `amount_similarity_threshold` of a dominant value.

    Args:
        df: Normalized transaction DataFrame.
        min_inputs: Minimum number of input addresses required (default: 5).
        min_outputs: Minimum number of output addresses required (default: 5).
        amount_similarity_threshold: Relative fractional tolerance for output amount
            clustering (default: 0.05, meaning +/-5%).

    Returns:
        pd.DataFrame with columns:
            txid, num_inputs, num_outputs, dominant_output_amount,
            pct_outputs_matching, is_coinjoin
    """
    cfg = load_config().get("evasion_detectors", {}).get("coinjoin", {})
    min_inputs = cfg.get("min_inputs", min_inputs)
    min_outputs = cfg.get("min_outputs", min_outputs)
    amount_similarity_threshold = cfg.get("amount_similarity_threshold", amount_similarity_threshold)

    records: List[Dict[str, Any]] = []

    for row in df.itertuples(index=True):
        txid = getattr(row, "txid", "")
        in_addrs = getattr(row, "input_addresses", [])
        out_amts = getattr(row, "output_amounts", [])

        n_inputs = len(in_addrs) if isinstance(in_addrs, (list, tuple)) else 0
        n_outputs = len(out_amts) if isinstance(out_amts, (list, tuple)) else 0

        dominant_amt = 0.0
        pct_matching = 0.0
        is_coinjoin = False

        if n_inputs >= min_inputs and n_outputs >= min_outputs:
            # Parse numeric output amounts
            float_amts: List[float] = []
            for a in out_amts:
                try:
                    val = float(a)
                    if val > 0:
                        float_amts.append(val)
                except (ValueError, TypeError):
                    continue

            if len(float_amts) >= min_outputs:
                # Find dominant cluster of amounts within tolerance
                best_count = 0
                best_target = 0.0

                for candidate in float_amts:
                    tolerance = candidate * amount_similarity_threshold
                    matches = [x for x in float_amts if abs(x - candidate) <= tolerance]
                    if len(matches) > best_count:
                        best_count = len(matches)
                        best_target = candidate

                dominant_amt = round(best_target, 8)
                pct_matching = round(best_count / n_outputs, 4)

                # Flag if a significant majority (>= 60%) of outputs match the dominant amount
                if pct_matching >= 0.60:
                    is_coinjoin = True

        records.append({
            "txid": str(txid).strip(),
            "num_inputs": n_inputs,
            "num_outputs": n_outputs,
            "dominant_output_amount": dominant_amt,
            "pct_outputs_matching": pct_matching,
            "is_coinjoin": is_coinjoin,
        })

    result_df = pd.DataFrame(records)
    flagged_count = int(result_df["is_coinjoin"].sum())
    logger.info("CoinJoin detection complete: flagged %d of %d transactions.", flagged_count, len(result_df))
    return result_df


def detect_mixer_behavior(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
    time_window_minutes: int = 60,
    min_sources: int = 4,
    min_destinations: int = 4,
) -> pd.DataFrame:
    """
    Detect mixer/hub pass-through behavior on wallet nodes within a rolling time window.

    Flags wallets that both receive from >= min_sources distinct wallets and send to
    >= min_destinations distinct wallets within any time_window_minutes window.

    Args:
        graph: Heterogeneous transaction MultiDiGraph.
        df: Normalized transaction DataFrame.
        time_window_minutes: Rolling time window duration in minutes (default: 60).
        min_sources: Minimum distinct receiving counterparties in window (default: 4).
        min_destinations: Minimum distinct sending counterparties in window (default: 4).

    Returns:
        pd.DataFrame with columns:
            wallet_id, time_window_start, distinct_sources,
            distinct_destinations, is_mixer_pattern
    """
    cfg = load_config().get("evasion_detectors", {}).get("mixer", {})
    time_window_minutes = cfg.get("time_window_minutes", time_window_minutes)
    min_sources = cfg.get("min_sources", min_sources)
    min_destinations = cfg.get("min_destinations", min_destinations)

    # Pre-parse timestamps and wallet linkages
    incoming_events: Dict[str, List[Tuple[pd.Timestamp, str]]] = defaultdict(list)
    outgoing_events: Dict[str, List[Tuple[pd.Timestamp, str]]] = defaultdict(list)

    total_sources: Dict[str, Set[str]] = defaultdict(set)
    total_destinations: Dict[str, Set[str]] = defaultdict(set)

    for row in df.itertuples(index=True):
        ts = getattr(row, "timestamp", None)
        in_addrs = getattr(row, "input_addresses", [])
        out_addrs = getattr(row, "output_addresses", [])

        if not isinstance(in_addrs, (list, tuple)) or not isinstance(out_addrs, (list, tuple)):
            continue

        clean_ins = [str(a).strip() for a in in_addrs if a and str(a).strip()]
        clean_outs = [str(a).strip() for a in out_addrs if a and str(a).strip()]

        if not clean_ins or not clean_outs:
            continue

        ts_val = pd.to_datetime(ts, utc=True) if ts is not None else pd.Timestamp.min

        # For every output wallet, it receives from all input wallets
        for w_out in clean_outs:
            for w_in in clean_ins:
                if w_in != w_out:
                    incoming_events[w_out].append((ts_val, w_in))
                    total_sources[w_out].add(w_in)

        # For every input wallet, it sends to all output wallets
        for w_in in clean_ins:
            for w_out in clean_outs:
                if w_in != w_out:
                    outgoing_events[w_in].append((ts_val, w_out))
                    total_destinations[w_in].add(w_out)

    # Filter candidate wallets: must have at least min_sources and min_destinations overall
    candidate_wallets = [
        w for w in set(incoming_events.keys()).intersection(outgoing_events.keys())
        if len(total_sources[w]) >= min_sources and len(total_destinations[w]) >= min_destinations
    ]

    window_delta = pd.Timedelta(minutes=time_window_minutes)
    records: List[Dict[str, Any]] = []

    for wallet in candidate_wallets:
        in_evs = incoming_events[wallet]
        out_evs = outgoing_events[wallet]

        # Collect and sort all timestamps where this wallet had activity
        timestamps = sorted({t for t, _ in in_evs}.union({t for t, _ in out_evs}))

        best_sources = 0
        best_destinations = 0
        best_window_start: Optional[pd.Timestamp] = None
        is_mixer = False

        for t_start in timestamps:
            t_end = t_start + window_delta
            window_srcs = {src for t, src in in_evs if t_start <= t <= t_end}
            window_dsts = {dst for t, dst in out_evs if t_start <= t <= t_end}

            if len(window_srcs) >= min_sources and len(window_dsts) >= min_destinations:
                is_mixer = True
                best_sources = len(window_srcs)
                best_destinations = len(window_dsts)
                best_window_start = t_start
                break
            elif len(window_srcs) + len(window_dsts) > best_sources + best_destinations:
                best_sources = len(window_srcs)
                best_destinations = len(window_dsts)
                best_window_start = t_start

        records.append({
            "wallet_id": wallet,
            "time_window_start": best_window_start,
            "distinct_sources": best_sources,
            "distinct_destinations": best_destinations,
            "is_mixer_pattern": is_mixer,
        })

    result_df = pd.DataFrame(records)
    if result_df.empty:
        result_df = pd.DataFrame(
            columns=[
                "wallet_id",
                "time_window_start",
                "distinct_sources",
                "distinct_destinations",
                "is_mixer_pattern",
            ]
        )

    flagged_count = int(result_df["is_mixer_pattern"].sum()) if not result_df.empty else 0
    logger.info("Mixer hub detection complete: flagged %d wallets.", flagged_count)
    return result_df


def run_all_evasion_detectors(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Execute all evasion detection strategies across the graph and transactions.

    Runs:
    1. CoinJoin-style mixing detector
    2. Peel chain detector (reusing src.graph.peel_chain)
    3. Mixer hub pattern detector

    Args:
        graph: Heterogeneous transaction MultiDiGraph.
        df: Normalized transaction DataFrame.

    Returns:
        dict: {
            "coinjoin_df": pd.DataFrame,
            "peel_chains": list[dict],
            "mixer_df": pd.DataFrame,
            "summary": {
                "coinjoin_transactions_flagged": int,
                "peel_chains_found": int,
                "mixer_wallets_flagged": int,
            }
        }
    """
    logger.info("Starting comprehensive evasion detector suite...")

    coinjoin_df = detect_coinjoin_transactions(df)
    peel_chains = detect_peel_chains(graph, df)
    mixer_df = detect_mixer_behavior(graph, df)

    coinjoin_flagged = int(coinjoin_df["is_coinjoin"].sum()) if not coinjoin_df.empty else 0
    peel_count = len(peel_chains)
    mixer_flagged = int(mixer_df["is_mixer_pattern"].sum()) if not mixer_df.empty else 0

    return {
        "coinjoin_df": coinjoin_df,
        "peel_chains": peel_chains,
        "mixer_df": mixer_df,
        "summary": {
            "coinjoin_transactions_flagged": coinjoin_flagged,
            "peel_chains_found": peel_count,
            "mixer_wallets_flagged": mixer_flagged,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Bitcoin evasion detectors (CoinJoin, Peel Chains, Mixer Hubs)."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input transaction file (.csv, .json, or .xml).",
    )
    return parser


def main() -> None:
    """CLI entrypoint for standalone evasion detector evaluation."""
    args = _build_parser().parse_args()

    print(f"[*] Parsing input transactions: {args.input}")
    df = parse_file(args.input)

    print(f"[*] Building graph from {len(df):,} transactions...")
    graph = build_graph(df)

    print("[*] Running evasion detectors (CoinJoin, Peel Chains, Mixer Hubs)...")
    results = run_all_evasion_detectors(graph, df)

    coinjoin_df = results["coinjoin_df"]
    peel_chains = results["peel_chains"]
    mixer_df = results["mixer_df"]
    summary = results["summary"]

    print("\n" + "=" * 65)
    print("                EVASION DETECTORS SUMMARY")
    print("=" * 65)
    print(f"CoinJoin Transactions Flagged: {summary['coinjoin_transactions_flagged']:,}")
    print(f"Peel Chains Identified:        {summary['peel_chains_found']:,}")
    print(f"Mixer Wallets Flagged:         {summary['mixer_wallets_flagged']:,}")
    print("=" * 65)

    # 1. Top 5 flagged CoinJoin transactions
    print("\n--- TOP 5 FLAGGED COINJOIN TRANSACTIONS ---")
    cj_flagged = coinjoin_df[coinjoin_df["is_coinjoin"]].head(5)
    if not cj_flagged.empty:
        for _, row in cj_flagged.iterrows():
            print(
                f"  TXID: {row['txid']}\n"
                f"    Inputs: {row['num_inputs']} | Outputs: {row['num_outputs']} | "
                f"Denom: {row['dominant_output_amount']:.4f} BTC ({row['pct_outputs_matching'] * 100:.1f}% match)"
            )
    else:
        print("  (None flagged)")

    # 2. Top 3 longest peel chains
    print("\n--- TOP 3 LONGEST PEEL CHAINS ---")
    sorted_peels = sorted(peel_chains, key=lambda c: c["total_chain_length"], reverse=True)
    if sorted_peels:
        for idx, pc in enumerate(sorted_peels[:3], 1):
            print(
                f"  Chain #{idx}: Length {pc['total_chain_length']} hops | "
                f"Start: {pc['starting_amount']:.4f} BTC | End: {pc['ending_amount']:.4f} BTC | "
                f"Peeled: {pc['total_peeled']:.4f} BTC"
            )
            print(f"    Wallets: {' -> '.join(pc['chain_wallets'])}")
    else:
        print("  (None found)")

    # 3. Top 5 flagged mixer wallets
    print("\n--- TOP 5 FLAGGED MIXER WALLETS ---")
    mixer_flagged = mixer_df[mixer_df["is_mixer_pattern"]].head(5)
    if not mixer_flagged.empty:
        for _, row in mixer_flagged.iterrows():
            print(
                f"  Wallet: {row['wallet_id']}\n"
                f"    Window Start: {row['time_window_start']} | "
                f"Sources: {row['distinct_sources']} | Destinations: {row['distinct_destinations']}"
            )
    else:
        print("  (No wallets met mixer pass-through criteria)")

    # 4. Cross-check against ground_truth.csv
    gt_path = Path(args.input).resolve().parent / "ground_truth.csv"
    if gt_path.is_file():
        try:
            gt_df = pd.read_csv(gt_path)
            print("\n" + "=" * 65)
            print("         GROUND TRUTH RECALL SANITY CROSS-CHECK")
            print("=" * 65)

            # A. CoinJoin cross-check
            gt_cj_txids = set(
                gt_df[
                    (gt_df["pattern_type"] == "coinjoin_mixing")
                    & (gt_df["entity_type"] == "txid")
                ]["entity_id"]
            )
            detected_cj_txids = set(coinjoin_df[coinjoin_df["is_coinjoin"]]["txid"])
            cj_overlap = detected_cj_txids.intersection(gt_cj_txids)
            print(f"CoinJoin Mixing (TXIDs):")
            print(f"  Ground Truth: {len(gt_cj_txids):,} | Detected: {len(detected_cj_txids):,} | Overlap: {len(cj_overlap):,}")
            if gt_cj_txids:
                print(f"  Recall: {len(cj_overlap) / len(gt_cj_txids):.1%}")

            # B. Peel chain cross-check
            gt_peel_wallets = set(
                gt_df[
                    (gt_df["pattern_type"] == "ransomware_peelchain")
                    & (gt_df["entity_type"] == "wallet")
                ]["entity_id"]
            )
            detected_peel_wallets = {
                w for pc in peel_chains for w in pc.get("chain_wallets", [])
            }
            peel_overlap = detected_peel_wallets.intersection(gt_peel_wallets)
            print(f"\nRansomware Peel Chains (Carrier Wallets):")
            print(f"  Detected Carriers: {len(detected_peel_wallets):,} | Overlap with GT: {len(peel_overlap):,}")
            if peel_overlap:
                print(f"  Identified Carrier Addresses: {list(peel_overlap)}")

            # C. Same actor cluster check
            gt_cluster_wallets = set(
                gt_df[
                    (gt_df["pattern_type"] == "same_actor_cluster")
                    & (gt_df["entity_type"] == "wallet")
                ]["entity_id"]
            )
            print(f"\nSame-Actor Multi-Wallet Cluster:")
            print(f"  Ground Truth Cluster Wallets: {len(gt_cluster_wallets):,}")

            print("=" * 65 + "\n")
        except Exception as err:
            logger.warning("Could not execute ground truth cross-check: %s", err)


if __name__ == "__main__":
    main()
