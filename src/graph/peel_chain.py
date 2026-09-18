"""
Peel Chain Detection and Analysis Module.

Detects classic peel chain patterns in Bitcoin transaction graphs where an entity
holding a significant balance repeatedly transfers the majority onward through a
sequence of new carrier wallets while peeling off smaller amounts at each hop.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import pandas as pd

from src.graph.builder import build_graph
from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def is_peel_transaction(
    df_row_or_amounts: Union[Dict[str, Any], List[float], Tuple[float, ...], Any],
    peel_ratio_threshold: float = 0.15,
) -> bool:
    """
    Check if a transaction exhibits a peel structure.

    Accepts transactions with 2 or more outputs where one dominant output
    carries the majority of value (the non-dominant outputs' share of total
    value is <= peel_ratio_threshold). This handles:
    - Classic 2-output peels (large carrier + small peel-off)
    - 3+ output transactions with a dominant carrier and small fee-change/peel outputs

    Args:
        df_row_or_amounts: A list of output amounts, a dictionary representing a row,
            or an object with an 'output_amounts' attribute.
        peel_ratio_threshold: Maximum ratio of non-dominant outputs to total output
            value (default: 0.15).

    Returns:
        bool: True if transaction has 2+ outputs and non-carrier / total <= threshold.
    """
    if isinstance(df_row_or_amounts, (list, tuple)):
        out_amts = df_row_or_amounts
    elif isinstance(df_row_or_amounts, dict):
        out_amts = df_row_or_amounts.get("output_amounts", [])
    elif hasattr(df_row_or_amounts, "output_amounts"):
        out_amts = getattr(df_row_or_amounts, "output_amounts", [])
    else:
        return False

    if not isinstance(out_amts, (list, tuple)) or len(out_amts) < 2:
        return False

    float_amts: List[float] = []
    for a in out_amts:
        try:
            float_amts.append(float(a))
        except (ValueError, TypeError):
            return False

    total = sum(float_amts)
    if total <= 0:
        return False

    largest = max(float_amts)
    non_carrier_total = total - largest
    ratio = non_carrier_total / total
    return ratio <= peel_ratio_threshold


def detect_peel_chains(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
    min_chain_length: int = 3,
    peel_ratio_threshold: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Detect peel chains of consecutive peel transactions across the transaction graph.

    Traces sequences where wallet A -> (peel step) -> wallet B -> (peel step) -> wallet C,
    where B is the primary carrier forwarding the vast majority of funds.

    Args:
        graph: Heterogeneous transaction MultiDiGraph.
        df: Normalized transaction DataFrame.
        min_chain_length: Minimum consecutive peel steps required (default: 3).
        peel_ratio_threshold: Maximum fraction for the peeled output (default: 0.15).

    Returns:
        list[dict]: Detected peel chains, where each entry contains:
            - chain_wallets: list of carrier wallet IDs in order
            - chain_txids: list of linking transaction IDs in order
            - total_chain_length: count of consecutive peel hops
            - starting_amount: initial input/output value before first hop
            - ending_amount: final remaining carrier amount
            - total_peeled: cumulative sum of amounts peeled off
    """
    if df.empty or graph.number_of_nodes() == 0:
        return []

    # Map candidate peel transitions: in_wallet -> list of outgoing peel steps
    transitions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    tx_lookup: Dict[str, Dict[str, Any]] = {}

    for row in df.itertuples(index=True):
        out_amts = getattr(row, "output_amounts", [])
        out_addrs = getattr(row, "output_addresses", [])
        in_addrs = getattr(row, "input_addresses", [])
        txid = getattr(row, "txid", None)
        ts = getattr(row, "timestamp", None)

        if not txid or not isinstance(out_amts, (list, tuple)) or not isinstance(out_addrs, (list, tuple)):
            continue

        if not is_peel_transaction(out_amts, peel_ratio_threshold=peel_ratio_threshold):
            continue

        # Require at least 2 outputs and matching address/amount counts
        if len(out_addrs) < 2 or len(out_amts) < 2 or len(out_addrs) != len(out_amts):
            continue

        clean_txid = str(txid).strip()

        # Parse all outputs and identify carrier (largest) vs peel (rest)
        parsed_outputs: List[Tuple[str, float]] = []
        for addr, amt in zip(out_addrs, out_amts):
            try:
                parsed_outputs.append((str(addr).strip(), float(amt)))
            except (ValueError, TypeError):
                continue

        if len(parsed_outputs) < 2:
            continue

        # Sort by amount descending; carrier is the largest output
        parsed_outputs.sort(key=lambda x: x[1], reverse=True)
        carrier_addr, carrier_amt = parsed_outputs[0]
        peel_amt = round(sum(a for _, a in parsed_outputs[1:]), 8)
        total_amt = round(carrier_amt + peel_amt, 8)
        # For display, record the first (largest) peel-off wallet
        peel_addr = parsed_outputs[1][0] if len(parsed_outputs) > 1 else ""

        step_info = {
            "txid": clean_txid,
            "carrier": carrier_addr,
            "peel_wallet": peel_addr,
            "carrier_amt": carrier_amt,
            "peel_amt": peel_amt,
            "total_amt": total_amt,
            "timestamp": ts,
        }
        tx_lookup[clean_txid] = step_info

        if isinstance(in_addrs, (list, tuple)):
            for in_w in in_addrs:
                if in_w and isinstance(in_w, str) and in_w.strip():
                    transitions[in_w.strip()].append(step_info)

    # Sort transitions chronologically if timestamps are available
    for in_w in transitions:
        transitions[in_w].sort(
            key=lambda s: s["timestamp"] if s["timestamp"] is not None else pd.Timestamp.min
        )

    # Helper function to find longest non-cyclic chain from a given wallet
    def _trace_longest_chain(
        curr_wallet: str,
        visited_wallets: List[str],
        accumulated_txids: List[str],
    ) -> Tuple[List[str], List[str]]:
        best_wallets = visited_wallets
        best_txids = accumulated_txids

        for step in transitions.get(curr_wallet, []):
            next_carrier = step["carrier"]
            txid = step["txid"]

            if next_carrier not in visited_wallets and txid not in accumulated_txids:
                sub_wallets, sub_txids = _trace_longest_chain(
                    next_carrier,
                    visited_wallets + [next_carrier],
                    accumulated_txids + [txid],
                )
                if len(sub_txids) > len(best_txids):
                    best_wallets = sub_wallets
                    best_txids = sub_txids

        return best_wallets, best_txids

    # Find longest candidate chain starting from every entry wallet
    candidate_chains: List[Tuple[List[str], List[str]]] = []
    for start_wallet in transitions:
        wallets, txids = _trace_longest_chain(start_wallet, [start_wallet], [])
        if len(txids) >= min_chain_length:
            candidate_chains.append((wallets, txids))

    # Sort candidate chains by length descending to prioritize maximal chains
    candidate_chains.sort(key=lambda c: len(c[1]), reverse=True)

    # Filter out overlapping subchains (a wallet can only be a carrier in one chain)
    claimed_wallets: Set[str] = set()
    claimed_txids: Set[str] = set()
    final_chains: List[Dict[str, Any]] = []

    for chain_wallets, chain_txids in candidate_chains:
        # If the starting wallet or txids are already part of an accepted longer chain, skip subchain
        if chain_wallets[0] in claimed_wallets or any(tx in claimed_txids for tx in chain_txids):
            continue

        # Compute chain financial totals
        first_step = tx_lookup[chain_txids[0]]
        last_step = tx_lookup[chain_txids[-1]]
        starting_amount = round(first_step["total_amt"], 8)
        ending_amount = round(last_step["carrier_amt"], 8)
        total_peeled = round(sum(tx_lookup[t]["peel_amt"] for t in chain_txids), 8)

        # Detect collector pattern: check if the chain's starting wallet has
        # high fan-in (many distinct source wallets feeding it), indicating a
        # ransomware collector or aggregation point.
        start_wallet = chain_wallets[0]
        feeder_wallets: Set[str] = set()
        feeder_txids: Set[str] = set()

        for row in df.itertuples(index=True):
            row_out_addrs = getattr(row, "output_addresses", [])
            row_in_addrs = getattr(row, "input_addresses", [])
            row_txid = getattr(row, "txid", None)

            if not isinstance(row_out_addrs, (list, tuple)):
                continue

            clean_outs = [str(a).strip() for a in row_out_addrs if a]
            if start_wallet not in clean_outs:
                continue

            # This transaction sends TO the chain start wallet
            row_txid_str = str(row_txid).strip() if row_txid else ""
            if row_txid_str and row_txid_str not in chain_txids:
                feeder_txids.add(row_txid_str)

            if isinstance(row_in_addrs, (list, tuple)):
                for a in row_in_addrs:
                    clean_a = str(a).strip() if a else ""
                    if clean_a and clean_a != start_wallet:
                        feeder_wallets.add(clean_a)

        is_collector = len(feeder_wallets) >= 3

        final_chains.append({
            "chain_wallets": chain_wallets,
            "chain_txids": chain_txids,
            "total_chain_length": len(chain_txids),
            "starting_amount": starting_amount,
            "ending_amount": ending_amount,
            "total_peeled": total_peeled,
            "collector_wallet": start_wallet if is_collector else None,
            "feeder_wallets": sorted(feeder_wallets),
            "feeder_txids": sorted(feeder_txids),
        })

        claimed_wallets.update(chain_wallets)
        claimed_txids.update(chain_txids)

    logger.info(
        "Peel chain detection complete: identified %d peel chains of length >= %d.",
        len(final_chains),
        min_chain_length,
    )
    return final_chains


def annotate_graph_with_peel_chains(
    graph: nx.MultiDiGraph,
    chains: List[Dict[str, Any]],
) -> nx.MultiDiGraph:
    """
    Annotate graph wallet nodes with peel chain membership and hop positions.

    Args:
        graph: Heterogeneous transaction MultiDiGraph.
        chains: List of detected peel chain dictionaries.

    Returns:
        The modified graph with 'peel_chain_id' and 'peel_chain_position' attributes.
    """
    for chain_idx, chain in enumerate(chains):
        wallets = chain.get("chain_wallets", [])
        for pos, wallet_id in enumerate(wallets):
            if wallet_id in graph:
                node_data = graph.nodes[wallet_id]
                # Store chain ID (or append if multi-chain membership)
                if "peel_chain_id" not in node_data:
                    node_data["peel_chain_id"] = chain_idx
                    node_data["peel_chain_position"] = pos
                else:
                    existing_id = node_data["peel_chain_id"]
                    if isinstance(existing_id, list):
                        node_data["peel_chain_id"].append(chain_idx)
                        node_data["peel_chain_position"].append(pos)
                    else:
                        node_data["peel_chain_id"] = [existing_id, chain_idx]
                        node_data["peel_chain_position"] = [node_data["peel_chain_position"], pos]

    logger.info("Graph annotated with %d peel chains.", len(chains))
    return graph


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Detect and trace Bitcoin peel chains across transaction graphs."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input transaction file (.csv, .json, or .xml).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Minimum consecutive peel steps to qualify as a peel chain (default: 3).",
    )
    parser.add_argument(
        "--peel-ratio",
        type=float,
        default=0.15,
        help="Maximum peel amount ratio relative to total output value (default: 0.15).",
    )
    return parser


def main() -> None:
    """CLI entry point for standalone peel chain detection."""
    args = _build_parser().parse_args()

    print(f"[*] Parsing input transactions: {args.input}")
    df = parse_file(args.input)

    print(f"[*] Building graph from {len(df):,} transactions...")
    graph = build_graph(df)

    print(f"[*] Running peel chain detection (min_length={args.min_length}, peel_ratio<={args.peel_ratio})...")
    chains = detect_peel_chains(
        graph=graph,
        df=df,
        min_chain_length=args.min_length,
        peel_ratio_threshold=args.peel_ratio,
    )

    # Annotate graph
    annotate_graph_with_peel_chains(graph, chains)

    print("\n" + "=" * 65)
    print("                PEEL CHAIN DETECTION SUMMARY")
    print("=" * 65)
    print(f"Total Peel Chains Detected: {len(chains)}")
    print("-" * 65)

    all_chain_wallets: Set[str] = set()
    for idx, chain in enumerate(chains, 1):
        wallets = chain["chain_wallets"]
        all_chain_wallets.update(wallets)
        print(f"\nChain #{idx}:")
        print(f"  Length:          {chain['total_chain_length']} consecutive hops")
        print(f"  Starting Amount: {chain['starting_amount']:.8f} BTC")
        print(f"  Ending Amount:   {chain['ending_amount']:.8f} BTC")
        print(f"  Total Peeled:    {chain['total_peeled']:.8f} BTC")
        print("  Wallet Sequence (Hop 0 -> Hop N):")
        for hop, w in enumerate(wallets):
            role = "Initial Source" if hop == 0 else f"Carrier Hop {hop}"
            print(f"    [{hop}] {w} ({role})")

    # Cross-check against ground_truth.csv if available
    gt_path = Path(args.input).resolve().parent / "ground_truth.csv"
    if gt_path.is_file():
        try:
            gt_df = pd.read_csv(gt_path)
            gt_ransomware_wallets = set(
                gt_df[
                    (gt_df["pattern_type"] == "ransomware_peelchain")
                    & (gt_df["entity_type"] == "wallet")
                ]["entity_id"]
            )
            overlap = all_chain_wallets.intersection(gt_ransomware_wallets)
            print("\n" + "-" * 65)
            print("CROSS-CHECK AGAINST GROUND TRUTH (ransomware_peelchain):")
            print(f"  Ground Truth Ransomware Wallets: {len(gt_ransomware_wallets):,}")
            print(f"  Wallets in Detected Peel Chains: {len(all_chain_wallets):,}")
            print(f"  Overlap Count (Carrier Wallets): {len(overlap):,}")
            if overlap:
                print(f"  Matched Wallets: {list(overlap)}")
            print("-" * 65)
        except Exception as err:
            logger.warning("Could not cross-check with ground_truth.csv: %s", err)

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
