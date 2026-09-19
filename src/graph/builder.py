"""
Heterogeneous Graph Builder for Bitcoin Transactions.

Builds a directed multigraph (nx.MultiDiGraph) from normalized Bitcoin transaction
DataFrames containing:
- Wallet nodes: Bitcoin addresses (inputs and outputs)
- Transaction nodes: Transactions with timestamp, fee, and script type
- IP nodes: Broadcasting IP addresses (with optional GeoIP/ASN metadata)

Edge types:
- input: wallet -> transaction (with 'amount' attribute)
- output: transaction -> wallet (with 'amount' attribute)
- broadcast: transaction -> ip (with 'timestamp' and 'src_port' attributes)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import pandas as pd

from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Construct a heterogeneous MultiDiGraph from a normalized Bitcoin transaction DataFrame.

    Args:
        df: Normalized transaction DataFrame.

    Returns:
        nx.MultiDiGraph with wallet, transaction, and IP nodes and input, output,
        and broadcast edges.
    """
    graph = nx.MultiDiGraph()

    if df.empty:
        logger.warning("Received empty DataFrame for graph construction.")
        return graph

    has_country = "src_ip_country" in df.columns
    has_asn = "src_ip_asn" in df.columns

    # Iterate through rows efficiently using itertuples
    for row in df.itertuples(index=True):
        try:
            txid = getattr(row, "txid", None)
            if not txid or not isinstance(txid, str) or not txid.strip():
                logger.warning("Skipping row index %s due to missing or invalid txid.", row.Index)
                continue

            txid = txid.strip()
            timestamp = getattr(row, "timestamp", None)
            fee = getattr(row, "fee", None)
            script_type = getattr(row, "script_type", "")
            src_ip = getattr(row, "src_ip", None)
            src_port = getattr(row, "src_port", 0)

            # 1. Add / update transaction node
            tx_attrs: Dict[str, Any] = {
                "node_type": "transaction",
                "timestamp": timestamp,
                "fee": fee,
                "script_type": script_type,
            }
            graph.add_node(txid, **tx_attrs)

            # 2. Add / update IP node & broadcast edge
            if src_ip and isinstance(src_ip, str) and src_ip.strip():
                src_ip = src_ip.strip()
                ip_attrs: Dict[str, Any] = {
                    "node_type": "ip",
                }
                if has_country:
                    country_val = getattr(row, "src_ip_country", None)
                    ip_attrs["country"] = country_val
                    ip_attrs["src_ip_country"] = country_val
                if has_asn:
                    asn_val = getattr(row, "src_ip_asn", None)
                    ip_attrs["asn"] = asn_val
                    ip_attrs["src_ip_asn"] = asn_val

                # If node already exists, preserve or update attributes
                if src_ip not in graph:
                    graph.add_node(src_ip, **ip_attrs)
                else:
                    for k, v in ip_attrs.items():
                        if v is not None:
                            graph.nodes[src_ip][k] = v

                # Add broadcast edge: transaction -> ip
                graph.add_edge(
                    txid,
                    src_ip,
                    edge_type="broadcast",
                    timestamp=timestamp,
                    src_port=src_port,
                )

            # 3. Add input wallets & INPUT edges: wallet -> transaction
            in_addrs = getattr(row, "input_addresses", [])
            in_amts = getattr(row, "input_amounts", [])

            if isinstance(in_addrs, (list, tuple)) and isinstance(in_amts, (list, tuple)):
                if len(in_addrs) != len(in_amts):
                    logger.warning(
                        "Row %s (txid=%s) has mismatched input address/amount counts (%d vs %d).",
                        row.Index,
                        txid,
                        len(in_addrs),
                        len(in_amts),
                    )

                for addr, amt in zip(in_addrs, in_amts):
                    if not addr or not isinstance(addr, str) or not addr.strip():
                        continue
                    clean_addr = addr.strip()
                    if clean_addr not in graph:
                        graph.add_node(clean_addr, node_type="wallet")

                    try:
                        amt_val = float(amt) if amt is not None else 0.0
                    except (ValueError, TypeError):
                        amt_val = 0.0

                    graph.add_edge(
                        clean_addr,
                        txid,
                        edge_type="input",
                        amount=amt_val,
                    )

            # 4. Add output wallets & OUTPUT edges: transaction -> wallet
            out_addrs = getattr(row, "output_addresses", [])
            out_amts = getattr(row, "output_amounts", [])

            if isinstance(out_addrs, (list, tuple)) and isinstance(out_amts, (list, tuple)):
                if len(out_addrs) != len(out_amts):
                    logger.warning(
                        "Row %s (txid=%s) has mismatched output address/amount counts (%d vs %d).",
                        row.Index,
                        txid,
                        len(out_addrs),
                        len(out_amts),
                    )

                for addr, amt in zip(out_addrs, out_amts):
                    if not addr or not isinstance(addr, str) or not addr.strip():
                        continue
                    clean_addr = addr.strip()
                    if clean_addr not in graph:
                        graph.add_node(clean_addr, node_type="wallet")

                    try:
                        amt_val = float(amt) if amt is not None else 0.0
                    except (ValueError, TypeError):
                        amt_val = 0.0

                    graph.add_edge(
                        txid,
                        clean_addr,
                        edge_type="output",
                        amount=amt_val,
                    )

        except Exception as exc:
            logger.warning("Error processing row index %s for graph: %s", getattr(row, "Index", "?"), exc)
            continue

    return graph


def get_nodes_by_type(graph: nx.MultiDiGraph, node_type: str) -> List[Any]:
    """
    Retrieve all node identifiers of a specified node_type.

    Args:
        graph: The transaction MultiDiGraph.
        node_type: Node type filter ('wallet', 'transaction', or 'ip').

    Returns:
        List of matching node IDs.
    """
    return [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == node_type
    ]


def get_graph_summary(graph: nx.MultiDiGraph) -> Dict[str, int]:
    """
    Calculate summary statistics and counts for node and edge types.

    Args:
        graph: The transaction MultiDiGraph.

    Returns:
        Dictionary with count breakdowns for all node and edge types.
    """
    wallet_count = 0
    transaction_count = 0
    ip_count = 0

    for _, data in graph.nodes(data=True):
        ntype = data.get("node_type")
        if ntype == "wallet":
            wallet_count += 1
        elif ntype == "transaction":
            transaction_count += 1
        elif ntype == "ip":
            ip_count += 1

    input_edge_count = 0
    output_edge_count = 0
    broadcast_edge_count = 0

    for _, _, data in graph.edges(data=True):
        etype = data.get("edge_type")
        if etype == "input":
            input_edge_count += 1
        elif etype == "output":
            output_edge_count += 1
        elif etype == "broadcast":
            broadcast_edge_count += 1

    return {
        "wallet_nodes": wallet_count,
        "transaction_nodes": transaction_count,
        "ip_nodes": ip_count,
        "input_edges": input_edge_count,
        "output_edges": output_edge_count,
        "broadcast_edges": broadcast_edge_count,
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Build and summarize a heterogeneous Bitcoin transaction graph."
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
    """CLI entry point for testing graph builder standalone."""
    args = _build_parser().parse_args()

    print(f"[*] Ingesting transactions from: {args.input}")
    df = parse_file(args.input)

    print(f"[*] Constructing heterogeneous graph from {len(df):,} transactions...")
    graph = build_graph(df)

    summary = get_graph_summary(graph)

    print("\n" + "=" * 55)
    print("           HETEROGENEOUS GRAPH SUMMARY")
    print("=" * 55)
    print(f"Total Nodes: {summary['total_nodes']:,}")
    print(f"  - Wallet Nodes:      {summary['wallet_nodes']:,}")
    print(f"  - Transaction Nodes: {summary['transaction_nodes']:,}")
    print(f"  - IP Nodes:          {summary['ip_nodes']:,}")
    print("-" * 55)
    print(f"Total Edges: {summary['total_edges']:,}")
    print(f"  - Input Edges:       {summary['input_edges']:,} (wallet -> transaction)")
    print(f"  - Output Edges:      {summary['output_edges']:,} (transaction -> wallet)")
    print(f"  - Broadcast Edges:   {summary['broadcast_edges']:,} (transaction -> ip)")
    print("=" * 55)

    print("\n" + "=" * 55)
    print("              EXAMPLE NODES BY TYPE")
    print("=" * 55)
    for ntype in ["wallet", "transaction", "ip"]:
        node_ids = get_nodes_by_type(graph, ntype)
        print(f"\n--- {ntype.upper()} NODES (Showing up to 3 of {len(node_ids):,}) ---")
        for nid in node_ids[:3]:
            attrs = graph.nodes[nid]
            print(f"  ID: {nid}")
            print(f"    Attributes: {attrs}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
