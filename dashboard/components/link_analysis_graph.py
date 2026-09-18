"""
Interactive Link-Analysis Graph Component — Enterprise Forensic Theme.
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict, List, Optional, Set
import networkx as nx
import pandas as pd
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components


def render_link_analysis_graph(
    graph: nx.MultiDiGraph,
    focused_wallet: str,
    max_hops: int = 2,
    max_nodes: int = 45,
) -> None:
    """
    Render interactive Pyvis link-analysis graph with professional forensic styling.
    """
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <h4 style="margin: 0; color: #C9D1D9; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 16px;">
                Graph Topology & Entity Neighbors
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if focused_wallet not in graph:
        st.warning(f"Wallet '{focused_wallet}' is not present in graph topology.")
        return

    # Extract k-hop neighborhood subgraph
    nodes_to_include: Set[str] = {focused_wallet}
    current_frontier = {focused_wallet}

    for _ in range(max_hops):
        next_frontier = set()
        for node in current_frontier:
            neighbors = set(graph.predecessors(node)).union(set(graph.successors(node)))
            next_frontier.update(neighbors)
        nodes_to_include.update(next_frontier)
        current_frontier = next_frontier
        if len(nodes_to_include) >= max_nodes:
            break

    subgraph_nodes = list(nodes_to_include)[:max_nodes]
    subgraph = graph.subgraph(subgraph_nodes)

    # Build Pyvis interactive network with professional dark canvas
    net = Network(
        height="520px",
        width="100%",
        bgcolor="#000000",
        font_color="#C9D1D9",
        directed=True,
    )
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
    )

    # Style and add nodes with professional muted palette
    for node, data in subgraph.nodes(data=True):
        node_type = data.get("node_type", "wallet")
        label = str(node)[:10] + "..." if len(str(node)) > 14 else str(node)

        if node == focused_wallet:
            # Target entity — slate blue, prominent but not glowing
            net.add_node(
                node,
                label=label,
                title=f"TARGET ENTITY:\n{node}",
                color={"background": "#5B8DEF", "border": "#C9D1D9",
                       "highlight": {"background": "#4A7AD8", "border": "#5B8DEF"}},
                borderWidth=2,
                size=30,
                shape="hexagon",
            )
        elif node_type == "wallet":
            net.add_node(
                node,
                label=label,
                title=f"Wallet: {node}",
                color={"background": "#2EA043", "border": "#3FB950"},
                borderWidth=1,
                size=18,
                shape="dot",
            )
        elif node_type == "transaction":
            fee_str = f"\nFee: {data.get('fee', 0)} BTC" if 'fee' in data else ""
            script_str = f"\nScript: {data.get('script_type', '')}" if 'script_type' in data else ""
            net.add_node(
                node,
                label=f"TX: {label}",
                title=f"TXID: {node}{fee_str}{script_str}",
                color={"background": "#6E7681", "border": "#8B949E"},
                borderWidth=1,
                size=16,
                shape="square",
            )
        elif node_type == "ip":
            net.add_node(
                node,
                label=label,
                title=f"Broadcasting IP: {node}",
                color={"background": "#BB8009", "border": "#D29922"},
                borderWidth=1.5,
                size=20,
                shape="diamond",
            )
        else:
            net.add_node(node, label=label, color="#8B949E", size=14)

    # Add edges with muted, professional colors
    for u, v, k, edata in subgraph.edges(data=True, keys=True):
        edge_type = edata.get("edge_type", "link")
        amt = edata.get("amount")
        amt_label = f"{amt:.3f} BTC" if amt is not None else ""

        if edge_type == "input":
            net.add_edge(u, v, title=f"Input Inflow: {amt_label}", label=amt_label,
                         color={"color": "#3FB950", "highlight": "#56D364"}, width=1.8)
        elif edge_type == "output":
            is_change = edata.get("likely_change_address", False)
            edge_color = "#D29922" if is_change else "#5B8DEF"
            label_suffix = " (Change)" if is_change else ""
            net.add_edge(u, v, title=f"Output Outflow: {amt_label}{label_suffix}",
                         label=f"{amt_label}{label_suffix}",
                         color={"color": edge_color, "highlight": "#C9D1D9"}, width=1.8)
        elif edge_type == "broadcast":
            net.add_edge(u, v, title="P2P Broadcast Relay", label="relayed",
                         color={"color": "#8B949E"}, dashes=True, width=1.0)
        elif edge_type == "common_input_ownership":
            net.add_edge(u, v, title="Co-Input Common Ownership", label="co-spend",
                         color={"color": "#DA3633"}, dashes=True, width=1.6)

    # Render HTML in temp file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tf:
        net.save_graph(tf.name)
        with open(tf.name, "r", encoding="utf-8") as f_html:
            html_content = f_html.read()

    components.html(html_content, height=540, scrolling=True)

    # Clean legend with CSS dots — no emojis
    st.markdown(
        """
        <div style="display: flex; gap: 24px; flex-wrap: wrap; background: #14181F; padding: 10px 16px;
                    border-radius: 6px; border: 1px solid #21262D; font-size: 12px; color: #8B949E;
                    font-family: 'Inter', sans-serif; margin-top: 8px;">
            <span style="display: inline-flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #5B8DEF;"></span>
                <b style="color: #C9D1D9;">Target</b>
            </span>
            <span style="display: inline-flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #3FB950;"></span>
                <b style="color: #C9D1D9;">Wallets</b>
            </span>
            <span style="display: inline-flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; background: #8B949E;"></span>
                <b style="color: #C9D1D9;">Transactions</b>
            </span>
            <span style="display: inline-flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; background: #D29922; transform: rotate(45deg);"></span>
                <b style="color: #C9D1D9;">Broadcasting IPs</b>
            </span>
            <span style="display: inline-flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 12px; height: 2px; background: #DA3633;"></span>
                <b style="color: #C9D1D9;">Co-Spend Link</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
