"""
Interactive Link-Analysis Graph Component with BitForge Emerald/Cyber Theme.
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
    Render interactive Pyvis link-analysis graph with BitForge cyberpunk emerald styling.
    """
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <h4 style="margin: 0; color: #F0FDF4; font-weight: 700;">
                Forensic Graph Topology & Entity Neighbors
            </h4>
            <span style="color: #00F29B; font-size: 0.8rem; background: rgba(0, 242, 155, 0.1); border: 1px solid rgba(0, 242, 155, 0.2); padding: 2px 10px; border-radius: 12px;">
                BITFORGE NEURAL GRAPH
            </span>
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

    # Build Pyvis interactive network with BitForge dark emerald canvas
    net = Network(
        height="500px",
        width="100%",
        bgcolor="#060C09",
        font_color="#F0FDF4",
        directed=True,
    )
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
    )

    # Style and add nodes matching BitForge screenshot colors
    for node, data in subgraph.nodes(data=True):
        node_type = data.get("node_type", "wallet")
        label = str(node)[:10] + "..." if len(str(node)) > 14 else str(node)
        
        if node == focused_wallet:
            # BitForge Glowing Hexagon Hub for Target
            net.add_node(
                node,
                label=f"🎯 {label}",
                title=f"TARGET ENTITY:\n{node}",
                color={"background": "#00F29B", "border": "#FFFFFF", "highlight": {"background": "#00DC82", "border": "#00F29B"}},
                borderWidth=3,
                size=34,
                shape="hexagon",
                shadow={"enabled": True, "color": "rgba(0, 242, 155, 0.8)", "size": 15},
            )
        elif node_type == "wallet":
            net.add_node(
                node,
                label=label,
                title=f"Wallet: {node}",
                color={"background": "#10B981", "border": "#34D399"},
                borderWidth=1.5,
                size=20,
                shape="dot",
            )
        elif node_type == "transaction":
            fee_str = f"\nFee: {data.get('fee', 0)} BTC" if 'fee' in data else ""
            script_str = f"\nScript: {data.get('script_type', '')}" if 'script_type' in data else ""
            net.add_node(
                node,
                label=f"TX: {label}",
                title=f"TXID: {node}{fee_str}{script_str}",
                color={"background": "#06B6D4", "border": "#67E8F9"},
                borderWidth=1.5,
                size=18,
                shape="square",
            )
        elif node_type == "ip":
            net.add_node(
                node,
                label=f"🌐 {label}",
                title=f"Broadcasting IP: {node}",
                color={"background": "#8B5CF6", "border": "#C4B5FD"},
                borderWidth=2,
                size=22,
                shape="diamond",
                shadow={"enabled": True, "color": "rgba(139, 92, 246, 0.5)", "size": 10},
            )
        else:
            net.add_node(node, label=label, color="#94A3B8", size=15)

    # Add edges with glowing cyber lines
    for u, v, k, edata in subgraph.edges(data=True, keys=True):
        edge_type = edata.get("edge_type", "link")
        amt = edata.get("amount")
        amt_label = f"{amt:.3f} BTC" if amt is not None else ""

        if edge_type == "input":
            net.add_edge(u, v, title=f"Input Inflow: {amt_label}", label=amt_label, color={"color": "#10B981", "highlight": "#00F29B"}, width=2.0)
        elif edge_type == "output":
            is_change = edata.get("likely_change_address", False)
            edge_color = "#FF80AB" if is_change else "#00F29B"
            label_suffix = " (Change)" if is_change else ""
            net.add_edge(u, v, title=f"Output Outflow: {amt_label}{label_suffix}", label=f"{amt_label}{label_suffix}", color={"color": edge_color, "highlight": "#FFFFFF"}, width=2.2)
        elif edge_type == "broadcast":
            net.add_edge(u, v, title="P2P Broadcast Relay", label="relayed", color={"color": "#C4B5FD"}, dashes=True, width=1.2)
        elif edge_type == "common_input_ownership":
            net.add_edge(u, v, title="Co-Input Common Ownership", label="co-spend", color={"color": "#EC4899"}, dashes=True, width=2.0)

    # Render HTML in temp file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tf:
        net.save_graph(tf.name)
        with open(tf.name, "r", encoding="utf-8") as f_html:
            html_content = f_html.read()

    components.html(html_content, height=520, scrolling=True)
    st.markdown(
        """
        <div style="display: flex; gap: 16px; flex-wrap: wrap; background: rgba(13, 27, 19, 0.5); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(0, 242, 155, 0.1); font-size: 0.8rem; color: #94A3B8;">
            <span>🟢 <b style="color: #00F29B;">Target Focus</b></span>
            <span>🟢 <b style="color: #10B981;">Wallets</b></span>
            <span>🔷 <b style="color: #06B6D4;">Transactions</b></span>
            <span>🟣 <b style="color: #8B5CF6;">Broadcasting IPs</b></span>
            <span>⚡ <b style="color: #EC4899;">Co-Spend Link</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
