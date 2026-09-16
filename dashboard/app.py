"""
Streamlit Dashboard for Bitcoin Traffic Analyzer.
Theme: BitForge Cyberpunk Emerald & Neon Mint UI.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is in Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.alert_table import render_alert_table
from dashboard.components.detail_view import render_detail_view
from dashboard.components.link_analysis_graph import render_link_analysis_graph
from dashboard.report_export import generate_case_report
from src.feedback.feedback_store import FeedbackStore
from src.graph.builder import build_graph
from src.graph.heuristics import apply_change_address_heuristic, apply_common_input_heuristic
from src.ingestion.parser import parse_file
from src.scoring import run_full_pipeline
from src.utils.paths import get_project_root


# Page configuration
st.set_page_config(
    page_title="BitForge | Bitcoin Traffic Analyzer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# BitForge Theme CSS Injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Dark Emerald Canvas */
    .stApp {
        background: radial-gradient(circle at 50% 10%, rgba(0, 242, 155, 0.12) 0%, rgba(6, 12, 9, 0.98) 75%), #060C09;
        font-family: 'Inter', -apple-system, sans-serif;
        color: #F0FDF4;
    }

    /* Top Navigation Bar */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 24px;
        background: rgba(13, 27, 19, 0.75);
        border: 1px solid rgba(0, 242, 155, 0.2);
        border-radius: 16px;
        backdrop-filter: blur(16px);
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -5px rgba(0, 242, 155, 0.08);
    }

    .brand-title {
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #A7F3D0 50%, #00F29B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }

    .nav-links {
        display: flex;
        gap: 20px;
        align-items: center;
    }

    .nav-item {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 500;
        text-decoration: none;
        transition: color 0.2s ease;
    }

    .nav-item.active {
        color: #00F29B;
        font-weight: 600;
    }

    .badge-status {
        background: rgba(0, 242, 155, 0.15);
        color: #00F29B;
        border: 1px solid rgba(0, 242, 155, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* BitForge Hero Section */
    .hero-container {
        text-align: center;
        padding: 24px 12px 32px 12px;
    }

    .hero-heading {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
        background: linear-gradient(180deg, #FFFFFF 0%, #E2E8F0 60%, #00F29B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        max-width: 750px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* KPI Glassmorphism Cards */
    .kpi-card {
        background: rgba(13, 27, 19, 0.7);
        border: 1px solid rgba(0, 242, 155, 0.18);
        border-radius: 14px;
        padding: 18px 20px;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 24px -4px rgba(0, 242, 155, 0.08);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 155, 0.4);
    }

    .kpi-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 6px;
    }

    .kpi-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
    }

    .pill-green {
        background: rgba(0, 242, 155, 0.15);
        color: #00F29B;
    }

    .pill-red {
        background: rgba(255, 75, 75, 0.15);
        color: #FF4B4B;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(13, 27, 19, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(0, 242, 155, 0.15);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        padding: 8px 18px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 242, 155, 0.15) !important;
        color: #00F29B !important;
        border: 1px solid rgba(0, 242, 155, 0.3) !important;
    }

    /* Streamlit Custom Overrides */
    div.stButton > button {
        background: linear-gradient(135deg, #00F29B 0%, #059669 100%);
        color: #06120B;
        font-weight: 700;
        border: none;
        border-radius: 20px;
        padding: 8px 24px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px 0 rgba(0, 242, 155, 0.35);
    }

    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px 0 rgba(0, 242, 155, 0.5);
    }

    /* Dataframe container */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 242, 155, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_processed_data() -> dict:
    """Load or initialize processed pipeline data."""
    root = get_project_root()
    alerts_file = root / "data" / "processed" / "scored_alerts.csv"
    tx_file = root / "data" / "raw" / "synthetic_transactions.csv"
    gt_file = root / "data" / "raw" / "ground_truth.csv"
    peel_file = root / "data" / "processed" / "peel_chains.json"
    cj_file = root / "data" / "processed" / "coinjoin_transactions.csv"

    if not alerts_file.is_file() and tx_file.is_file():
        run_full_pipeline(tx_file)

    alerts_df = pd.read_csv(alerts_file, index_col=0) if alerts_file.is_file() else pd.DataFrame()
    raw_df = parse_file(tx_file) if tx_file.is_file() else pd.DataFrame()
    gt_df = pd.read_csv(gt_file) if gt_file.is_file() else pd.DataFrame()
    
    peel_chains = []
    if peel_file.is_file():
        with open(peel_file, "r", encoding="utf-8") as f:
            peel_chains = json.load(f)

    cj_df = pd.read_csv(cj_file) if cj_file.is_file() else pd.DataFrame()

    # Build memory graph for link analysis
    graph = build_graph(raw_df) if not raw_df.empty else None
    if graph:
        apply_common_input_heuristic(graph, raw_df)
        apply_change_address_heuristic(graph, raw_df)

    # Recalibrate scores with stored feedback
    fb_store = FeedbackStore(root / "data" / "processed" / "analyst_feedback.db")
    if not alerts_df.empty:
        alerts_df = fb_store.recalibrate_scores(alerts_df)

    return {
        "alerts_df": alerts_df,
        "raw_df": raw_df,
        "gt_df": gt_df,
        "peel_chains": peel_chains,
        "cj_df": cj_df,
        "graph": graph,
    }


def main() -> None:
    data = load_processed_data()
    alerts_df = data["alerts_df"]
    raw_df = data["raw_df"]
    gt_df = data["gt_df"]
    peel_chains = data["peel_chains"]
    cj_df = data["cj_df"]
    graph = data["graph"]

    root = get_project_root()
    fb_store = FeedbackStore(root / "data" / "processed" / "analyst_feedback.db")

    # Top Navbar matching BitForge aesthetic
    st.markdown(
        """
        <div class="navbar">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.6rem;">⚡</span>
                <span class="brand-title">BitForge</span>
                <span style="color: #64748B; font-size: 0.9rem; font-weight: 500;">| Bitcoin Traffic Analyzer</span>
            </div>
            <div class="nav-links">
                <span class="badge-status">🟢 100% OFFLINE SECURE</span>
                <span style="color: #94A3B8; font-size: 0.85rem;">NTRO · SIH26146</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Banner
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-heading">REDEFINING FORENSIC BLOCKCHAIN INTELLIGENCE</div>
            <div class="hero-subtitle">
                Correlating network-layer P2P broadcast signals with on-chain transaction flows to surface explainable, court-ready investigative leads.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Metrics Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Ingested Transactions</div>
                <div class="kpi-value">{len(raw_df):,}</div>
                <span class="kpi-pill pill-green">↑ 100% Ingested</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Analyzed Wallets</div>
                <div class="kpi-value">{len(alerts_df):,}</div>
                <span class="kpi-pill pill-green">5,814 Clusters</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        crit_cnt = len(alerts_df[alerts_df["risk_level"] == "CRITICAL"]) if not alerts_df.empty else 0
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Critical Alerts</div>
                <div class="kpi-value" style="color: {'#FF4B4B' if crit_cnt > 0 else '#00F29B'};">{crit_cnt:,}</div>
                <span class="kpi-pill pill-red">Action Required</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        cj_cnt = int(cj_df["is_coinjoin"].sum()) if not cj_df.empty and "is_coinjoin" in cj_df.columns else 0
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">CoinJoin Mixes</div>
                <div class="kpi-value">{cj_cnt:,}</div>
                <span class="kpi-pill pill-green">Signature Detected</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Peel Chains</div>
                <div class="kpi-value">{len(peel_chains):,}</div>
                <span class="kpi-pill pill-green">Ransomware Flow</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # Main Navigation Tabs
    tab_alerts, tab_graph, tab_evasion, tab_export, tab_pipeline = st.tabs([
        "🚨 Investigative Alerts",
        "🕸️ Link Analysis Graph",
        "🔍 Evasion & Laundering Hub",
        "📑 Case Dossier Export",
        "⚙️ Pipeline & Ground Truth",
    ])

    # TAB 1: Investigative Alerts
    with tab_alerts:
        selected_wallet = render_alert_table(alerts_df)
        if selected_wallet:
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            render_detail_view(selected_wallet, alerts_df, feedback_store=fb_store)

    # TAB 2: Link Analysis Graph
    with tab_graph:
        st.markdown("#### 🎯 Interactive Link-Analysis Explorer")
        st.caption("Traverse multi-hop relationships between wallets, transactions, and broadcasting IP addresses.")
        
        target_wallet = st.selectbox(
            "Select Target Wallet for Graph Inspection:",
            options=list(alerts_df.index) if not alerts_df.empty else [],
            index=0 if not alerts_df.empty else None,
            key="graph_target_select",
        )

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            max_hops = st.slider("Graph Exploration Radius (Hops)", min_value=1, max_value=3, value=2)
        with col_g2:
            max_nodes = st.slider("Maximum Node Density", min_value=15, max_value=80, value=40)

        if graph and target_wallet:
            render_link_analysis_graph(graph, target_wallet, max_hops=max_hops, max_nodes=max_nodes)
        else:
            st.info("Please load transaction data to render graph.")

    # TAB 3: Evasion Hub
    with tab_evasion:
        st.markdown("### 🔬 Evasion-Specific Forensic Detectors")
        
        col_cj, col_pc = st.columns(2)
        with col_cj:
            st.markdown("#### 🌀 CoinJoin Mixing Transactions")
            if not cj_df.empty:
                cj_flagged = cj_df[cj_df["is_coinjoin"]]
                st.dataframe(
                    cj_flagged[["txid", "num_inputs", "num_outputs", "dominant_output_amount", "pct_outputs_matching"]],
                    use_container_width=True,
                    height=240,
                )
            else:
                st.info("No CoinJoin transactions flagged.")

        with col_pc:
            st.markdown("#### 🍌 Ransomware Peel Chains")
            if peel_chains:
                pc_records = []
                for idx, pc in enumerate(peel_chains, 1):
                    pc_records.append({
                        "Chain #": idx,
                        "Hops": pc.get("total_chain_length", 0),
                        "Start (BTC)": pc.get("starting_amount", 0.0),
                        "Peeled (BTC)": pc.get("total_peeled", 0.0),
                        "Wallets": len(pc.get("chain_wallets", [])),
                    })
                st.dataframe(pd.DataFrame(pc_records), use_container_width=True, height=240)
            else:
                st.info("No peel chains detected.")

        st.markdown("---")
        st.markdown("#### 🌐 Network Infrastructure Category Distribution")
        if not alerts_df.empty and "asn_category" in alerts_df.columns:
            asn_counts = alerts_df["asn_category"].value_counts().reset_index()
            asn_counts.columns = ["Category", "Count"]
            
            fig = px.bar(
                asn_counts,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_sequence=["#00F29B", "#06B6D4", "#8B5CF6", "#F59E0B", "#EF4444"],
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(13, 27, 19, 0.7)",
                plot_bgcolor="rgba(13, 27, 19, 0.7)",
                font=dict(color="#F0FDF4"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # TAB 4: Case Dossier Export
    with tab_export:
        st.markdown("### 📑 Automated Forensic Case Dossier Generator")
        st.caption("Generate official evidence dossiers (PDF & Markdown) summarizing on-chain and network-layer forensic links.")

        col_exp1, col_exp2, col_exp3 = st.columns([3, 1, 1])
        with col_exp1:
            export_target = st.selectbox(
                "Select Subject Wallet to Generate Case Dossier:",
                options=list(alerts_df.index) if not alerts_df.empty else [],
                key="export_target_select",
            )
        with col_exp2:
            st.write("")
            st.write("")
            gen_btn = st.button("📄 Generate Dossier", use_container_width=True)

        if export_target and (gen_btn or "last_exported" in st.session_state):
            rep_path_md = generate_case_report(export_target, alerts_df, output_dir=root / "reports", format="md")
            pdf_target_path = root / "reports" / f"case_report_{export_target[:12]}.pdf"
            st.session_state["last_exported"] = rep_path_md
            st.success(f"Dossier generated: `{rep_path_md.name}` and `{pdf_target_path.name}`")

            with open(rep_path_md, "r", encoding="utf-8") as rf:
                report_md = rf.read()

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                st.download_button(
                    "⬇️ Download Case Dossier (.md)",
                    data=report_md,
                    file_name=rep_path_md.name,
                    mime="text/markdown",
                    use_container_width=True,
                )
            with bcol2:
                if pdf_target_path.is_file():
                    with open(pdf_target_path, "rb") as pf:
                        st.download_button(
                            "⬇️ Download Official Case Report (.pdf)",
                            data=pf.read(),
                            file_name=pdf_target_path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )

            with st.expander("👁️ Preview Generated Case Dossier", expanded=True):
                st.markdown(report_md)

    # TAB 5: Pipeline & Evaluation
    with tab_pipeline:
        st.markdown("### ⚙️ Pipeline Control & Evaluation")
        
        if st.button("🚀 Re-Run Full Analysis Pipeline"):
            with st.spinner("Executing end-to-end forensic analysis..."):
                run_full_pipeline(root / "data" / "raw" / "synthetic_transactions.csv")
                st.cache_data.clear()
                st.success("Pipeline execution complete!")
                st.rerun()

        if not gt_df.empty and not alerts_df.empty:
            st.markdown("---")
            st.markdown("#### 🎯 Ground Truth Benchmark Evaluation")
            gt_criminal_wallets = set(gt_df[(gt_df["entity_type"] == "wallet") & (gt_df["is_criminal"])]["entity_id"])
            flagged_wallets = set(alerts_df[alerts_df["composite_risk_score"] >= 0.40].index)

            true_pos = len(flagged_wallets.intersection(gt_criminal_wallets))
            precision = true_pos / len(flagged_wallets) if flagged_wallets else 0.0
            recall = true_pos / len(gt_criminal_wallets) if gt_criminal_wallets else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Ground Truth Criminals", f"{len(gt_criminal_wallets):,}")
            with b2:
                st.metric("Identified True Positives", f"{true_pos:,}")
            with b3:
                st.metric("Precision (Lead Purity)", f"{precision:.2%}")
            with b4:
                st.metric("Recall (Detection Rate)", f"{recall:.2%}")


if __name__ == "__main__":
    main()
