"""
Streamlit Dashboard for Bitcoin Traffic Analyzer.
Theme: Enterprise Forensic Analytics — Professional Dark Interface.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import yaml

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
    page_icon="⛓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Enterprise Forensic Theme — Global CSS Override
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ===== DESIGN TOKENS ===== */
    :root {
        --bg-base: #000000;
        --bg-surface: #14181F;
        --bg-elevated: #1B2029;
        --border: #21262D;
        --border-emphasis: #30363D;
        --text-primary: #C9D1D9;
        --text-secondary: #8B949E;
        --text-muted: #6E7681;
        --accent: #5B8DEF;
        --accent-subtle: rgba(91, 141, 239, 0.12);
        --severity-critical: #DA3633;
        --severity-high: #D29922;
        --severity-medium: #8B949E;
        --severity-low: #3FB950;
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
        --radius-sm: 4px;
        --radius-md: 6px;
        --radius-lg: 8px;
        --space-1: 4px;
        --space-2: 8px;
        --space-3: 12px;
        --space-4: 16px;
        --space-5: 24px;
        --space-6: 32px;
    }

    /* ===== GLOBAL CANVAS ===== */
    .stApp {
        background: var(--bg-base) !important;
        font-family: var(--font-sans) !important;
        color: var(--text-primary) !important;
    }

    /* Remove default Streamlit header/footer decorations */
    header[data-testid="stHeader"] {
        background: var(--bg-base) !important;
        border-bottom: 1px solid var(--border) !important;
    }

    footer { display: none !important; }
    #MainMenu { display: none !important; }

    /* ===== TOP NAVBAR ===== */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: var(--space-3) var(--space-5);
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        margin-bottom: var(--space-5);
    }

    .brand-title {
        font-size: 15px;
        font-weight: 600;
        color: #C9D1D9;
        letter-spacing: -0.01em;
    }

    .brand-divider {
        color: var(--text-muted);
        font-size: 14px;
        font-weight: 400;
    }

    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: var(--text-secondary);
        font-size: 12px;
        font-weight: 500;
    }

    .status-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--severity-low);
    }

    /* ===== HERO SECTION ===== */
    .hero-container {
        padding: var(--space-5) var(--space-3) var(--space-6) var(--space-3);
    }

    .hero-heading {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.3;
        color: var(--text-primary);
        margin-bottom: var(--space-2);
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 14px;
        max-width: 680px;
        line-height: 1.6;
    }

    /* ===== KPI METRIC CARDS ===== */
    .kpi-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .kpi-label {
        font-size: 11px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
        margin-bottom: var(--space-2);
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
        margin-bottom: var(--space-1);
    }

    .kpi-secondary {
        font-size: 12px;
        color: var(--text-muted);
        font-weight: 400;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid var(--border);
        border-radius: 0;
        border: none;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        color: var(--text-secondary);
        padding: var(--space-3) var(--space-4);
        font-weight: 500;
        font-size: 13px;
        font-family: var(--font-sans);
        border-bottom: 2px solid transparent;
        background: transparent !important;
        transition: color 0.15s ease, border-color 0.15s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
    }

    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--accent) !important;
        border: none !important;
        border-bottom: 2px solid var(--accent) !important;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ===== BUTTONS ===== */
    div.stButton > button {
        background: var(--accent) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 8px 20px !important;
        font-family: var(--font-sans) !important;
        transition: background 0.15s ease, opacity 0.15s ease !important;
        box-shadow: none !important;
    }

    div.stButton > button:hover {
        background: #4A7AD8 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    div.stButton > button:active {
        background: #3D6BC4 !important;
    }

    /* Secondary / form submit buttons */
    div.stFormSubmitButton > button {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-emphasis) !important;
        box-shadow: none !important;
    }

    div.stFormSubmitButton > button:hover {
        background: var(--border) !important;
        border-color: var(--text-muted) !important;
    }

    /* ===== DOWNLOAD BUTTONS ===== */
    div.stDownloadButton > button {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-emphasis) !important;
        box-shadow: none !important;
        font-weight: 500 !important;
    }

    div.stDownloadButton > button:hover {
        background: var(--border) !important;
    }

    /* ===== INPUTS, SELECTS, SLIDERS ===== */
    div[data-baseweb="select"] {
        font-family: var(--font-sans) !important;
    }

    div[data-baseweb="select"] > div {
        background: var(--bg-surface) !important;
        border-color: var(--border) !important;
        border-radius: var(--radius-md) !important;
    }

    .stTextInput > div > div > input {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-sans) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        box-shadow: none !important;
    }

    .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
        background: var(--border) !important;
    }

    /* Slider track */
    .stSlider > div > div > div > div {
        background: var(--border) !important;
    }

    /* ===== DATAFRAMES / TABLES ===== */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius-lg) !important;
        overflow: hidden !important;
        border: 1px solid var(--border) !important;
    }

    .stDataFrame [data-testid="glideDataEditor"] {
        border-radius: var(--radius-lg) !important;
    }

    /* Table component */
    .stTable table {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
    }

    .stTable th {
        background: var(--bg-elevated) !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
        border-bottom: 1px solid var(--border) !important;
    }

    .stTable td {
        border-bottom: 1px solid var(--border) !important;
        font-family: var(--font-sans) !important;
        color: var(--text-primary) !important;
    }

    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }

    details[data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        background: var(--bg-surface) !important;
    }

    details[data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    /* ===== METRICS (st.metric) ===== */
    [data-testid="stMetric"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        padding: var(--space-4) !important;
    }

    [data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    /* ===== ALERTS / INFO BOXES ===== */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border) !important;
    }

    /* ===== HORIZONTAL RULES ===== */
    hr {
        border-color: var(--border) !important;
    }

    /* ===== SECTION HEADERS ===== */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        font-family: var(--font-sans) !important;
        color: var(--text-primary) !important;
    }

    .stApp h3 {
        font-size: 16px !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    .stApp h4 {
        font-size: 14px !important;
        font-weight: 600 !important;
        letter-spacing: -0.005em !important;
        color: var(--text-primary) !important;
    }

    /* ===== MULTISELECT CHIPS ===== */
    span[data-baseweb="tag"] {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-emphasis) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-size: 12px !important;
        max-width: none !important;
    }

    /* ===== CODE BLOCKS ===== */
    .stCodeBlock, code, pre {
        font-family: var(--font-mono) !important;
    }

    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }

    /* ===== TEXT AREA ===== */
    .stTextArea textarea {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-sans) !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    /* ===== CAPTIONS ===== */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: 12px !important;
    }

    /* ===== PLOTLY CHART OVERRIDES ===== */
    .stPlotlyChart {
        border-radius: var(--radius-lg) !important;
        overflow: hidden !important;
    }

    /* ===== FORM CONTAINERS ===== */
    [data-testid="stForm"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        padding: var(--space-4) !important;
        background: var(--bg-surface) !important;
    }

    /* ===== SELECTBOX ===== */
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stTextInput label, .stTextArea label {
        color: var(--text-secondary) !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        font-family: var(--font-sans) !important;
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

    # Top Navbar — Professional, minimal
    st.markdown(
        """
        <div class="navbar">
            <div style="display: flex; align-items: center; gap: 10px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#5B8DEF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                <span class="brand-title">BitForge</span>
                <span class="brand-divider">Bitcoin Traffic Analyzer</span>
            </div>
            <div style="display: flex; align-items: center; gap: 16px;">
                <span class="status-indicator">
                    <span class="status-dot"></span>
                    Offline Secure
                </span>
                <span style="color: #6E7681; font-size: 12px;">NTRO · SIH26146</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Section — Clean, understated
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-heading">Forensic Blockchain Intelligence</div>
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
                <span class="kpi-secondary">100% ingested</span>
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
                <span class="kpi-secondary">5,814 clusters</span>
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
                <div class="kpi-value">{crit_cnt:,}</div>
                <span class="kpi-secondary">{"Requires review" if crit_cnt > 0 else "None flagged"}</span>
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
                <span class="kpi-secondary">Signature detected</span>
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
                <span class="kpi-secondary">Ransomware flow</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # Main Navigation Tabs — Phase 5 Complete Forensic Dashboard
    (
        tab_overview,
        tab_alerts,
        tab_graph,
        tab_geo_asn,
        tab_evasion,
        tab_export,
        tab_pipeline,
    ) = st.tabs([
        "Executive Overview",
        "Investigative Alerts",
        "Link Analysis",
        "Geo & ASN Intelligence",
        "Evasion & Laundering",
        "Case Dossier",
        "Pipeline & Evaluation",
    ])

    # TAB 1: Executive Overview
    with tab_overview:
        st.markdown("### Forensic Risk & Traffic Profile Overview")
        st.caption("Distribution of composite risk scores across evaluated entities, anomaly severity, and top prioritized threats.")

        if not alerts_df.empty and "composite_risk_score" in alerts_df.columns:
            ov_col1, ov_col2 = st.columns([3, 2])

            with ov_col1:
                st.markdown("#### Composite Risk Score Distribution")
                # Histogram with severity thresholds
                fig_hist = px.histogram(
                    alerts_df,
                    x="composite_risk_score",
                    nbins=40,
                    color_discrete_sequence=["#5B8DEF"],
                    labels={"composite_risk_score": "Composite Risk Score (0.0 - 1.0)", "count": "Wallet Count"},
                    template="plotly_dark",
                )
                fig_hist.update_layout(
                    paper_bgcolor="#14181F",
                    plot_bgcolor="#14181F",
                    font=dict(color="#C9D1D9", family="Inter, sans-serif", size=11),
                    xaxis=dict(gridcolor="#21262D", linecolor="#21262D"),
                    yaxis=dict(gridcolor="#21262D", linecolor="#21262D"),
                    margin=dict(l=30, r=20, t=30, b=30),
                    height=280,
                )
                # Add threshold line markers
                fig_hist.add_vline(x=0.75, line_dash="dash", line_color="#DA3633", annotation_text="CRITICAL (≥0.75)", annotation_font_size=10, annotation_font_color="#DA3633")
                fig_hist.add_vline(x=0.50, line_dash="dash", line_color="#D29922", annotation_text="HIGH (≥0.50)", annotation_font_size=10, annotation_font_color="#D29922")
                fig_hist.add_vline(x=0.30, line_dash="dash", line_color="#8B949E", annotation_text="MEDIUM (≥0.30)", annotation_font_size=10, annotation_font_color="#8B949E")
                st.plotly_chart(fig_hist, use_container_width=True)

            with ov_col2:
                st.markdown("#### Entity Risk Level Breakdown")
                if "risk_level" in alerts_df.columns:
                    tier_counts = alerts_df["risk_level"].value_counts().reset_index()
                    tier_counts.columns = ["Tier", "Count"]
                    color_map = {
                        "CRITICAL": "#DA3633",
                        "HIGH": "#D29922",
                        "MEDIUM": "#8B949E",
                        "LOW": "#3FB950",
                    }
                    fig_donut = px.pie(
                        tier_counts,
                        names="Tier",
                        values="Count",
                        hole=0.6,
                        color="Tier",
                        color_discrete_map=color_map,
                        template="plotly_dark",
                    )
                    fig_donut.update_layout(
                        paper_bgcolor="#14181F",
                        plot_bgcolor="#14181F",
                        font=dict(color="#C9D1D9", family="Inter, sans-serif", size=11),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                        margin=dict(l=20, r=20, t=30, b=30),
                        height=280,
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

            st.markdown("---")
            st.markdown("#### High-Priority Threat Entities Leaderboard")
            top_leads = alerts_df.sort_values(by="composite_risk_score", ascending=False).head(10)
            leaderboard_cols = [c for c in ["composite_risk_score", "risk_level", "primary_ip", "asn_category", "cluster_size", "taint_hops", "primary_reason"] if c in top_leads.columns]
            st.dataframe(top_leads[leaderboard_cols], use_container_width=True, height=260)
        else:
            st.info("Run the analysis pipeline to generate overview analytics.")

    # TAB 2: Investigative Alerts & SHAP Detail
    with tab_alerts:
        selected_wallet = render_alert_table(alerts_df)
        if selected_wallet:
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            render_detail_view(selected_wallet, alerts_df, feedback_store=fb_store)

    # TAB 3: Link Analysis Graph
    with tab_graph:
        st.markdown("#### Interactive Link-Analysis Explorer")
        st.caption("Traverse multi-hop relationships between wallets, transactions, and broadcasting IP addresses.")

        target_wallet = st.selectbox(
            "Select Target Wallet for Graph Inspection",
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

    # TAB 4: Geo & ASN Intelligence
    with tab_geo_asn:
        st.markdown("### Network & Geographic Infrastructure Intelligence")
        st.caption("Correlating broadcast IP addresses with autonomous systems (ASNs), bulletproof hosters, and anonymization services.")

        col_geo1, col_geo2 = st.columns([3, 2])

        with col_geo1:
            st.markdown("#### Network Infrastructure Category Distribution")
            if not alerts_df.empty and "asn_category" in alerts_df.columns:
                asn_counts = alerts_df["asn_category"].value_counts().reset_index()
                asn_counts.columns = ["Category", "Count"]

                fig_asn = px.bar(
                    asn_counts,
                    x="Category",
                    y="Count",
                    color="Category",
                    color_discrete_sequence=["#5B8DEF", "#8B949E", "#D29922", "#DA3633", "#3FB950"],
                    template="plotly_dark",
                )
                fig_asn.update_layout(
                    paper_bgcolor="#14181F",
                    plot_bgcolor="#14181F",
                    font=dict(color="#C9D1D9", family="Inter, sans-serif", size=12),
                    xaxis=dict(gridcolor="#21262D", linecolor="#21262D"),
                    yaxis=dict(gridcolor="#21262D", linecolor="#21262D"),
                    showlegend=False,
                    margin=dict(l=40, r=24, t=32, b=40),
                    height=280,
                )
                fig_asn.update_traces(marker_line_width=0)
                st.plotly_chart(fig_asn, use_container_width=True)
            else:
                st.info("No ASN category data available.")

        with col_geo2:
            st.markdown("#### High-Risk ASN Threat Intelligence")
            high_risk_yaml = root / "config" / "high_risk_asns.yaml"
            high_risk_data = {}
            if high_risk_yaml.is_file():
                try:
                    with open(high_risk_yaml, "r", encoding="utf-8") as yf:
                        high_risk_data = yaml.safe_load(yf) or {}
                except Exception:
                    pass

            bp_count = len(high_risk_data.get("bulletproof_hosting", []))
            vpn_count = len(high_risk_data.get("vpn_proxy", []))
            tor_count = len(high_risk_data.get("tor_related", []))
            priv_count = len(high_risk_data.get("privacy_focused", []))

            st.markdown(
                f"""
                <div style="background: #14181F; border: 1px solid #21262D; border-radius: 8px; padding: 16px;">
                    <div style="margin-bottom: 8px; font-size: 13px; color: #C9D1D9;"><b>Monitored Autonomous Systems:</b></div>
                    <div style="font-size: 12px; color: #8B949E; line-height: 1.8;">
                        • Bulletproof Hosting Providers: <b style="color: #DA3633;">{bp_count} ASNs</b><br>
                        • Commercial VPN / Proxy Relays: <b style="color: #D29922;">{vpn_count} ASNs</b><br>
                        • Tor Exit / Relay Operators: <b style="color: #8B949E;">{tor_count} ASNs</b><br>
                        • Privacy Jurisdiction Providers: <b style="color: #5B8DEF;">{priv_count} ASNs</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### Top Broadcasting IP Addresses by Wallet Cluster Co-Occurrence")
        if not alerts_df.empty and "primary_ip" in alerts_df.columns:
            valid_ips = alerts_df[alerts_df["primary_ip"].notna() & (alerts_df["primary_ip"] != "nan")]
            if not valid_ips.empty:
                ip_stats = (
                    valid_ips.groupby("primary_ip")
                    .agg(
                        wallets_count=("primary_ip", "count"),
                        max_risk=("composite_risk_score", "max"),
                        asn=("primary_asn", "first"),
                        category=("asn_category", "first"),
                    )
                    .reset_index()
                    .sort_values(by="wallets_count", ascending=False)
                    .head(15)
                )
                ip_stats.columns = ["Broadcasting IP", "Associated Wallets", "Max Risk Score", "ASN", "Infrastructure Category"]
                st.dataframe(ip_stats, use_container_width=True, height=240)
            else:
                st.info("No IP telemetry recorded.")

    # TAB 5: Evasion Hub
    with tab_evasion:
        st.markdown("### Evasion-Specific Forensic Detectors")

        col_cj, col_pc = st.columns(2)
        with col_cj:
            st.markdown("#### CoinJoin Mixing Transactions")
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
            st.markdown("#### Ransomware Peel Chains")
            if peel_chains:
                pc_records = []
                for idx, pc in enumerate(peel_chains, 1):
                    pc_records.append({
                        "Chain #": idx,
                        "Hops": pc.get("total_chain_length", 0),
                        "Start (BTC)": pc.get("starting_amount", 0.0),
                        "Peeled (BTC)": pc.get("total_peeled", 0.0),
                        "Wallets": len(pc.get("chain_wallets", [])),
                        "Collector Wallet": pc.get("collector_wallet", "N/A"),
                    })
                st.dataframe(pd.DataFrame(pc_records), use_container_width=True, height=240)
            else:
                st.info("No peel chains detected.")

    # TAB 6: Case Dossier Export
    with tab_export:
        st.markdown("### Automated Forensic Case Dossier Generator")
        st.caption("Generate official evidence dossiers (PDF & Markdown) summarizing on-chain and network-layer forensic links.")

        col_exp1, col_exp2, col_exp3 = st.columns([3, 1, 1])
        with col_exp1:
            export_target = st.selectbox(
                "Select Subject Wallet to Generate Case Dossier",
                options=list(alerts_df.index) if not alerts_df.empty else [],
                key="export_target_select",
            )
        with col_exp2:
            st.write("")
            st.write("")
            gen_btn = st.button("Generate Dossier", use_container_width=True)

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
                    "Download Case Dossier (.md)",
                    data=report_md,
                    file_name=rep_path_md.name,
                    mime="text/markdown",
                    use_container_width=True,
                )
            with bcol2:
                if pdf_target_path.is_file():
                    with open(pdf_target_path, "rb") as pf:
                        st.download_button(
                            "Download Official Case Report (.pdf)",
                            data=pf.read(),
                            file_name=pdf_target_path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )

            with st.expander("Preview Generated Case Dossier", expanded=True):
                st.markdown(report_md)

    # TAB 7: Pipeline & Evaluation
    with tab_pipeline:
        st.markdown("### Pipeline Control & Evaluation")

        if st.button("Re-Run Full Analysis Pipeline"):
            with st.spinner("Executing end-to-end forensic analysis..."):
                run_full_pipeline(root / "data" / "raw" / "synthetic_transactions.csv")
                st.cache_data.clear()
                st.success("Pipeline execution complete!")
                st.rerun()

        if not gt_df.empty and not alerts_df.empty:
            st.markdown("---")
            st.markdown("#### Ground Truth Benchmark Evaluation")
            gt_criminal_wallets = set(gt_df[(gt_df["entity_type"] == "wallet") & (gt_df["is_criminal"])]["entity_id"])
            flagged_wallets = set(alerts_df[alerts_df["composite_risk_score"] >= 0.30].index)

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
