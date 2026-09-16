"""
Alert Table Component for Streamlit Dashboard with BitForge Emerald/Mint Theme.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st


def render_alert_table(alerts_df: pd.DataFrame) -> Optional[str]:
    """
    Render filterable alerts table and return the selected wallet/entity ID.
    """
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #00F29B; box-shadow: 0 0 10px #00F29B;"></span>
                <h3 style="margin: 0; color: #F0FDF4; font-weight: 700; font-size: 1.3rem; letter-spacing: -0.02em;">
                    Prioritized Investigative Leads
                </h3>
            </div>
            <span style="background: rgba(0, 242, 155, 0.12); color: #00F29B; border: 1px solid rgba(0, 242, 155, 0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                LIVE SURVEILLANCE FEED
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if alerts_df.empty:
        st.info("No alerts generated yet. Run the analysis pipeline to populate leads.")
        return None

    # Filter Controls in Glassmorphic Container
    with st.container():
        st.markdown(
            """
            <div style="background: rgba(13, 27, 19, 0.7); border: 1px solid rgba(0, 242, 155, 0.15); border-radius: 12px; padding: 14px 16px 6px 16px; margin-bottom: 16px; backdrop-filter: blur(12px);">
            """,
            unsafe_allow_html=True,
        )
        col1, col2, col3, col4 = st.columns([2, 2, 2, 3])

        with col1:
            min_score = st.slider("Min Risk Score", min_value=0.0, max_value=1.0, value=0.30, step=0.05)

        with col2:
            risk_levels = st.multiselect(
                "Risk Levels",
                options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                default=["CRITICAL", "HIGH", "MEDIUM"],
            )

        with col3:
            status_filter = st.multiselect(
                "Review Status",
                options=["PENDING", "CONFIRMED", "FALSE_POSITIVE"],
                default=["PENDING", "CONFIRMED", "FALSE_POSITIVE"],
            )

        with col4:
            search_query = st.text_input("🔍 Search Entity", placeholder="Address, IP, ASN...")

        st.markdown("</div>", unsafe_allow_html=True)

    # Filter DataFrame
    filtered = alerts_df.copy()
    if "composite_risk_score" in filtered.columns:
        filtered = filtered[filtered["composite_risk_score"] >= min_score]
    if "risk_level" in filtered.columns and risk_levels:
        filtered = filtered[filtered["risk_level"].isin(risk_levels)]
    if "analyst_status" in filtered.columns and status_filter:
        filtered = filtered[filtered["analyst_status"].isin(status_filter)]

    if search_query.strip():
        q = search_query.strip().lower()
        mask = (
            filtered.index.astype(str).str.lower().str.contains(q)
            | filtered["primary_ip"].astype(str).str.lower().str.contains(q)
            | filtered["primary_reason"].astype(str).str.lower().str.contains(q)
            | filtered["asn_category"].astype(str).str.lower().str.contains(q)
        )
        filtered = filtered[mask]

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.85rem; color: #94A3B8;">Surfacing <b style="color: #00F29B;">{len(filtered):,}</b> matching leads out of <b style="color: #E2E8F0;">{len(alerts_df):,}</b> evaluated wallets</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.warning("No alerts match the selected filter criteria.")
        return None

    # Format table for display
    display_cols = [
        "composite_risk_score",
        "risk_level",
        "primary_ip",
        "asn_category",
        "taint_hops",
        "analyst_status",
        "primary_reason",
    ]
    existing_display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[existing_display_cols],
        use_container_width=True,
        height=320,
    )

    # Entity selection for deep-dive inspection
    selected_entity = st.selectbox(
        "🎯 Select Entity for Deep-Dive Forensics & Dossier Generation:",
        options=list(filtered.index),
        index=0,
    )
    return selected_entity
