"""
Alert Table Component for Streamlit Dashboard — Enterprise Forensic Theme.
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
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
            <h3 style="margin: 0; color: #C9D1D9; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 16px; letter-spacing: -0.01em;">
                Prioritized Investigative Leads
            </h3>
            <span style="display: inline-flex; align-items: center; gap: 6px; color: #8B949E; font-size: 12px; font-weight: 500;">
                <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #3FB950;"></span>
                Live
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if alerts_df.empty:
        st.info("No alerts generated yet. Run the analysis pipeline to populate leads.")
        return None

    # Filter Controls
    with st.container():
        st.markdown(
            """
            <div style="background: #14181F; border: 1px solid #21262D; border-radius: 8px; padding: 16px 16px 8px 16px; margin-bottom: 16px;">
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
            search_query = st.text_input("Search Entity", placeholder="Address, IP, ASN...")

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
            <span style="font-size: 12px; color: #8B949E; font-family: 'Inter', sans-serif;">
                Showing <b style="color: #5B8DEF;">{len(filtered):,}</b> matching leads out of <b style="color: #C9D1D9;">{len(alerts_df):,}</b> evaluated wallets
            </span>
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
        "Select Entity for Deep-Dive Forensics",
        options=list(filtered.index),
        index=0,
    )
    return selected_entity
