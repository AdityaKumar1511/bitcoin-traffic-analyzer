"""
Detail View Component for Streamlit Dashboard — Enterprise Forensic Theme.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from src.feedback.feedback_store import FeedbackStore


def render_detail_view(
    entity_id: str,
    alerts_df: pd.DataFrame,
    feedback_store: Optional[FeedbackStore] = None,
) -> None:
    """
    Render comprehensive forensic deep-dive view for a selected wallet.
    """
    if entity_id not in alerts_df.index:
        st.error(f"Entity '{entity_id}' not found in alerts database.")
        return

    row = alerts_df.loc[entity_id]
    risk_score = float(row.get("composite_risk_score", 0.0))
    risk_lvl = str(row.get("risk_level", "LOW"))

    lvl_meta = {
        "CRITICAL": {"color": "#DA3633", "bg": "rgba(218, 54, 51, 0.12)", "border": "rgba(218, 54, 51, 0.30)"},
        "HIGH": {"color": "#D29922", "bg": "rgba(210, 153, 34, 0.12)", "border": "rgba(210, 153, 34, 0.30)"},
        "MEDIUM": {"color": "#8B949E", "bg": "rgba(139, 148, 158, 0.12)", "border": "rgba(139, 148, 158, 0.30)"},
        "LOW": {"color": "#3FB950", "bg": "rgba(63, 185, 80, 0.12)", "border": "rgba(63, 185, 80, 0.30)"},
    }
    badge_info = lvl_meta.get(risk_lvl, lvl_meta["LOW"])

    # Header Card
    st.markdown(
        f"""
        <div style="background: #14181F; padding: 20px 24px; border-radius: 8px;
                    border: 1px solid #21262D; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span style="color: #8B949E; font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 500;
                                 letter-spacing: 0.05em; text-transform: uppercase;">
                        Subject Target Address
                    </span>
                    <h3 style="margin: 4px 0 0 0; color: #C9D1D9; font-family: 'JetBrains Mono', monospace;
                               font-size: 16px; font-weight: 500;">
                        {entity_id}
                    </h3>
                </div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="text-align: right;">
                        <span style="font-size: 11px; color: #8B949E; text-transform: uppercase;
                                     letter-spacing: 0.05em; font-family: 'Inter', sans-serif;">Risk Score</span>
                        <div style="font-size: 20px; font-weight: 700; color: {badge_info['color']};
                                    font-family: 'Inter', sans-serif;">
                            {risk_score:.4f}
                        </div>
                    </div>
                    <span style="background: {badge_info['bg']}; color: {badge_info['color']};
                                 border: 1px solid {badge_info['border']};
                                 padding: 4px 12px; border-radius: 4px; font-weight: 600; font-size: 11px;
                                 letter-spacing: 0.03em; font-family: 'Inter', sans-serif;">
                        {risk_lvl}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)

    metric_card_style = (
        "background: #14181F; border: 1px solid #21262D; border-radius: 8px; "
        "padding: 16px; text-align: center; min-height: 88px; display: flex; "
        "flex-direction: column; justify-content: center;"
    )
    label_style = "font-size: 11px; color: #8B949E; font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 0.04em;"
    value_style = "margin: 6px 0 0 0; color: #C9D1D9; font-size: 20px; font-weight: 700; font-family: 'Inter', sans-serif;"

    with m1:
        st.markdown(
            f"""
            <div style="{metric_card_style}">
                <span style="{label_style}">Anomaly Score</span>
                <h4 style="{value_style}">{float(row.get('anomaly_score', 0.0)):.3f}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div style="{metric_card_style}">
                <span style="{label_style}">Evasion Score</span>
                <h4 style="{value_style}">{float(row.get('evasion_score', 0.0)):.3f}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div style="{metric_card_style}">
                <span style="{label_style}">Network Correlation</span>
                <h4 style="{value_style}">{float(row.get('network_correlation_score', 0.0)):.3f}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div style="{metric_card_style}">
                <span style="{label_style}">Taint Propagation</span>
                <h4 style="{value_style}">{float(row.get('taint_score', 0.0)):.3f}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Forensic Breakdown
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Forensic Lead Justification")
        st.info(row.get("detailed_justification", row.get("primary_reason", "No justification recorded.")))

        # Taint provenance trail
        taint_path = row.get("taint_path")
        if taint_path and pd.notna(taint_path) and str(taint_path).strip():
            st.markdown("#### Multi-Hop Taint Provenance Path")
            st.code(str(taint_path), language="text")

        # Associated Transactions
        txids = row.get("associated_txids", [])
        if isinstance(txids, list) and txids:
            st.markdown(f"#### Associated Transactions ({len(txids)})")
            st.dataframe(pd.DataFrame({"TXID": txids}), use_container_width=True, height=150)

    with col_right:
        st.markdown("#### Network & Entity Metadata")
        meta_data = {
            "Broadcasting IP": row.get("primary_ip", "N/A"),
            "ASN": row.get("primary_asn", "N/A"),
            "ASN Category": row.get("asn_category", "N/A"),
            "Entity Cluster ID": row.get("cluster_id", "Singleton"),
            "Cluster Size (Wallets)": row.get("cluster_size", 1),
            "Taint Seed Source": row.get("taint_source", "None"),
            "Taint Hop Distance": row.get("taint_hops", 0),
        }
        st.table(pd.Series(meta_data, name="Value"))

        # Analyst Feedback Form
        st.markdown("#### Analyst Review & Triage")
        current_status = str(row.get("analyst_status", "PENDING"))
        current_notes = str(row.get("analyst_notes", "")) if pd.notna(row.get("analyst_notes")) else ""

        with st.form(key=f"feedback_form_{entity_id}"):
            new_status = st.selectbox(
                "Mark Status:",
                options=["PENDING", "CONFIRMED", "FALSE_POSITIVE"],
                index=["PENDING", "CONFIRMED", "FALSE_POSITIVE"].index(current_status)
                if current_status in ["PENDING", "CONFIRMED", "FALSE_POSITIVE"]
                else 0,
            )
            new_notes = st.text_area("Analyst Notes:", value=current_notes, height=80)
            submitted = st.form_submit_button("Save & Recalibrate")

            if submitted and feedback_store:
                feedback_store.set_feedback(
                    entity_id=entity_id,
                    status=new_status,
                    entity_type="wallet",
                    notes=new_notes,
                )
                st.success(f"Feedback saved: marked as {new_status} and risk score recalibrated!")
                st.rerun()
