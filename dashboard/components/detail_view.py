"""
Detail View Component for Streamlit Dashboard with BitForge Emerald/Mint Theme.
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
        "CRITICAL": {"color": "#FF4B4B", "bg": "rgba(255, 75, 75, 0.15)", "border": "#FF4B4B"},
        "HIGH": {"color": "#FFA500", "bg": "rgba(255, 165, 0, 0.15)", "border": "#FFA500"},
        "MEDIUM": {"color": "#FBBF24", "bg": "rgba(251, 191, 36, 0.15)", "border": "#FBBF24"},
        "LOW": {"color": "#00F29B", "bg": "rgba(0, 242, 155, 0.15)", "border": "#00F29B"},
    }
    badge_info = lvl_meta.get(risk_lvl, lvl_meta["LOW"])

    # Header Card with Emerald/Cyber Glassmorphism
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(13, 27, 19, 0.9) 0%, rgba(6, 15, 10, 0.95) 100%); 
                    padding: 20px; border-radius: 12px; border: 1px solid rgba(0, 242, 155, 0.25);
                    box-shadow: 0 10px 25px -5px rgba(0, 242, 155, 0.1); margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="color: #00F29B; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;">
                        Subject Target Address
                    </span>
                    <h3 style="margin: 4px 0 0 0; color: #FFFFFF; font-family: monospace; font-size: 1.25rem;">
                        {entity_id}
                    </h3>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="text-align: right;">
                        <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Risk Score</span>
                        <div style="font-size: 1.4rem; font-weight: 800; color: {badge_info['color']};">
                            {risk_score:.4f}
                        </div>
                    </div>
                    <span style="background: {badge_info['bg']}; color: {badge_info['color']}; border: 1px solid {badge_info['border']}; 
                                 padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                        {risk_lvl} PRIORITY
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div style="background: rgba(13, 27, 19, 0.6); border: 1px solid rgba(0, 242, 155, 0.15); border-radius: 10px; padding: 12px; text-align: center;">
                <span style="font-size: 0.75rem; color: #94A3B8;">Anomaly Score</span>
                <h4 style="margin: 4px 0 0 0; color: #00F29B; font-size: 1.3rem;">{float(row.get('anomaly_score', 0.0)):.3f}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div style="background: rgba(13, 27, 19, 0.6); border: 1px solid rgba(0, 242, 155, 0.15); border-radius: 10px; padding: 12px; text-align: center;">
                <span style="font-size: 0.75rem; color: #94A3B8;">Evasion Score</span>
                <h4 style="margin: 4px 0 0 0; color: #38BDF8; font-size: 1.3rem;">{float(row.get('evasion_score', 0.0)):.3f}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div style="background: rgba(13, 27, 19, 0.6); border: 1px solid rgba(0, 242, 155, 0.15); border-radius: 10px; padding: 12px; text-align: center;">
                <span style="font-size: 0.75rem; color: #94A3B8;">Network Correlation</span>
                <h4 style="margin: 4px 0 0 0; color: #A78BFA; font-size: 1.3rem;">{float(row.get('network_correlation_score', 0.0)):.3f}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div style="background: rgba(13, 27, 19, 0.6); border: 1px solid rgba(0, 242, 155, 0.15); border-radius: 10px; padding: 12px; text-align: center;">
                <span style="font-size: 0.75rem; color: #94A3B8;">Taint Propagation</span>
                <h4 style="margin: 4px 0 0 0; color: #F43F5E; font-size: 1.3rem;">{float(row.get('taint_score', 0.0)):.3f}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    # Forensic Breakdown
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 📝 Forensic Lead Justification")
        st.info(row.get("detailed_justification", row.get("primary_reason", "No justification recorded.")))

        # Taint provenance trail
        taint_path = row.get("taint_path")
        if taint_path and pd.notna(taint_path) and str(taint_path).strip():
            st.markdown("#### ⛓️ Multi-Hop Taint Provenance Path")
            st.code(str(taint_path), language="text")

        # Associated Transactions
        txids = row.get("associated_txids", [])
        if isinstance(txids, list) and txids:
            st.markdown(f"#### 🔗 Associated Transactions ({len(txids)})")
            st.dataframe(pd.DataFrame({"TXID": txids}), use_container_width=True, height=150)

    with col_right:
        st.markdown("#### 🌐 Network & Entity Metadata")
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
        st.markdown("#### 🧑‍✈️ Analyst Review & Triage")
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
            submitted = st.form_submit_button("💾 Save & Recalibrate")

            if submitted and feedback_store:
                feedback_store.set_feedback(
                    entity_id=entity_id,
                    status=new_status,
                    entity_type="wallet",
                    notes=new_notes,
                )
                st.success(f"Feedback saved: marked as {new_status} and risk score recalibrated!")
                st.rerun()
