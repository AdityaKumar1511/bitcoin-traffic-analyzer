"""
Forensic Case Report Generator Module.

Exports court-ready forensic case dossiers in both PDF and Markdown formats,
summarizing on-chain metadata, network correlation, evasion signals, and multi-hop taint paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, List, Optional, Union

import pandas as pd

# Ensure repository root is in Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import get_logger  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


logger = get_logger(__name__)


def generate_pdf_report(
    wallet_id: str,
    row: pd.Series,
    output_path: Path,
) -> Path:
    """
    Generate an official, styled PDF forensic evidence dossier using ReportLab.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Custom styles matching cyber intelligence forensic document theme
    coral_alert = colors.HexColor("#FF4B4B")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )

    heading2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0F5132"),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
    )

    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
    )

    elements: List[Any] = []

    # Header Banner
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    case_id = f"NTRO-BTC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{wallet_id[:8].upper()}"

    elements.append(Paragraph("NATIONAL TECHNICAL RESEARCH ORGANISATION (NTRO)", title_style))
    elements.append(Paragraph(f"BLOCKCHAIN TRANSACTION FORENSIC DOSSIER · REF: <b>{case_id}</b>", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#059669"), spaceAfter=10))

    # Case Metadata Table
    risk_score = float(row.get("composite_risk_score", 0.0))
    risk_level = str(row.get("risk_level", "UNKNOWN"))
    primary_ip = str(row.get("primary_ip", "N/A"))
    primary_asn = str(row.get("primary_asn", "N/A"))
    asn_cat = str(row.get("asn_category", "N/A"))
    taint_src = str(row.get("taint_source", "None"))
    taint_hops = int(row.get("taint_hops", 0))
    taint_path = str(row.get("taint_path", "Direct / None"))
    analyst_status = str(row.get("analyst_status", "PENDING"))
    analyst_notes = str(row.get("analyst_notes", "No notes recorded."))
    patterns = str(row.get("pattern_types", "None"))

    meta_table_data = [
        [
            Paragraph("<b>Target Entity (Wallet):</b>", body_style),
            Paragraph(f"<font name='Courier'>{wallet_id}</font>", body_style),
            Paragraph("<b>Generated:</b>", body_style),
            Paragraph(now_utc, body_style),
        ],
        [
            Paragraph("<b>Composite Risk Score:</b>", body_style),
            Paragraph(f"<b><font color='{coral_alert.hexval()}'>{risk_score:.4f}</font></b> ({risk_level})", body_style),
            Paragraph("<b>Review Status:</b>", body_style),
            Paragraph(f"<b>{analyst_status}</b>", body_style),
        ],
        [
            Paragraph("<b>Primary Broadcast IP:</b>", body_style),
            Paragraph(f"<font name='Courier'>{primary_ip}</font>", body_style),
            Paragraph("<b>Origin ASN:</b>", body_style),
            Paragraph(f"AS{primary_asn} ({asn_cat})", body_style),
        ],
        [
            Paragraph("<b>Evasion Signatures:</b>", body_style),
            Paragraph(patterns, body_style),
            Paragraph("<b>Taint Distance:</b>", body_style),
            Paragraph(f"{taint_hops} hops from {taint_src[:10]}...", body_style),
        ],
    ]

    t_meta = Table(meta_table_data, colWidths=[1.6 * inch, 2.1 * inch, 1.4 * inch, 2.1 * inch])
    t_meta.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    elements.append(t_meta)
    elements.append(Spacer(1, 12))

    # Executive Summary Box
    elements.append(Paragraph("1. Executive Risk Summary", heading2_style))
    reason_p = Paragraph(f"<b>Summary Finding:</b> {row.get('primary_reason', 'N/A')}", body_style)
    t_reason = Table([[reason_p]], colWidths=[7.2 * inch])
    t_reason.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ECFDF5")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#A7F3D0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(t_reason)
    elements.append(Spacer(1, 10))

    # Multi-Hop Taint Pathway
    elements.append(Paragraph("2. Multi-Hop Risk Taint Provenance", heading2_style))
    path_p = Paragraph(f"<font name='Courier'>{taint_path}</font>", code_style)
    t_path = Table([[path_p]], colWidths=[7.2 * inch])
    t_path.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(t_path)
    elements.append(Spacer(1, 10))

    # Detailed Forensic Justification
    elements.append(Paragraph("3. Detailed Forensic Evidence Breakdown", heading2_style))
    just_text = str(row.get("detailed_justification", row.get("primary_reason", "N/A")))
    import re
    for line in just_text.split("\n"):
        if line.strip():
            # Safely convert markdown formatting to XML
            clean_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            clean_line = clean_line.replace("•", "&bull;")
            # Convert **bold** to <b>bold</b>
            clean_line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", clean_line)
            # Convert `code` to <font name='Courier'>code</font>
            clean_line = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", clean_line)
            elements.append(Paragraph(clean_line, body_style))
            elements.append(Spacer(1, 2))

    elements.append(Spacer(1, 14))

    # Investigator Attestation Block
    elements.append(Paragraph("4. Investigator Sign-Off & Audit Certification", heading2_style))
    attestation_data = [
        [
            Paragraph("<b>Analyst Evaluation Notes:</b>", body_style),
            Paragraph(analyst_notes, body_style),
        ],
        [
            Paragraph("<b>Investigator Signature:</b>", body_style),
            Paragraph("________________________________________", body_style),
        ],
        [
            Paragraph("<b>Date / Security Clearance:</b>", body_style),
            Paragraph(f"{now_utc} · LEVEL-3 CYBER FORENSICS", body_style),
        ],
    ]
    t_attest = Table(attestation_data, colWidths=[2.2 * inch, 5.0 * inch])
    t_attest.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ])
    )
    elements.append(KeepTogether([t_attest]))

    doc.build(elements)
    logger.info("PDF Case report generated at: %s", output_path)
    return output_path


def generate_case_report(
    wallet_id: str,
    alerts_df: Optional[pd.DataFrame] = None,
    output_dir: Union[str, Path] = "reports",
    format: str = "both",  # "pdf", "md", "both"
) -> Path:
    """
    Generate forensic case report in PDF, Markdown, or both formats.
    """
    project_root = get_project_root()
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    if alerts_df is None:
        alerts_csv = project_root / "data" / "processed" / "scored_alerts.csv"
        if alerts_csv.is_file():
            alerts_df = pd.read_csv(alerts_csv, index_col=0)
        else:
            raise FileNotFoundError(f"Alerts file not found at {alerts_csv}. Please run pipeline first.")

    if wallet_id not in alerts_df.index:
        matches = alerts_df[alerts_df.index.astype(str) == str(wallet_id)]
        if matches.empty:
            raise ValueError(f"Wallet '{wallet_id}' not found in alerts database.")
        row = matches.iloc[0]
    else:
        row = alerts_df.loc[wallet_id]

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    case_id = f"CASE-BTC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{wallet_id[:8].upper()}"

    risk_score = float(row.get("composite_risk_score", 0.0))
    risk_level = str(row.get("risk_level", "UNKNOWN"))
    primary_reason = str(row.get("primary_reason", "N/A"))
    detailed_justification = str(row.get("detailed_justification", "N/A"))
    primary_ip = str(row.get("primary_ip", "N/A"))
    primary_asn = str(row.get("primary_asn", "N/A"))
    asn_category = str(row.get("asn_category", "N/A"))
    taint_source = str(row.get("taint_source", "None"))
    taint_hops = int(row.get("taint_hops", 0))
    taint_path = str(row.get("taint_path", "Direct / None"))
    analyst_status = str(row.get("analyst_status", "PENDING"))
    analyst_notes = str(row.get("analyst_notes", "None recorded."))

    # Generate Markdown Dossier
    report_content = f"""# 🛡️ NATIONAL TECHNICAL RESEARCH ORGANISATION (NTRO)
## BLOCKCHAIN TRANSACTION FORENSIC DOSSIER

---

### **CASE IDENTIFICATION**
- **Case Reference:** `{case_id}`
- **Investigation Subject (Wallet):** `{wallet_id}`
- **Generated At:** `{now_utc}`
- **Classification Status:** `OFFICIAL / LAW ENFORCEMENT SENSITIVE`

---

### **1. EXECUTIVE RISK ASSESSMENT**
| Metric | Assessment |
|---|---|
| **Composite Risk Score** | **`{risk_score:.4f}`** (Scale 0.0 – 1.0) |
| **Risk Priority Tier** | **`{risk_level}`** |
| **Analyst Review Status** | **`{analyst_status}`** |
| **Identified Evasion Patterns** | `{row.get('pattern_types', 'None')}` |

#### **Summary Finding:**
> {primary_reason}

---

### **2. NETWORK & GEOLOCATION CORRELATION**
| Signal | Value | Forensic Evaluation |
|---|---|---|
| **Primary Broadcast IP** | `{primary_ip}` | Relay node observed broadcasting transactions |
| **Origin ASN** | `AS{primary_asn}` | Autonomous System identifier |
| **Infrastructure Category** | `{asn_category}` | Network risk classification |

---

### **3. MULTI-HOP TAINT TRACEABILITY**
- **Taint Seed Source:** `{taint_source}`
- **Graph Distance:** `{taint_hops}` transaction hops
- **Provenance Pathway:**
```
{taint_path}
```

---

### **4. DETAILED FORENSIC JUSTIFICATION & EVIDENCE MATRIX**
{detailed_justification}

---

### **5. INVESTIGATOR ATTESTATION & AUDIT LOG**
- **Analyst Notes:** {analyst_notes}
- **Investigator Signature:** ___________________________
- **Date / Time:** `{now_utc}`

*Confidential Investigative Artifact Generated by Bitcoin Traffic Analyzer v2.0 (Offline Mode)*
"""

    md_file = out_path / f"case_report_{wallet_id[:12]}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    pdf_file = out_path / f"case_report_{wallet_id[:12]}.pdf"
    try:
        generate_pdf_report(wallet_id, row, pdf_file)
    except Exception as err:
        logger.warning("PDF generation fallback: %s", err)

    if format == "pdf" and pdf_file.is_file():
        return pdf_file
    return md_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Export forensic case report (PDF & Markdown) for a flagged wallet.")
    parser.add_argument(
        "--wallet-id",
        "-w",
        type=str,
        required=True,
        help="Target wallet address to generate report for",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reports",
        help="Directory to save generated case reports",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["pdf", "md", "both"],
        default="both",
        help="Output report format (default: both)",
    )
    args = parser.parse_args()

    report_path = generate_case_report(args.wallet_id, output_dir=args.output, format=args.format)
    print("\n[+] Forensic case report successfully generated:")
    print(f"    Path: {report_path}\n")


if __name__ == "__main__":
    main()
