"""
Transaction Data Quality Validation Module.

This module validates normalized Bitcoin transaction DataFrames produced by
the ingestion parser (src/ingestion/parser.py) without mutating or dropping rows.
It inspects for structural anomalies, corrupt identifiers, negative amounts,
address/amount length mismatches, and invalid IP addresses.
"""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Regular expression for 64-character hexadecimal transaction ID
TXID_HEX_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")


def _is_valid_ip(ip_str: Any) -> bool:
    """Validate whether the string represents a valid IPv4 or IPv6 address."""
    if not isinstance(ip_str, str) or not ip_str.strip():
        return False
    try:
        ipaddress.ip_address(ip_str.strip())
        return True
    except ValueError:
        return False


def _has_negative_amount(amounts: Any) -> bool:
    """Check if any numeric value in the amount sequence is negative."""
    if not isinstance(amounts, (list, tuple, np.ndarray)):
        return False
    for val in amounts:
        try:
            if float(val) < 0:
                return True
        except (ValueError, TypeError):
            continue
    return False


def validate_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate a normalized Bitcoin transaction DataFrame and report all data quality issues.

    Does NOT silently mutate or drop any rows.

    Args:
        df: Normalized DataFrame to validate.

    Returns:
        tuple[pd.DataFrame, dict]:
            - The original unmodified DataFrame.
            - A summary dictionary with identified issues and affected row indices.
    """
    total_rows = len(df)

    # 1. Check for null / NaN fields across all columns
    null_fields: Dict[str, int] = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            null_fields[col] = null_count

    malformed_txids: List[int] = []
    negative_amounts: List[int] = []
    mismatched_address_amount_counts: List[int] = []
    invalid_ips: List[int] = []

    # Iterate through DataFrame rows to detect issue indices
    for idx, row in df.iterrows():
        # A. txid 64-char hex format
        txid_val = row.get("txid", "")
        if not isinstance(txid_val, str) or not TXID_HEX_REGEX.match(txid_val.strip()):
            malformed_txids.append(idx)

        # B. Negative amounts
        in_amts = row.get("input_amounts", [])
        out_amts = row.get("output_amounts", [])
        fee_val = row.get("fee", 0.0)
        has_neg = _has_negative_amount(in_amts) or _has_negative_amount(out_amts)
        if not has_neg and pd.notna(fee_val):
            try:
                if float(fee_val) < 0:
                    has_neg = True
            except (ValueError, TypeError):
                pass
        if has_neg:
            negative_amounts.append(idx)

        # C. Address / Amount count mismatch
        in_addrs = row.get("input_addresses", [])
        out_addrs = row.get("output_addresses", [])
        in_addrs_len = len(in_addrs) if isinstance(in_addrs, (list, tuple)) else 0
        in_amts_len = len(in_amts) if isinstance(in_amts, (list, tuple)) else 0
        out_addrs_len = len(out_addrs) if isinstance(out_addrs, (list, tuple)) else 0
        out_amts_len = len(out_amts) if isinstance(out_amts, (list, tuple)) else 0

        if in_addrs_len != in_amts_len or out_addrs_len != out_amts_len:
            mismatched_address_amount_counts.append(idx)

        # D. IP address format check
        src_ip = row.get("src_ip", "")
        dst_ip = row.get("dst_ip", "")
        if not _is_valid_ip(src_ip) or not _is_valid_ip(dst_ip):
            invalid_ips.append(idx)

    # 2. Duplicate txid check (same txid appearing in multiple rows)
    if "txid" in df.columns:
        duplicate_mask = df["txid"].duplicated(keep=False) & (df["txid"].astype(str).str.strip() != "")
        duplicate_txids = df.loc[duplicate_mask, "txid"].unique().tolist()
    else:
        duplicate_txids = []

    # Log warnings for detected issues
    if null_fields:
        logger.warning(
            "Found null/NaN values across columns: %s",
            ", ".join(f"{col}={cnt}" for col, cnt in null_fields.items()),
        )
    if malformed_txids:
        logger.warning("Found %d rows with malformed txid", len(malformed_txids))
    if negative_amounts:
        logger.warning("Found %d rows with negative amounts", len(negative_amounts))
    if duplicate_txids:
        logger.warning("Found %d duplicated txid values across the dataset", len(duplicate_txids))
    if mismatched_address_amount_counts:
        logger.warning(
            "Found %d rows with mismatched address/amount counts",
            len(mismatched_address_amount_counts),
        )
    if invalid_ips:
        logger.warning("Found %d rows with invalid IP addresses", len(invalid_ips))

    total_issue_types = sum([
        1 if null_fields else 0,
        1 if malformed_txids else 0,
        1 if negative_amounts else 0,
        1 if duplicate_txids else 0,
        1 if mismatched_address_amount_counts else 0,
        1 if invalid_ips else 0,
    ])

    logger.info(
        "Validation completed: %d total rows checked, %d issue categories flagged.",
        total_rows,
        total_issue_types,
    )

    issues: Dict[str, Any] = {
        "total_rows": total_rows,
        "null_fields": null_fields,
        "malformed_txids": malformed_txids,
        "negative_amounts": negative_amounts,
        "duplicate_txids": duplicate_txids,
        "mismatched_address_amount_counts": mismatched_address_amount_counts,
        "invalid_ips": invalid_ips,
    }

    return df, issues


def get_clean_subset(df: pd.DataFrame, issues: Dict[str, Any]) -> pd.DataFrame:
    """
    Optionally filter a DataFrame to exclude rows flagged in any issue category.

    Args:
        df: The raw or normalized DataFrame.
        issues: The dictionary of issues returned by validate_transactions.

    Returns:
        Filtered DataFrame containing only clean rows.
    """
    bad_indices: Set[int] = set()

    for category in [
        "malformed_txids",
        "negative_amounts",
        "mismatched_address_amount_counts",
        "invalid_ips",
    ]:
        bad_indices.update(issues.get(category, []))

    # Exclude rows with duplicate txids
    dup_txids = set(issues.get("duplicate_txids", []))
    if dup_txids and "txid" in df.columns:
        bad_indices.update(df.index[df["txid"].isin(dup_txids)].tolist())

    # Exclude rows with any null/NaN in required columns
    if issues.get("null_fields"):
        bad_indices.update(df.index[df.isna().any(axis=1)].tolist())

    clean_df = df.drop(index=list(bad_indices)).copy()
    logger.info("Filtered dataset: retained %d of %d rows (%d dropped).", len(clean_df), len(df), len(bad_indices))
    return clean_df


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate transaction DataFrame data quality without silent dropping."
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
    """CLI entrypoint for running transaction validation."""
    args = _build_parser().parse_args()

    print(f"[*] Parsing input file: {args.input}")
    df = parse_file(args.input)

    print(f"[*] Running data quality validation on {len(df):,} transactions...")
    _, issues = validate_transactions(df)

    print("\n" + "=" * 55)
    print("           DATA QUALITY VALIDATION SUMMARY")
    print("=" * 55)
    print(f"Total Rows Evaluated: {issues['total_rows']:,}")
    print("-" * 55)
    print(f"Malformed TXIDs:               {len(issues['malformed_txids']):,}")
    print(f"Negative Amounts:              {len(issues['negative_amounts']):,}")
    print(f"Duplicate TXIDs:               {len(issues['duplicate_txids']):,}")
    print(f"Address/Amount Count Mismatch: {len(issues['mismatched_address_amount_counts']):,}")
    print(f"Invalid IP Addresses:          {len(issues['invalid_ips']):,}")
    print("Null / NaN Columns:")
    if issues["null_fields"]:
        for col, count in issues["null_fields"].items():
            print(f"  - {col}: {count:,} null rows")
    else:
        print("  - None")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
