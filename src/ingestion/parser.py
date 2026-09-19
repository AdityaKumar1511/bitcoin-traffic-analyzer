"""
Data Ingestion and Normalization Parser for Bitcoin Transactions.

This module provides functions to ingest Bitcoin transaction data from CSV, JSON,
or XML sources and normalize them into a uniform pandas DataFrame.

Standardized Final DataFrame Schema:
- timestamp: datetime64[ns, UTC]
- src_ip: str
- dst_ip: str
- src_port: int
- dst_port: int
- txid: str
- input_addresses: list[str]
- output_addresses: list[str]
- input_amounts: list[float]
- output_amounts: list[float]
- fee: float (may contain NaN if missing)
- script_type: str
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Expected canonical columns in order
EXPECTED_COLUMNS: List[str] = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "txid",
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
    "fee",
    "script_type",
]


def _parse_address_list(val: Any) -> List[str]:
    """Parse string or sequence into a list of wallet addresses."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, np.ndarray, set)):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str:
            return []
        delimiter = ";" if ";" in val_str else ("," if "," in val_str else None)
        if delimiter:
            return [part.strip() for part in val_str.split(delimiter) if part.strip()]
        return [val_str]
    return [str(val).strip()]


def _parse_amount_list(val: Any) -> List[float]:
    """Parse string, number, or sequence into a list of float amounts."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (int, float)):
        return [float(val)]
    if isinstance(val, (list, tuple, np.ndarray, set)):
        result = []
        for x in val:
            try:
                if x is not None and str(x).strip() != "":
                    result.append(float(x))
            except (ValueError, TypeError) as err:
                logger.warning("Skipping invalid float amount in list '%s': %s", x, err)
        return result
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str:
            return []
        delimiter = ";" if ";" in val_str else ("," if "," in val_str else None)
        parts = val_str.split(delimiter) if delimiter else [val_str]
        result = []
        for part in parts:
            part_str = part.strip()
            if not part_str:
                continue
            try:
                result.append(float(part_str))
            except (ValueError, TypeError) as err:
                logger.warning("Skipping invalid float amount string '%s': %s", part_str, err)
        return result
    try:
        return [float(val)]
    except (ValueError, TypeError):
        return []


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an ingested raw DataFrame into the canonical schema and data types.

    Args:
        df: Raw DataFrame ingested from CSV, JSON, or XML.

    Returns:
        Canonicalized pd.DataFrame with guaranteed columns and data types.
    """
    # Ensure all expected columns exist, creating missing ones if necessary
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            logger.warning("Expected column '%s' missing from input. Initializing with defaults.", col)
            df[col] = np.nan

    # Reorder columns to canonical order
    df = df[EXPECTED_COLUMNS].copy()

    # Normalize data types
    # 1. timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # 2. String fields
    df["src_ip"] = df["src_ip"].fillna("").astype(str)
    df["dst_ip"] = df["dst_ip"].fillna("").astype(str)
    df["txid"] = df["txid"].fillna("").astype(str)
    df["script_type"] = df["script_type"].fillna("").astype(str)

    # 3. Integer ports
    df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce").fillna(0).astype(int)
    df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).astype(int)

    # 4. List fields
    df["input_addresses"] = df["input_addresses"].apply(_parse_address_list)
    df["output_addresses"] = df["output_addresses"].apply(_parse_address_list)
    df["input_amounts"] = df["input_amounts"].apply(_parse_amount_list)
    df["output_amounts"] = df["output_amounts"].apply(_parse_amount_list)

    # 5. Float fee (keep float NaN if missing)
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce").astype(float)

    return df


def parse_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Parse a Bitcoin transaction CSV file into a normalized DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Normalized pd.DataFrame.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Read CSV treating all columns initially as object/string to prevent unintended parsing
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return normalize_dataframe(df)


def parse_json(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Parse a Bitcoin transaction JSON file into a normalized DataFrame.

    Supports both top-level list of objects and a dictionary containing
    a 'transactions' key with that list.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Normalized pd.DataFrame.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "transactions" in data and isinstance(data["transactions"], list):
            tx_list = data["transactions"]
        else:
            # Check if dict is keyed by txid or single transaction
            if any(k in EXPECTED_COLUMNS for k in data.keys()):
                tx_list = [data]
            else:
                logger.warning("Unrecognized JSON object structure. Attempting to parse dictionary values.")
                tx_list = list(data.values())
    elif isinstance(data, list):
        tx_list = data
    else:
        raise ValueError(f"Invalid JSON content in {path}: expected list or dict with 'transactions' key.")

    df = pd.DataFrame(tx_list)
    return normalize_dataframe(df)


def parse_xml(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Parse a Bitcoin transaction XML file into a normalized DataFrame.

    Supports root element containing repeated <transaction> elements, with fields
    either as semicolon-separated text or child elements.

    Args:
        filepath: Path to the XML file.

    Returns:
        Normalized pd.DataFrame.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"XML file not found: {path}")

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as err:
        raise ValueError(f"Malformed XML file {path}: {err}") from err

    records: List[Dict[str, Any]] = []

    # Find transaction elements (either direct children or recursive)
    tx_elements = root.findall(".//transaction")
    if not tx_elements and root.tag == "transaction":
        tx_elements = [root]
    elif not tx_elements:
        # Fallback: each child of root might be a transaction
        tx_elements = list(root)

    for elem in tx_elements:
        row: Dict[str, Any] = {}
        for child in elem:
            tag = child.tag.strip()
            # If child has nested elements (e.g. repeated address or amount tags)
            if len(child) > 0:
                child_texts = [sub.text.strip() for sub in child if sub.text and sub.text.strip()]
                row[tag] = child_texts
            else:
                text_val = child.text.strip() if child.text else ""
                row[tag] = text_val

        # Handle alternative repeated child tags directly under transaction
        # (e.g. <input_address>addr1</input_address><input_address>addr2</input_address>)
        if "input_addresses" not in row:
            addrs = [c.text.strip() for c in elem.findall("input_address") if c.text and c.text.strip()]
            if addrs:
                row["input_addresses"] = addrs
        if "output_addresses" not in row:
            addrs = [c.text.strip() for c in elem.findall("output_address") if c.text and c.text.strip()]
            if addrs:
                row["output_addresses"] = addrs
        if "input_amounts" not in row:
            amts = [c.text.strip() for c in elem.findall("input_amount") if c.text and c.text.strip()]
            if amts:
                row["input_amounts"] = amts
        if "output_amounts" not in row:
            amts = [c.text.strip() for c in elem.findall("output_amount") if c.text and c.text.strip()]
            if amts:
                row["output_amounts"] = amts

        records.append(row)

    df = pd.DataFrame(records)
    return normalize_dataframe(df)


def parse_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Auto-detect format from file extension and parse into a normalized DataFrame.

    Supported formats: .csv, .json, .xml

    Args:
        filepath: Path to the input file.

    Returns:
        Normalized pd.DataFrame.

    Raises:
        ValueError: If file extension is unsupported.
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        return parse_csv(path)
    elif ext == ".json":
        return parse_json(path)
    elif ext == ".xml":
        return parse_xml(path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}' for file: {path}. "
            "Expected one of: .csv, .json, .xml"
        )


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Ingest and normalize Bitcoin transactions from CSV, JSON, or XML."
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
    """CLI entry point for testing parser standalone."""
    args = _build_parser().parse_args()
    try:
        df = parse_file(args.input)
        print("Successfully ingested and normalized transaction data.")
        print(f"DataFrame Shape: {df.shape}")
        print("\nDataFrame dtypes:")
        print(df.dtypes)
        print("\nFirst 3 rows:")
        print(df.head(3))
    except Exception as exc:
        logger.error("Failed to parse file '%s': %s", args.input, exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
