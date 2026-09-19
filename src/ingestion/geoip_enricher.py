# 1. Create a free account at https://www.maxmind.com/en/geolite2/signup
# 2. Download GeoLite2-Country.mmdb and GeoLite2-ASN.mmdb
# 3. Place both files in data/geoip/
# These files are gitignored and must be downloaded locally by each team member.

"""
Offline GeoIP and ASN Enrichment Module.

Enriches Bitcoin transaction IP addresses with country codes and Autonomous System
Numbers (ASN) using local MaxMind GeoLite2 databases (.mmdb) via the geoip2 library.
Operates completely offline without external network queries.
"""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
from typing import Any, Dict, Optional, Union

import geoip2.database
import geoip2.errors
import pandas as pd

from src.ingestion.parser import parse_file
from src.utils.logging_config import get_logger
from src.utils.paths import get_project_root, load_config

logger = get_logger(__name__)


class GeoIPEnricher:
    """
    Offline GeoIP and ASN database lookup enricher using MaxMind GeoLite2 databases.

    Opens country and ASN databases once on initialization for high-throughput lookups.
    Supports context manager protocol for automated resource cleanup.
    """

    def __init__(
        self,
        country_db_path: Optional[Union[str, Path]] = None,
        asn_db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize database readers.

        Args:
            country_db_path: Path to GeoLite2-Country.mmdb. If None, reads from config.
            asn_db_path: Path to GeoLite2-ASN.mmdb. If None, reads from config.

        Raises:
            FileNotFoundError: If either .mmdb database file cannot be located.
        """
        project_root = get_project_root()
        cfg = load_config()
        paths_cfg = cfg.get("paths", {})

        if country_db_path is None:
            country_db_path = paths_cfg.get(
                "geoip_country_db", "data/geoip/GeoLite2-Country.mmdb"
            )
        if asn_db_path is None:
            asn_db_path = paths_cfg.get(
                "geoip_asn_db", "data/geoip/GeoLite2-ASN.mmdb"
            )

        country_path = Path(country_db_path)
        if not country_path.is_absolute():
            country_path = project_root / country_path

        asn_path = Path(asn_db_path)
        if not asn_path.is_absolute():
            asn_path = project_root / asn_path

        instructions = (
            "Please download GeoLite2-Country.mmdb and GeoLite2-ASN.mmdb from MaxMind "
            "(https://www.maxmind.com/en/geolite2/signup) and place them in data/geoip/"
        )

        if not country_path.is_file():
            raise FileNotFoundError(
                f"GeoLite2 Country database not found at '{country_path}'. {instructions}"
            )

        if not asn_path.is_file():
            raise FileNotFoundError(
                f"GeoLite2 ASN database not found at '{asn_path}'. {instructions}"
            )

        self._country_path = country_path
        self._asn_path = asn_path

        self._country_reader = geoip2.database.Reader(str(self._country_path))
        self._asn_reader = geoip2.database.Reader(str(self._asn_path))
        logger.debug(
            "Opened GeoIP databases: country=%s, asn=%s",
            self._country_path,
            self._asn_path,
        )

    def lookup(self, ip: str) -> Dict[str, Optional[Union[str, int]]]:
        """
        Lookup country and ASN details for a single IP address string.

        Args:
            ip: IPv4 or IPv6 address string.

        Returns:
            dict: {
                "country": ISO country code (str) or None,
                "asn": Autonomous System Number (int) or None,
                "asn_org": Autonomous System Organization (str) or None,
            }
        """
        empty_result: Dict[str, Optional[Union[str, int]]] = {
            "country": None,
            "asn": None,
            "asn_org": None,
        }

        if not ip or not isinstance(ip, str):
            return empty_result

        clean_ip = ip.strip()

        # Validate IP syntax and filter non-routable / private ranges
        try:
            ip_obj = ipaddress.ip_address(clean_ip)
        except ValueError:
            logger.debug("Invalid IP address string: '%s'", clean_ip)
            return empty_result

        if (
            ip_obj.is_private
            or ip_obj.is_reserved
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            logger.debug("Skipping GeoIP lookup for private/reserved IP: %s", clean_ip)
            return empty_result

        country: Optional[str] = None
        asn: Optional[int] = None
        asn_org: Optional[str] = None

        # Country lookup
        try:
            resp_c = self._country_reader.country(clean_ip)
            country = (
                resp_c.country.iso_code
                or resp_c.registered_country.iso_code
                or resp_c.represented_country.iso_code
            )
        except geoip2.errors.AddressNotFoundError:
            logger.debug("IP address %s not found in Country database", clean_ip)
        except Exception as err:
            logger.debug("Unexpected error during Country lookup for %s: %s", clean_ip, err)

        # ASN lookup
        try:
            resp_a = self._asn_reader.asn(clean_ip)
            asn = resp_a.autonomous_system_number
            asn_org = resp_a.autonomous_system_organization
        except geoip2.errors.AddressNotFoundError:
            logger.debug("IP address %s not found in ASN database", clean_ip)
        except Exception as err:
            logger.debug("Unexpected error during ASN lookup for %s: %s", clean_ip, err)

        return {"country": country, "asn": asn, "asn_org": asn_org}

    def close(self) -> None:
        """Close both database readers cleanly."""
        if hasattr(self, "_country_reader") and self._country_reader is not None:
            self._country_reader.close()
        if hasattr(self, "_asn_reader") and self._asn_reader is not None:
            self._asn_reader.close()
        logger.debug("Closed GeoIP database readers.")

    def __enter__(self) -> "GeoIPEnricher":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def enrich_dataframe(
    df: pd.DataFrame,
    ip_column: str = "src_ip",
    enricher: Optional[GeoIPEnricher] = None,
) -> pd.DataFrame:
    """
    Enrich a DataFrame with '{ip_column}_country' and '{ip_column}_asn' columns.

    Reuses or creates a GeoIPEnricher instance and caches unique IP lookups
    for optimal performance on large datasets.

    Args:
        df: Input DataFrame containing the IP address column.
        ip_column: Name of the column with IP addresses (default: 'src_ip').
        enricher: Optional existing GeoIPEnricher instance.

    Returns:
        pd.DataFrame: A copy of the DataFrame with added country and ASN columns.
    """
    if ip_column not in df.columns:
        raise KeyError(f"Specified IP column '{ip_column}' not found in DataFrame.")

    should_close = False
    if enricher is None:
        enricher = GeoIPEnricher()
        should_close = True

    try:
        # Cache lookups for unique IPs to avoid redundant queries
        unique_ips = df[ip_column].dropna().unique()
        cache: Dict[str, Dict[str, Optional[Union[str, int]]]] = {}
        for ip in unique_ips:
            cache[str(ip)] = enricher.lookup(str(ip))

        result_df = df.copy()
        result_df[f"{ip_column}_country"] = result_df[ip_column].map(
            lambda x: cache.get(str(x), {}).get("country") if pd.notna(x) else None
        )
        result_df[f"{ip_column}_asn"] = result_df[ip_column].map(
            lambda x: cache.get(str(x), {}).get("asn") if pd.notna(x) else None
        )
        return result_df
    finally:
        if should_close:
            enricher.close()


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Enrich transaction IP addresses with GeoIP country and ASN metadata."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input transaction file (.csv, .json, or .xml).",
    )
    parser.add_argument(
        "--ip-col",
        type=str,
        default="src_ip",
        help="IP column name to enrich (default: src_ip).",
    )
    return parser


def main() -> None:
    """CLI entrypoint for standalone enrichment testing."""
    args = _build_parser().parse_args()

    print(f"[*] Ingesting transactions from: {args.input}")
    df = parse_file(args.input)

    print(f"[*] Enriching transactions using column '{args.ip_col}'...")
    try:
        enriched_df = enrich_dataframe(df, ip_column=args.ip_col)
    except FileNotFoundError as err:
        logger.error("%s", err)
        raise SystemExit(1) from err

    country_col = f"{args.ip_col}_country"
    asn_col = f"{args.ip_col}_asn"

    print("\n" + "=" * 55)
    print("           GEOIP ENRICHMENT SUMMARY")
    print("=" * 55)
    print(f"Total Transactions Enriched: {len(enriched_df):,}")
    print("\nTop Countries (value_counts):")
    counts = enriched_df[country_col].value_counts(dropna=False)
    print(counts.head(10))

    print("\nSample 5 Rows (IP -> Country -> ASN):")
    sample_cols = [args.ip_col, country_col, asn_col]
    sample = enriched_df[sample_cols].head(5)
    print(sample.to_string(index=False))
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
