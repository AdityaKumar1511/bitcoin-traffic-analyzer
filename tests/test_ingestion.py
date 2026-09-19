"""
Unit tests for data ingestion parser (CSV, JSON, XML).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.ingestion.parser import (
    EXPECTED_COLUMNS,
    normalize_dataframe,
    parse_csv,
    parse_file,
    parse_json,
    parse_xml,
)


class TestIngestionParser(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parse_csv_synthetic(self) -> None:
        csv_path = Path("data/raw/synthetic_transactions.csv")
        if not csv_path.exists():
            self.skipTest("Synthetic transactions file not present")

        df = parse_csv(csv_path)
        self.assertEqual(list(df.columns), EXPECTED_COLUMNS)
        self.assertEqual(len(df), 2000)

        # Check types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        self.assertEqual(str(df["timestamp"].dt.tz), "UTC")
        self.assertTrue(pd.api.types.is_integer_dtype(df["src_port"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["dst_port"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["fee"]))

        # Check list types in cells
        first_row = df.iloc[0]
        self.assertIsInstance(first_row["input_addresses"], list)
        self.assertIsInstance(first_row["output_addresses"], list)
        self.assertIsInstance(first_row["input_amounts"], list)
        self.assertIsInstance(first_row["output_amounts"], list)

        self.assertIsInstance(first_row["input_addresses"][0], str)
        self.assertIsInstance(first_row["input_amounts"][0], float)

    def test_parse_json_variations(self) -> None:
        # 1. Top-level list with arrays
        data_list = [
            {
                "timestamp": "2026-09-01T12:00:00Z",
                "src_ip": "1.2.3.4",
                "dst_ip": "5.6.7.8",
                "src_port": 8333,
                "dst_port": 8333,
                "txid": "a" * 64,
                "input_addresses": ["1addr1", "bc1addr2"],
                "output_addresses": ["1out1"],
                "input_amounts": [1.5, 2.5],
                "output_amounts": [3.999],
                "fee": 0.001,
                "script_type": "P2WPKH",
            }
        ]
        json_file1 = self.temp_path / "test1.json"
        json_file1.write_text(json.dumps(data_list))

        df1 = parse_json(json_file1)
        self.assertEqual(len(df1), 1)
        self.assertEqual(df1.iloc[0]["input_addresses"], ["1addr1", "bc1addr2"])
        self.assertEqual(df1.iloc[0]["input_amounts"], [1.5, 2.5])

        # 2. Dict with "transactions" key and semicolon strings
        data_dict = {
            "transactions": [
                {
                    "timestamp": "2026-09-02T15:30:00Z",
                    "src_ip": "9.10.11.12",
                    "dst_ip": "13.14.15.16",
                    "src_port": "12345",
                    "dst_port": "8333",
                    "txid": "b" * 64,
                    "input_addresses": "bc1in1;1in2",
                    "output_addresses": "1out1;1out2",
                    "input_amounts": "0.5;1.5",
                    "output_amounts": "1.0;0.999",
                    "fee": "0.001",
                    "script_type": "P2SH",
                }
            ]
        }
        json_file2 = self.temp_path / "test2.json"
        json_file2.write_text(json.dumps(data_dict))

        df2 = parse_json(json_file2)
        self.assertEqual(len(df2), 1)
        self.assertEqual(df2.iloc[0]["input_addresses"], ["bc1in1", "1in2"])
        self.assertEqual(df2.iloc[0]["input_amounts"], [0.5, 1.5])
        self.assertEqual(df2.iloc[0]["src_port"], 12345)

    def test_parse_xml_variations(self) -> None:
        # XML with semicolon strings and repeated tags
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <transactions>
            <transaction>
                <timestamp>2026-09-03T10:00:00Z</timestamp>
                <src_ip>100.1.2.3</src_ip>
                <dst_ip>100.4.5.6</dst_ip>
                <src_port>8333</src_port>
                <dst_port>8333</dst_port>
                <txid>c000000000000000000000000000000000000000000000000000000000000001</txid>
                <input_addresses>1addrA;1addrB</input_addresses>
                <output_addresses>1addrOut</output_addresses>
                <input_amounts>2.0;3.0</input_amounts>
                <output_amounts>4.9999</output_amounts>
                <fee>0.0001</fee>
                <script_type>P2PKH</script_type>
            </transaction>
            <transaction>
                <timestamp>2026-09-03T11:00:00Z</timestamp>
                <src_ip>100.1.2.4</src_ip>
                <dst_ip>100.4.5.7</dst_ip>
                <src_port>8333</src_port>
                <dst_port>8333</dst_port>
                <txid>c000000000000000000000000000000000000000000000000000000000000002</txid>
                <input_addresses>
                    <address>bc1sub1</address>
                    <address>bc1sub2</address>
                </input_addresses>
                <output_addresses>
                    <address>bc1subout</address>
                </output_addresses>
                <input_amounts>
                    <amount>1.25</amount>
                    <amount>0.75</amount>
                </input_amounts>
                <output_amounts>
                    <amount>1.999</amount>
                </output_amounts>
                <fee>0.001</fee>
                <script_type>P2WSH</script_type>
            </transaction>
        </transactions>
        """
        xml_file = self.temp_path / "test.xml"
        xml_file.write_text(xml_content)

        df = parse_xml(xml_file)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["input_addresses"], ["1addrA", "1addrB"])
        self.assertEqual(df.iloc[0]["input_amounts"], [2.0, 3.0])
        self.assertEqual(df.iloc[1]["input_addresses"], ["bc1sub1", "bc1sub2"])
        self.assertEqual(df.iloc[1]["input_amounts"], [1.25, 0.75])

    def test_parse_file_dispatcher(self) -> None:
        csv_file = self.temp_path / "sample.csv"
        csv_file.write_text(
            "timestamp,src_ip,dst_ip,src_port,dst_port,txid,input_addresses,output_addresses,input_amounts,output_amounts,fee,script_type\n"
            "2026-09-01T00:00:00Z,1.1.1.1,2.2.2.2,8333,8333,deadbeef,addr1,addr2,1.0,0.999,0.001,P2PKH\n"
        )
        df = parse_file(csv_file)
        self.assertEqual(len(df), 1)

        # Unsupported format raises ValueError
        invalid_file = self.temp_path / "sample.parquet"
        invalid_file.write_text("data")
        with self.assertRaises(ValueError):
            parse_file(invalid_file)

        # Nonexistent file raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            parse_file(self.temp_path / "nonexistent.csv")


from src.ingestion.validator import get_clean_subset, validate_transactions


class TestTransactionValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_row = {
            "timestamp": pd.to_datetime("2026-09-01T12:00:00Z", utc=True),
            "src_ip": "192.0.2.1",
            "dst_ip": "198.51.100.1",
            "src_port": 8333,
            "dst_port": 8333,
            "txid": "e" * 64,
            "input_addresses": ["1addr1"],
            "output_addresses": ["bc1addr2"],
            "input_amounts": [1.0],
            "output_amounts": [0.999],
            "fee": 0.001,
            "script_type": "P2WPKH",
        }

    def test_validate_clean_transactions(self) -> None:
        df = pd.DataFrame([self.valid_row])
        validated_df, issues = validate_transactions(df)

        self.assertEqual(len(validated_df), 1)
        self.assertEqual(issues["total_rows"], 1)
        self.assertEqual(len(issues["malformed_txids"]), 0)
        self.assertEqual(len(issues["negative_amounts"]), 0)
        self.assertEqual(len(issues["duplicate_txids"]), 0)
        self.assertEqual(len(issues["mismatched_address_amount_counts"]), 0)
        self.assertEqual(len(issues["invalid_ips"]), 0)
        self.assertEqual(len(issues["null_fields"]), 0)

    def test_validate_detects_all_issue_categories(self) -> None:
        rows = [
            # Row 0: Malformed TXID
            dict(self.valid_row, txid="not-a-valid-hex"),
            # Row 1: Negative amount
            dict(self.valid_row, txid="1" * 64, output_amounts=[-0.5]),
            # Row 2: Address/amount count mismatch
            dict(self.valid_row, txid="2" * 64, input_addresses=["1a", "1b"], input_amounts=[1.0]),
            # Row 3: Invalid IP
            dict(self.valid_row, txid="3" * 64, src_ip="999.999.999.999"),
            # Row 4: Duplicate TXID (part 1)
            dict(self.valid_row, txid="4" * 64),
            # Row 5: Duplicate TXID (part 2)
            dict(self.valid_row, txid="4" * 64),
            # Row 6: Null field in fee
            dict(self.valid_row, txid="6" * 64, fee=np.nan),
            # Row 7: Clean row
            dict(self.valid_row, txid="7" * 64),
        ]
        df = pd.DataFrame(rows)
        validated_df, issues = validate_transactions(df)

        # Unmodified DataFrame returned
        self.assertEqual(len(validated_df), 8)

        # Category checks
        self.assertIn(0, issues["malformed_txids"])
        self.assertIn(1, issues["negative_amounts"])
        self.assertIn(2, issues["mismatched_address_amount_counts"])
        self.assertIn(3, issues["invalid_ips"])
        self.assertIn("4" * 64, issues["duplicate_txids"])
        self.assertIn("fee", issues["null_fields"])
        self.assertEqual(issues["null_fields"]["fee"], 1)

        # Test get_clean_subset
        clean_df = get_clean_subset(df, issues)
        # Only Row 7 is completely clean
        self.assertEqual(len(clean_df), 1)
        self.assertEqual(clean_df.iloc[0]["txid"], "7" * 64)


from unittest.mock import MagicMock, patch
from src.ingestion.geoip_enricher import GeoIPEnricher, enrich_dataframe
import geoip2.errors


class TestGeoIPEnricher(unittest.TestCase):
    def test_missing_db_files_raises_helpful_error(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            GeoIPEnricher(
                country_db_path="nonexistent_country.mmdb",
                asn_db_path="nonexistent_asn.mmdb",
            )
        self.assertIn("GeoLite2", str(ctx.exception))
        self.assertIn("maxmind.com", str(ctx.exception))

    @patch("geoip2.database.Reader")
    def test_lookup_public_ip(self, mock_reader_cls: MagicMock) -> None:
        mock_country_reader = MagicMock()
        mock_asn_reader = MagicMock()
        mock_reader_cls.side_effect = [mock_country_reader, mock_asn_reader]

        # Setup country response
        mock_country_resp = MagicMock()
        mock_country_resp.country.iso_code = "US"
        mock_country_reader.country.return_value = mock_country_resp

        # Setup ASN response
        mock_asn_resp = MagicMock()
        mock_asn_resp.autonomous_system_number = 15169
        mock_asn_resp.autonomous_system_organization = "Google LLC"
        mock_asn_reader.asn.return_value = mock_asn_resp

        # Create dummy existing files for initialization
        with tempfile.NamedTemporaryFile() as c_file, tempfile.NamedTemporaryFile() as a_file:
            with GeoIPEnricher(c_file.name, a_file.name) as enricher:
                result = enricher.lookup("8.8.8.8")
                self.assertEqual(result["country"], "US")
                self.assertEqual(result["asn"], 15169)
                self.assertEqual(result["asn_org"], "Google LLC")

                # Test private IP skips lookup
                priv_result = enricher.lookup("192.168.1.1")
                self.assertIsNone(priv_result["country"])
                self.assertIsNone(priv_result["asn"])

                # Test loopback skips lookup
                loop_result = enricher.lookup("127.0.0.1")
                self.assertIsNone(loop_result["country"])

                # Test invalid IP skips lookup
                inv_result = enricher.lookup("invalid-ip")
                self.assertIsNone(inv_result["country"])

            # Verify close called on readers
            mock_country_reader.close.assert_called_once()
            mock_asn_reader.close.assert_called_once()

    @patch("geoip2.database.Reader")
    def test_lookup_address_not_found(self, mock_reader_cls: MagicMock) -> None:
        mock_country_reader = MagicMock()
        mock_asn_reader = MagicMock()
        mock_reader_cls.side_effect = [mock_country_reader, mock_asn_reader]

        mock_country_reader.country.side_effect = geoip2.errors.AddressNotFoundError("Not found")
        mock_asn_reader.asn.side_effect = geoip2.errors.AddressNotFoundError("Not found")

        with tempfile.NamedTemporaryFile() as c_file, tempfile.NamedTemporaryFile() as a_file:
            enricher = GeoIPEnricher(c_file.name, a_file.name)
            result = enricher.lookup("203.0.113.1")
            self.assertIsNone(result["country"])
            self.assertIsNone(result["asn"])
            enricher.close()

    @patch("geoip2.database.Reader")
    def test_enrich_dataframe(self, mock_reader_cls: MagicMock) -> None:
        mock_country_reader = MagicMock()
        mock_asn_reader = MagicMock()
        mock_reader_cls.side_effect = [mock_country_reader, mock_asn_reader]

        mock_country_resp = MagicMock()
        mock_country_resp.country.iso_code = "DE"
        mock_country_reader.country.return_value = mock_country_resp

        mock_asn_resp = MagicMock()
        mock_asn_resp.autonomous_system_number = 24940
        mock_asn_resp.autonomous_system_organization = "Hetzner Online GmbH"
        mock_asn_reader.asn.return_value = mock_asn_resp

        df = pd.DataFrame({
            "src_ip": ["195.201.0.1", "195.201.0.1", "10.0.0.1"],
            "amount": [1.0, 2.0, 3.0],
        })

        with tempfile.NamedTemporaryFile() as c_file, tempfile.NamedTemporaryFile() as a_file:
            enricher = GeoIPEnricher(c_file.name, a_file.name)
            enriched_df = enrich_dataframe(df, ip_column="src_ip", enricher=enricher)

            self.assertIn("src_ip_country", enriched_df.columns)
            self.assertIn("src_ip_asn", enriched_df.columns)

            self.assertEqual(enriched_df.iloc[0]["src_ip_country"], "DE")
            self.assertEqual(enriched_df.iloc[0]["src_ip_asn"], 24940)
            self.assertEqual(enriched_df.iloc[1]["src_ip_country"], "DE")
            self.assertTrue(pd.isna(enriched_df.iloc[2]["src_ip_country"]))
            self.assertTrue(pd.isna(enriched_df.iloc[2]["src_ip_asn"]))

            enricher.close()


if __name__ == "__main__":
    unittest.main()
