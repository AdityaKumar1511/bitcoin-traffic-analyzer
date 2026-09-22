"""
Unit tests for composite scoring, reason generation, and report export.
"""

from pathlib import Path
import tempfile
import unittest
import pandas as pd

from src.explainability.reason_generator import ReasonGenerator
from src.feedback.feedback_store import FeedbackStore
from dashboard.report_export import generate_case_report


class TestScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_reason_generator(self) -> None:
        reasons = ReasonGenerator.generate_reasons(
            entity_id="1TestWallet",
            entity_type="wallet",
            risk_score=0.88,
            features={"fan_in_ratio": 0.92, "fan_out_ratio": 0.10},
            evasion_flags=["Ransomware_PeelChain"],
            network_data={
                "primary_broadcast_ip": "198.51.100.99",
                "primary_asn": 48031,
                "asn_category": "BULLETPROOF_HOSTING",
                "observation_count": 3,
            },
            taint_data={
                "taint_score": 0.60,
                "taint_hops": 1,
                "taint_source": "1SeedWallet",
                "taint_path": "1SeedWallet -> tx1 -> 1TestWallet",
            },
        )

        self.assertIn("primary_reason", reasons)
        self.assertIn("detailed_justification", reasons)
        self.assertIn("ransomware", reasons["primary_reason"].lower())
        self.assertIn("bulletproof", reasons["detailed_justification"].lower())

    def test_feedback_store(self) -> None:
        db_file = self.temp_path / "test_feedback.db"
        fb_store = FeedbackStore(db_file)

        fb_store.set_feedback("1TestWallet", "CONFIRMED", notes="Verified criminal entity")
        record = fb_store.get_feedback("1TestWallet")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["status"], "CONFIRMED")
        self.assertEqual(record["notes"], "Verified criminal entity")

        # Update to false positive
        fb_store.set_feedback("1TestWallet", "FALSE_POSITIVE", notes="Legitimate exchange")
        record_updated = fb_store.get_feedback("1TestWallet")
        self.assertIsNotNone(record_updated)
        assert record_updated is not None
        self.assertEqual(record_updated["status"], "FALSE_POSITIVE")

    def test_report_export(self) -> None:
        sample_alerts = pd.DataFrame(
            [
                {
                    "composite_risk_score": 0.95,
                    "risk_level": "CRITICAL",
                    "primary_reason": "Ransomware peel chain carrier",
                    "detailed_justification": "Identified in sequential peel chain.",
                    "primary_ip": "1.2.3.4",
                    "primary_asn": 12345,
                    "asn_category": "STANDARD_ISP",
                    "taint_source": "1Seed",
                    "taint_hops": 1,
                    "taint_path": "1Seed -> 1Victim",
                    "analyst_status": "PENDING",
                    "analyst_notes": "",
                    "pattern_types": ["Ransomware_PeelChain"],
                }
            ],
            index=["1Victim"],
        )

        report_file = generate_case_report("1Victim", alerts_df=sample_alerts, output_dir=self.temp_path)
        self.assertTrue(report_file.is_file())
        content = report_file.read_text(encoding="utf-8")
        self.assertIn("1Victim", content)
        self.assertIn("CRITICAL", content)


if __name__ == "__main__":
    unittest.main()
