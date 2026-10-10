"""
Smoke tests for the Streamlit dashboard (app.py).
Ensures all bundled sample datasets render without unhandled exceptions
and that the compliant leader dataset is never marked [DISQUALIFIED].
"""

from __future__ import annotations

import os
import unittest

from advisor_engine import EcoGraniteAdvisor
from esg_parser import parse_document

try:
    from streamlit.testing.v1 import AppTest
    APPTEST_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    APPTEST_AVAILABLE = False


class TestDashboardDataQualitySmoke(unittest.TestCase):
    """Verifies all bundled datasets against the audit pipeline loaded by app.py."""

    def setUp(self) -> None:
        self.advisor = EcoGraniteAdvisor()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def test_compliant_sample_is_never_disqualified(self) -> None:
        data = self.advisor.load_document(os.path.join(self.base_dir, "sample_esg_report.json"))
        analysis = self.advisor.analyze_compliance(data)
        self.assertTrue(analysis["can_audit"])
        self.assertNotIn("DISQUALIFIED", analysis["rating_tier"])
        self.assertEqual(analysis["esg_readiness_score"], 100.0)

    def test_all_bundled_datasets_smoke(self) -> None:
        samples = [
            "sample_esg_report.json",
            "sample_esg_report_india.json",
            "sample_esg_report_siemens.json",
            "sample_esg_report_poor.json",
            "sample_esg_report_invalid.json",
        ]
        for s in samples:
            data = self.advisor.load_document(os.path.join(self.base_dir, s))
            analysis = self.advisor.analyze_compliance(data)
            self.assertIn("esg_readiness_score", analysis)
            report = self.advisor.generate_audit_report(data, analysis)
            self.assertTrue(len(report) > 0)


@unittest.skipUnless(APPTEST_AVAILABLE, "streamlit.testing.v1.AppTest requires Streamlit >= 1.28")
class TestAppStreamlitSmoke(unittest.TestCase):
    """End-to-end Streamlit UI rendering smoke tests (active in CI / Streamlit 1.28+)."""

    def setUp(self) -> None:
        self.app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

    def test_app_renders_default_sample_without_exceptions(self) -> None:
        at = AppTest.from_file(self.app_path).run()
        self.assertFalse(at.exception)
        text = " ".join(str(getattr(w, "value", "")) for w in at.markdown)
        self.assertNotIn("DISQUALIFIED", text)

    def test_app_renders_all_datasets_without_exceptions(self) -> None:
        options = [
            "📗 Sample: EcoGlobal Enterprise (Good/Compliant)",
            "🇮🇳 India: Infosys Limited (SEBI BRSR Core Mandate)",
            "📙 Sample: CarbonHeavy Corp (Lagging/Poor)",
            "📕 Sample: Incoherent Disclosures (Reconciliation Failure)",
        ]
        for opt in options:
            at = AppTest.from_file(self.app_path).run()
            if at.sidebar.radio:
                at.sidebar.radio[0].set_value(opt).run()
            self.assertFalse(at.exception, f"Rendering failed for dataset option: {opt}")


if __name__ == "__main__":
    unittest.main()
