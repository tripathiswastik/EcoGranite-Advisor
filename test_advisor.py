"""
Comprehensive Test Suite for EcoGranite-Advisor Engine
Covers:
- Continuous scoring in both directions (higher/lower is better, edge cases, target=0)
- Data integrity & GHG Protocol reconciliation (S1+S2+S3 != Reported Total)
- Preserving anomalies (no silent clamping, collecting validation warnings)
- Three-state missing data handling (PASS, FAIL, UNKNOWN)
- Refusal on extraction failure (never inventing fake numbers)
- Dynamic Scope 3 derivation
- Decoupled AI insight generation and WatsonX error diagnostics
- End-to-end evaluation on Good, Poor, and Invalid datasets
- Code review regressions & security verifications (TestReviewRegressions)
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

from advisor_engine import EcoGraniteAdvisor
from esg_parser import (
    MAX_UPLOAD_BYTES,
    extract_esg_from_text,
    parse_document,
    validate_and_normalize_esg,
)
from granite_client import GraniteReasoningClient


class TestEcoGraniteAdvisor(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for name in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID", "HUGGINGFACE_API_KEY"):
            os.environ.pop(name, None)
        self.advisor = EcoGraniteAdvisor()

    # =========================================================================
    # 1. Continuous Scoring Mathematics (Both Directions)
    # =========================================================================
    def test_continuous_scoring_higher_is_better(self) -> None:
        """Test proportional scoring where higher values are superior."""
        self.assertEqual(self.advisor.calculate_continuous_score(60.0, 60.0, 25.0, higher_is_better=True), 25.0)
        self.assertEqual(self.advisor.calculate_continuous_score(80.0, 60.0, 25.0, higher_is_better=True), 25.0)
        self.assertEqual(self.advisor.calculate_continuous_score(30.0, 60.0, 25.0, higher_is_better=True), 12.5)
        self.assertEqual(self.advisor.calculate_continuous_score(0.0, 60.0, 25.0, higher_is_better=True), 0.0)
        self.assertEqual(self.advisor.calculate_continuous_score(None, 60.0, 25.0, higher_is_better=True), 0.0)
        self.assertEqual(self.advisor.calculate_continuous_score(10.0, 0.0, 25.0, higher_is_better=True), 25.0)

    def test_continuous_scoring_lower_is_better(self) -> None:
        """Test proportional scoring where lower values are superior (e.g. emissions intensity)."""
        target = 100.0
        max_pts = 20.0
        self.assertEqual(self.advisor.calculate_continuous_score(100.0, target, max_pts, higher_is_better=False), 20.0)
        self.assertEqual(self.advisor.calculate_continuous_score(50.0, target, max_pts, higher_is_better=False), 20.0)
        self.assertEqual(self.advisor.calculate_continuous_score(0.0, target, max_pts, higher_is_better=False), 20.0)
        self.assertEqual(self.advisor.calculate_continuous_score(200.0, target, max_pts, higher_is_better=False), 10.0)
        self.assertEqual(self.advisor.calculate_continuous_score(1000.0, target, max_pts, higher_is_better=False), 2.0)

    # =========================================================================
    # 2. Data Integrity: GHG Scope Reconciliation Mismatch
    # =========================================================================
    def test_ghg_reconciliation_failure_detection(self) -> None:
        """Detect when reported total contradicts Scope 1 + Scope 2 + Scope 3 sum."""
        data = {
            "company_name": "Contradictory Corp",
            "emissions_metric_tons_co2e": {
                "scope_1_direct": 100.0,
                "scope_2_indirect_market": 100.0,
                "scope_3_value_chain": 100.0,
                "total_ghg": 5000.0
            }
        }
        normalized = validate_and_normalize_esg(data)
        dq = normalized["data_quality"]

        self.assertEqual(dq["reconciliation_status"], "FAILED")
        self.assertEqual(dq["calculated_total_ghg"], 300.0)
        self.assertEqual(dq["reported_total_ghg"], 5000.0)
        self.assertEqual(dq["total_ghg_variance"], 4700.0)
        self.assertTrue(any("does not reconcile" in w for w in dq["validation_warnings"]))

        analysis = self.advisor.analyze_compliance(normalized)
        self.assertIn("[DISQUALIFIED]", analysis["rating_tier"])
        self.assertEqual(analysis["benchmarks"]["GHG Scope Reconciliation (S1+S2+S3 = Total)"]["status"], "Failed Reconciliation")

    # =========================================================================
    # 3. Preserving Anomalies (No Silent Clamping)
    # =========================================================================
    def test_unclamped_anomalies_recorded_as_warnings(self) -> None:
        """Ensure values like 135% diversity or renewable MWh > total MWh are flagged."""
        data = {
            "company_name": "Anomalous Disclosures",
            "emissions_metric_tons_co2e": {
                "scope_1_direct": 500.0,
                "scope_2_indirect_market": 500.0,
                "scope_3_value_chain": 1000.0,
                "total_ghg": 2000.0
            },
            "renewable_energy": {
                "total_mwh_consumed": 1000.0,
                "renewable_mwh": 2500.0,
                "renewable_share_pct": 250.0
            },
            "social_and_governance": {
                "female_board_representation_pct": 135.0
            }
        }
        normalized = validate_and_normalize_esg(data)
        warnings = normalized["data_quality"]["validation_warnings"]

        self.assertTrue(any("exceeds total" in w for w in warnings))
        self.assertTrue(any("135.0%" in w or "outside" in w for w in warnings))

    # =========================================================================
    # 4. Three-State Evaluation & Missing Data Handling
    # =========================================================================
    def test_three_state_missing_data(self) -> None:
        """Ensure missing metrics are labeled Unknown/Unrated, not penalized as zero performance."""
        data = {
            "company_name": "Sparse Reporting LLC",
            "emissions_metric_tons_co2e": {
                "scope_1_direct": 1000.0,
                "scope_2_indirect_market": 500.0
            },
            "renewable_energy": {}
        }
        normalized = validate_and_normalize_esg(data)
        analysis = self.advisor.analyze_compliance(normalized)

        ren_bench = analysis["benchmarks"]["Renewable Energy Share (Target >= 60%)"]
        self.assertEqual(ren_bench["status"], "Unknown (Missing Data)")
        self.assertEqual(ren_bench["risk"], "Unrated")
        self.assertEqual(ren_bench["score"], "0.0/25.0")
        self.assertLess(normalized["data_quality"]["completeness_pct"], 100.0)

    # =========================================================================
    # 5. Refusal on Extraction Failure (Never Invent Sample Values)
    # =========================================================================
    def test_refusal_when_extraction_fails(self) -> None:
        """When an unstructured or empty document has no ESG metrics, refuse to audit."""
        empty_text = "This is a random corporate memo with no greenhouse gas data."
        extracted = extract_esg_from_text(empty_text, "empty_memo.pdf")

        self.assertEqual(extracted["status"], "extraction_failed")
        self.assertGreater(len(extracted["errors"]), 0)

        analysis = self.advisor.analyze_compliance(extracted)
        self.assertFalse(analysis["can_audit"])
        self.assertIn("EXTRACTION FAILED", analysis["rating_tier"])

        report = self.advisor.generate_audit_report(extracted, analysis)
        self.assertIn("AUDIT REFUSAL (EXTRACTION FAILED)", report)

    def test_pdf_binary_stream_refusal(self) -> None:
        """Binary PDF bytes with no decompressed text must cleanly fail without naive UTF-8 regex fallback."""
        fake_binary_pdf = io.BytesIO(b"%PDF-1.4\x00\x01\x02fake_binary_stream\xff\xfe")
        fake_binary_pdf.name = "unreadable_report.pdf"
        extracted = self.advisor.load_document(fake_binary_pdf)
        self.assertEqual(extracted["status"], "extraction_failed")
        self.assertTrue(any("PDF extraction failed" in err for err in extracted["errors"]))

    # =========================================================================
    # 6. Dynamic Scope 3 Percentage Derivation
    # =========================================================================
    def test_dynamic_scope3_percentage(self) -> None:
        """Scope 3 percentage must be mathematically derived from total emissions."""
        data = {
            "company_name": "Test Energy",
            "emissions_metric_tons_co2e": {
                "scope_1_direct": 1000.0,
                "scope_2_indirect_market": 1000.0,
                "scope_3_value_chain": 8000.0,
                "total_ghg": 10000.0,
                "achieved_reduction_yoy_pct": 6.0
            },
            "renewable_energy": {"renewable_share_pct": 80.0},
            "water_and_waste": {"waste_diverted_from_landfill_pct": 80.0, "water_recycled_pct": 35.0},
            "social_and_governance": {"female_board_representation_pct": 50.0}
        }
        analysis = self.advisor.analyze_compliance(data)
        self.assertAlmostEqual(analysis["metrics"]["scope_3_pct"], 80.0, places=1)
        self.assertTrue(analysis["reduction_pass"])

        report = self.advisor.generate_audit_report(data, analysis)
        self.assertIn("80.0%", report)
        self.assertIn("35.0%", report)

    # =========================================================================
    # 7. Decoupled AI Insights & Transparent Runtime Metadata
    # =========================================================================
    def test_decoupled_ai_insights_and_runtime_metadata(self) -> None:
        """Verify get_runtime_metadata provides transparent diagnostics."""
        meta = self.advisor.granite_client.get_runtime_metadata()
        self.assertIn("engine", meta)
        self.assertIn("label", meta)
        self.assertTrue(meta["is_fallback"])
        self.assertIn("fallback_reason", meta)

    # =========================================================================
    # 8. Testing End-to-End Sample Files
    # =========================================================================
    def test_sample_files_evaluation(self) -> None:
        """Test good, poor, and invalid datasets."""
        base_dir = os.path.dirname(os.path.abspath(__file__))

        good_path = os.path.join(base_dir, "sample_esg_report.json")
        good_data = self.advisor.load_document(good_path)
        good_analysis = self.advisor.analyze_compliance(good_data)
        self.assertEqual(good_analysis["data_quality"]["reconciliation_status"], "PASSED")
        self.assertEqual(good_analysis["esg_readiness_score"], 100.0)

        poor_path = os.path.join(base_dir, "sample_esg_report_poor.json")
        poor_data = self.advisor.load_document(poor_path)
        poor_analysis = self.advisor.analyze_compliance(poor_data)
        self.assertEqual(poor_analysis["esg_readiness_score"], 47.8)

        invalid_path = os.path.join(base_dir, "sample_esg_report_invalid.json")
        invalid_data = self.advisor.load_document(invalid_path)
        invalid_analysis = self.advisor.analyze_compliance(invalid_data)
        self.assertEqual(invalid_analysis["data_quality"]["reconciliation_status"], "FAILED")
        self.assertIn("[DISQUALIFIED]", invalid_analysis["rating_tier"])
        self.assertEqual(invalid_analysis["esg_readiness_score"], 47.4)

        india_path = os.path.join(base_dir, "sample_esg_report_india.json")
        india_data = self.advisor.load_document(india_path)
        india_analysis = self.advisor.analyze_compliance(india_data)
        self.assertEqual(india_analysis["data_quality"]["reconciliation_status"], "PASSED")
        self.assertIn("Infosys Limited", india_analysis["company_name"])
        self.assertIn("SEBI BRSR", india_analysis["standards"])
        self.assertEqual(india_analysis["esg_readiness_score"], 99.5)

    # =========================================================================
    # 9. v3.0 SASB Sector Weighting & CSRD Double Materiality
    # =========================================================================
    def test_sasb_sector_weighting_and_double_materiality(self) -> None:
        """Test SASB SICS sector weighting profiles and CSRD double materiality."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        good_path = os.path.join(base_dir, "sample_esg_report.json")
        good_data = self.advisor.load_document(good_path)
        analysis = self.advisor.analyze_compliance(good_data)

        tech_weighted = self.advisor.calculate_sector_weighted_score(analysis, sector="Technology & Software")
        self.assertIn("Scope 3 Value Chain (45%)", tech_weighted["weights"])
        self.assertGreater(tech_weighted["weighted_score"], 80.0)

        heavy_weighted = self.advisor.calculate_sector_weighted_score(analysis, sector="Heavy Industry & Metals")
        self.assertIn("Scope 1 Direct Operations (35%)", heavy_weighted["weights"])
        self.assertGreater(heavy_weighted["weighted_score"], 80.0)

        dm = self.advisor.evaluate_double_materiality(analysis)
        self.assertIn("financial_risk_score", dm)
        self.assertIn("impact_materiality_score", dm)
        self.assertIn("quadrant", dm)

        scorecard = self.advisor.generate_multi_framework_scorecard()
        self.assertEqual(len(scorecard), 3)
        self.assertTrue(any("Infosys" in c["company_name"] for c in scorecard))
        self.assertTrue(any("Siemens" in c["company_name"] for c in scorecard))

        gri_eval = self.advisor.calculate_framework_score(analysis, framework="GRI Baseline")
        self.assertEqual(gri_eval["framework"], "GRI Baseline")
        self.assertEqual(gri_eval["score"], 100.0)

        csrd_eval = self.advisor.calculate_framework_score(analysis, framework="CSRD (ESRS)")
        self.assertIn("CSRD", csrd_eval["framework"])
        self.assertLessEqual(csrd_eval["score"], 100.0)

        issb_eval = self.advisor.calculate_framework_score(analysis, framework="ISSB (IFRS S2)")
        self.assertIn("ISSB", issb_eval["framework"])
        self.assertGreaterEqual(issb_eval["score"], 80.0)


GOOD_EMISSIONS = {
    "scope_1_direct": 100.0, "scope_2_indirect_market": 100.0,
    "scope_3_value_chain": 100.0, "total_ghg": 300.0,
}


class TestReviewRegressions(unittest.TestCase):
    """Regression tests for defects found in the code review."""

    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for name in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID", "HUGGINGFACE_API_KEY"):
            os.environ.pop(name, None)
        self.advisor = EcoGraniteAdvisor()

    def test_nan_and_inf_are_rejected_not_reconciled(self) -> None:
        data = {"emissions_metric_tons_co2e": dict(GOOD_EMISSIONS, scope_1_direct=float("nan"), total_ghg=float("inf"))}
        out = validate_and_normalize_esg(data)
        self.assertIsNone(out["emissions_metric_tons_co2e"]["scope_1_direct"])
        self.assertTrue(any("not a finite number" in w for w in out["data_quality"]["validation_warnings"]))
        self.assertNotEqual(out["data_quality"]["reconciliation_status"], "PASSED")

    def test_empty_payload_is_refused(self) -> None:
        out = validate_and_normalize_esg({})
        self.assertEqual(out["status"], "extraction_failed")
        self.assertFalse(self.advisor.analyze_compliance(out)["can_audit"])

    def test_missing_total_is_unverified_and_cannot_be_leader(self) -> None:
        data = {"emissions_metric_tons_co2e": {"scope_1_direct": 1.0, "scope_2_indirect_market": 1.0}}
        out = validate_and_normalize_esg(data)
        self.assertEqual(out["data_quality"]["reconciliation_status"], "UNVERIFIED")
        analysis = self.advisor.analyze_compliance(out)
        self.assertNotIn("[A]", analysis["rating_tier"])

    def test_missing_governance_metrics_get_no_credit(self) -> None:
        out = validate_and_normalize_esg({"emissions_metric_tons_co2e": GOOD_EMISSIONS})
        bench = self.advisor.analyze_compliance(out)["benchmarks"]
        self.assertEqual(bench["Gender Pay Equity (Target >= 0.98)"]["status"], "Unknown (Missing Data)")
        self.assertEqual(bench["Gender Pay Equity (Target >= 0.98)"]["score"], "0.0/7.5")

    def test_report_does_not_crash_when_yoy_missing(self) -> None:
        out = validate_and_normalize_esg({"company_name": "X", "emissions_metric_tons_co2e": GOOD_EMISSIONS})
        report = self.advisor.generate_audit_report(out, self.advisor.analyze_compliance(out))
        self.assertIn("N/A", report)

    def test_raw_dict_is_normalized_before_scoring(self) -> None:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_esg_report.json"), encoding="utf-8") as f:
            raw = json.load(f)
        analysis = self.advisor.analyze_compliance(raw)
        self.assertEqual(analysis["data_quality"]["reconciliation_status"], "PASSED")
        self.assertNotIn("DISQUALIFIED", analysis["rating_tier"])

    def test_malformed_numbers_in_text_do_not_crash(self) -> None:
        text = "Scope 1: 12,3.4.5\nScope 2: 5\nRenewable: 10%"
        out = extract_esg_from_text(text, "x.pdf")
        self.assertIn(out["status"], ("success", "extraction_failed"))

    def test_out_of_range_percentage_earns_no_points(self) -> None:
        data = {"emissions_metric_tons_co2e": GOOD_EMISSIONS, "renewable_energy": {"renewable_share_pct": 250.0}}
        bench = self.advisor.analyze_compliance(validate_and_normalize_esg(data))["benchmarks"]
        row = bench["Renewable Energy Share (Target >= 60%)"]
        self.assertEqual(row["status"], "Invalid (Out of Range)")
        self.assertEqual(row["score"], "0.0/25.0")

    def test_malformed_json_path_returns_failure(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write("{not json")
        self.addCleanup(os.unlink, tmp.name)
        self.assertEqual(parse_document(tmp.name)["status"], "extraction_failed")

    def test_docx_with_entity_declaration_is_refused(self) -> None:
        xml = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("word/document.xml", xml)
        buf.seek(0)
        buf.name = "evil.docx"
        self.assertEqual(parse_document(buf)["status"], "extraction_failed")

    def test_scorecard_is_computed_not_hardcoded(self) -> None:
        rows = self.advisor.generate_multi_framework_scorecard()
        eco = next(r for r in rows if r["company_name"].startswith("EcoGlobal"))
        self.assertIn("95.0", eco["csrd_esrs_score"])

    def test_unsupported_sector_raises(self) -> None:
        analysis = self.advisor.analyze_compliance(validate_and_normalize_esg({"emissions_metric_tons_co2e": GOOD_EMISSIONS}))
        with self.assertRaises(ValueError):
            self.advisor.calculate_sector_weighted_score(analysis, sector="Mining")

    def test_huggingface_key_alone_does_not_enable_live_mode(self) -> None:
        with mock.patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}):
            client = GraniteReasoningClient()
            self.assertFalse(client.is_live)
            self.assertTrue(client.is_hf_live)
            meta = client.get_runtime_metadata()
            self.assertEqual(meta["engine"], "huggingface")
            self.assertIn("Hugging Face Inference API", meta["label"])

    def test_oversized_upload_is_refused(self) -> None:
        fake_huge_stream = io.BytesIO(b"0" * (MAX_UPLOAD_BYTES + 1024))
        fake_huge_stream.name = "huge_file.json"
        result = parse_document(fake_huge_stream)
        self.assertEqual(result["status"], "extraction_failed")
        self.assertTrue(any("exceeds" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main()
