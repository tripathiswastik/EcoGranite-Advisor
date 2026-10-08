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
"""

import io
import json
import os
import unittest
from advisor_engine import EcoGraniteAdvisor
from esg_parser import validate_and_normalize_esg, parse_document, extract_esg_from_text


class TestEcoGraniteAdvisor(unittest.TestCase):
    def setUp(self):
        self.advisor = EcoGraniteAdvisor()

    # =========================================================================
    # 1. Continuous Scoring Mathematics (Both Directions)
    # =========================================================================
    def test_continuous_scoring_higher_is_better(self):
        """Test proportional scoring where higher values are superior."""
        # Exact target: 60/60 -> 100% of 25 pts = 25.0
        self.assertEqual(self.advisor.calculate_continuous_score(60.0, 60.0, 25.0, higher_is_better=True), 25.0)

        # Above target: 80/60 -> capped at 100% = 25.0
        self.assertEqual(self.advisor.calculate_continuous_score(80.0, 60.0, 25.0, higher_is_better=True), 25.0)

        # Halfway: 30/60 -> 50% = 12.5
        self.assertEqual(self.advisor.calculate_continuous_score(30.0, 60.0, 25.0, higher_is_better=True), 12.5)

        # Zero value: 0/60 -> 0 pts
        self.assertEqual(self.advisor.calculate_continuous_score(0.0, 60.0, 25.0, higher_is_better=True), 0.0)

        # None value (missing): 0 pts
        self.assertEqual(self.advisor.calculate_continuous_score(None, 60.0, 25.0, higher_is_better=True), 0.0)

        # Zero target:
        self.assertEqual(self.advisor.calculate_continuous_score(10.0, 0.0, 25.0, higher_is_better=True), 25.0)

    def test_continuous_scoring_lower_is_better(self):
        """Test proportional scoring where lower values are superior (e.g. emissions intensity)."""
        target = 100.0
        max_pts = 20.0

        # Exact target: 100 vs target 100 -> full 20 pts
        self.assertEqual(self.advisor.calculate_continuous_score(100.0, target, max_pts, higher_is_better=False), 20.0)

        # Better than target: 50 vs target 100 -> full 20 pts
        self.assertEqual(self.advisor.calculate_continuous_score(50.0, target, max_pts, higher_is_better=False), 20.0)

        # Zero value (best possible performance): 0 vs target 100 -> full 20 pts
        self.assertEqual(self.advisor.calculate_continuous_score(0.0, target, max_pts, higher_is_better=False), 20.0)

        # Worse than target: 200 vs target 100 -> target/value = 100/200 = 50% = 10 pts
        self.assertEqual(self.advisor.calculate_continuous_score(200.0, target, max_pts, higher_is_better=False), 10.0)

        # Extremely poor: 1000 vs target 100 -> 100/1000 = 10% = 2.0 pts
        self.assertEqual(self.advisor.calculate_continuous_score(1000.0, target, max_pts, higher_is_better=False), 2.0)

    # =========================================================================
    # 2. Data Integrity: GHG Scope Reconciliation Mismatch
    # =========================================================================
    def test_ghg_reconciliation_failure_detection(self):
        """Detect when reported total contradicts Scope 1 + Scope 2 + Scope 3 sum."""
        data = {
            "company_name": "Contradictory Corp",
            "emissions_metric_tons_co2e": {
                "scope_1_direct": 100.0,
                "scope_2_indirect_market": 100.0,
                "scope_3_value_chain": 100.0,
                "total_ghg": 5000.0  # Contradicts sum of 300!
            }
        }
        normalized = validate_and_normalize_esg(data)
        dq = normalized["data_quality"]

        self.assertEqual(dq["reconciliation_status"], "FAILED")
        self.assertEqual(dq["calculated_total_ghg"], 300.0)
        self.assertEqual(dq["reported_total_ghg"], 5000.0)
        self.assertEqual(dq["total_ghg_variance"], 4700.0)
        self.assertTrue(any("does not reconcile" in w for w in dq["validation_warnings"]))

        # Analysis must disqualify company from [A] rating
        analysis = self.advisor.analyze_compliance(normalized)
        self.assertIn("[DISQUALIFIED]", analysis["rating_tier"])
        self.assertEqual(analysis["benchmarks"]["GHG Scope Reconciliation (S1+S2+S3 = Total)"]["status"], "Failed Reconciliation")

    # =========================================================================
    # 3. Preserving Anomalies (No Silent Clamping)
    # =========================================================================
    def test_unclamped_anomalies_recorded_as_warnings(self):
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
                "renewable_mwh": 2500.0,  # Exceeds total!
                "renewable_share_pct": 250.0  # > 100%
            },
            "social_and_governance": {
                "female_board_representation_pct": 135.0  # > 100%
            }
        }
        normalized = validate_and_normalize_esg(data)
        warnings = normalized["data_quality"]["validation_warnings"]

        # Warnings must capture both violations
        self.assertTrue(any("exceeds total" in w for w in warnings))
        self.assertTrue(any("135.0%" in w or "outside" in w for w in warnings))

    # =========================================================================
    # 4. Three-State Evaluation & Missing Data Handling
    # =========================================================================
    def test_three_state_missing_data(self):
        """Ensure missing metrics are labeled Unknown/Unrated, not penalized as zero performance."""
        data = {
            "company_name": "Sparse Reporting LLC",
            "emissions_metric_tons_co2e": {
                "scope_1_direct": 1000.0,
                "scope_2_indirect_market": 500.0
                # Scope 3 and YoY reduction omitted
            },
            "renewable_energy": {}  # Renewable share omitted
        }
        normalized = validate_and_normalize_esg(data)
        analysis = self.advisor.analyze_compliance(normalized)

        # Renewable energy benchmark status must be Unknown, not just Failed
        ren_bench = analysis["benchmarks"]["Renewable Energy Share (Target >= 60%)"]
        self.assertEqual(ren_bench["status"], "Unknown (Missing Data)")
        self.assertEqual(ren_bench["risk"], "Unrated")
        self.assertEqual(ren_bench["score"], "0.0/25.0")

        # Completeness should reflect missing fields
        self.assertLess(normalized["data_quality"]["completeness_pct"], 100.0)

    # =========================================================================
    # 5. Refusal on Extraction Failure (Never Invent Sample Values)
    # =========================================================================
    def test_refusal_when_extraction_fails(self):
        """When an unstructured or empty document has no ESG metrics, refuse to audit."""
        empty_text = "This is a random corporate memo with no greenhouse gas data."
        extracted = extract_esg_from_text(empty_text, "empty_memo.pdf")

        self.assertEqual(extracted["status"], "extraction_failed")
        self.assertGreater(len(extracted["errors"]), 0)

        # Audit engine must refuse to produce a compliance score
        analysis = self.advisor.analyze_compliance(extracted)
        self.assertFalse(analysis["can_audit"])
        self.assertIn("EXTRACTION FAILED", analysis["rating_tier"])

        # Audit report output must report refusal
        report = self.advisor.generate_audit_report(extracted, analysis)
        self.assertIn("AUDIT REFUSAL (EXTRACTION FAILED)", report)

    # =========================================================================
    # 6. Dynamic Scope 3 Percentage Derivation
    # =========================================================================
    def test_dynamic_scope3_percentage(self):
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

        # Dynamic recommendation test
        report = self.advisor.generate_audit_report(data, analysis)
        self.assertIn("80.0%", report)
        self.assertIn("35.0%", report)

    # =========================================================================
    # 7. Decoupled AI Insights & Transparent Runtime Metadata
    # =========================================================================
    def test_decoupled_ai_insights_and_runtime_metadata(self):
        """Verify get_runtime_metadata provides transparent diagnostics."""
        meta = self.advisor.granite_client.get_runtime_metadata()
        self.assertIn("engine", meta)
        self.assertIn("label", meta)
        self.assertTrue(meta["is_fallback"] if not meta.get("is_live") else True)

    # =========================================================================
    # 8. Testing End-to-End Sample Files
    # =========================================================================
    def test_sample_files_evaluation(self):
        """Test good, poor, and invalid datasets."""
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Good dataset
        good_path = os.path.join(base_dir, "sample_esg_report.json")
        good_data = self.advisor.load_document(good_path)
        good_analysis = self.advisor.analyze_compliance(good_data)
        self.assertEqual(good_analysis["data_quality"]["reconciliation_status"], "PASSED")
        self.assertGreaterEqual(good_analysis["esg_readiness_score"], 85.0)

        # 2. Poor dataset
        poor_path = os.path.join(base_dir, "sample_esg_report_poor.json")
        poor_data = self.advisor.load_document(poor_path)
        poor_analysis = self.advisor.analyze_compliance(poor_data)
        self.assertLess(poor_analysis["esg_readiness_score"], 60.0)

        # 3. Invalid dataset
        invalid_path = os.path.join(base_dir, "sample_esg_report_invalid.json")
        invalid_data = self.advisor.load_document(invalid_path)
        invalid_analysis = self.advisor.analyze_compliance(invalid_data)
        self.assertEqual(invalid_analysis["data_quality"]["reconciliation_status"], "FAILED")
        self.assertIn("[DISQUALIFIED]", invalid_analysis["rating_tier"])


if __name__ == "__main__":
    unittest.main()
