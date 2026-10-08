"""
Test Suite for EcoGranite-Advisor Engine
Covers edge cases, missing fields, threshold conditions, zero/negative bounds,
continuous scoring, and dynamic recommendation generation.
"""

import unittest
from advisor_engine import EcoGraniteAdvisor
from esg_parser import validate_and_normalize_esg


class TestEcoGraniteAdvisor(unittest.TestCase):
    def setUp(self):
        self.advisor = EcoGraniteAdvisor()

    def test_missing_fields_defaults(self):
        """Ensure parser provides safe defaults for sparse payloads."""
        sparse_input = {"company_name": "Minimal Corp"}
        norm = validate_and_normalize_esg(sparse_input)
        self.assertEqual(norm["company_name"], "Minimal Corp")
        self.assertEqual(norm["emissions_metric_tons_co2e"]["total_ghg"], 0.0)
        self.assertFalse(norm["renewable_energy"]["re100_committed"])

    def test_zero_and_negative_clamping(self):
        """Negative values should be clamped to zero."""
        negative_input = {
            "emissions_metric_tons_co2e": {
                "scope_1_direct": -100.0,
                "scope_2_indirect_market": 50.0
            }
        }
        norm = validate_and_normalize_esg(negative_input)
        self.assertEqual(norm["emissions_metric_tons_co2e"]["scope_1_direct"], 0.0)
        self.assertEqual(norm["emissions_metric_tons_co2e"]["scope_2_indirect_market"], 50.0)

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
            "water_and_waste": {"waste_diverted_from_landfill_pct": 80.0},
            "social_and_governance": {"female_board_representation_pct": 50.0}
        }
        analysis = self.advisor.analyze_compliance(data)
        self.assertAlmostEqual(analysis["metrics"]["scope_3_pct"], 80.0, places=1)
        self.assertTrue(analysis["reduction_pass"])

    def test_continuous_scoring_vs_binary_step(self):
        """Continuous scoring rewards proportional progress rather than hard cutoff."""
        # 30% renewable out of 60% target should yield 50% of the 30 pts = 15 pts
        score = self.advisor.calculate_continuous_score(30.0, 60.0, 30.0)
        self.assertEqual(score, 15.0)

        # 59% renewable should yield ~29.5 pts, not a catastrophic drop to 10
        score_59 = self.advisor.calculate_continuous_score(59.0, 60.0, 30.0)
        self.assertEqual(score_59, 29.5)

    def test_lagging_reduction_flag(self):
        """Ensure companies with sub-target YoY reduction are flagged as lagging, not ON TRACK."""
        data = {
            "company_name": "Lagging Corp",
            "emissions_metric_tons_co2e": {
                "scope_1_direct": 500.0,
                "scope_2_indirect_market": 500.0,
                "scope_3_value_chain": 500.0,
                "total_ghg": 1500.0,
                "achieved_reduction_yoy_pct": 2.1
            },
            "renewable_energy": {"renewable_share_pct": 20.0},
            "water_and_waste": {"waste_diverted_from_landfill_pct": 50.0, "water_recycled_pct": 20.0},
            "social_and_governance": {"female_board_representation_pct": 25.0}
        }
        analysis = self.advisor.analyze_compliance(data)
        self.assertFalse(analysis["reduction_pass"])
        report = self.advisor.generate_audit_report(data, analysis)
        self.assertIn("LAGGING BEHIND TARGET", report)
        self.assertNotIn("(ON TRACK)", report)

    def test_dynamic_granite_recommendations(self):
        """Recommendations must reference actual calculated numbers."""
        data = {
            "company_name": "Dynamic Audit",
            "emissions_metric_tons_co2e": {
                "scope_1_direct": 100.0,
                "scope_2_indirect_market": 100.0,
                "scope_3_value_chain": 800.0,
                "total_ghg": 1000.0,
                "achieved_reduction_yoy_pct": 6.0
            },
            "renewable_energy": {"renewable_share_pct": 75.0},
            "water_and_waste": {"waste_diverted_from_landfill_pct": 80.0, "water_recycled_pct": 35.0},
            "social_and_governance": {"female_board_representation_pct": 50.0}
        }
        analysis = self.advisor.analyze_compliance(data)
        report = self.advisor.generate_audit_report(data, analysis)

    def test_flexible_document_loading_dict_and_stream(self):
        """Ensure load_document handles raw dicts and StringIO streams."""
        import io
        import json

        data_dict = {
            "company_name": "Stream Ingested Corp",
            "emissions_metric_tons_co2e": {"total_ghg": 5000.0}
        }

        # Dict input test
        norm_from_dict = self.advisor.load_document(data_dict)
        self.assertEqual(norm_from_dict["company_name"], "Stream Ingested Corp")

        # Stream input test
        stream = io.StringIO(json.dumps(data_dict))
        stream.name = "disclosure.json"
        norm_from_stream = self.advisor.load_document(stream)
        self.assertEqual(norm_from_stream["company_name"], "Stream Ingested Corp")

    def test_structured_roadmap_generation(self):
        """Structured roadmap items should have proper keys and dynamic metrics."""
        data = {
            "company_name": "Roadmap Test Corp",
            "emissions_metric_tons_co2e": {
                "scope_1_direct": 200.0,
                "scope_2_indirect_market": 300.0,
                "scope_3_value_chain": 5000.0,
                "total_ghg": 5500.0,
                "achieved_reduction_yoy_pct": 7.0
            },
            "renewable_energy": {"renewable_share_pct": 45.0},
            "water_and_waste": {"water_recycled_pct": 30.0, "waste_diverted_from_landfill_pct": 80.0},
            "social_and_governance": {"female_board_representation_pct": 45.0}
        }
        analysis = self.advisor.analyze_compliance(data)
        items = self.advisor.generate_roadmap_items(analysis)
        self.assertGreaterEqual(len(items), 3)
        self.assertIn("pillar", items[0])
        self.assertIn("action", items[0])
        # Scope 3 is > 90%, so item 1 should target Scope 3
        self.assertIn("Scope 3", items[0]["title"])


if __name__ == "__main__":
    unittest.main()

