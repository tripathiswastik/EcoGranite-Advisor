"""
EcoGranite-Advisor Engine
Comprehensive ESG Sustainability Auditor evaluating Scope 1-3 GHG emissions,
dynamic continuous scoring, and risk-weighted compliance analysis.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

from esg_parser import parse_document
from granite_client import GraniteReasoningClient


class EcoGraniteAdvisor:
    def __init__(self, model_id: str = "ibm-granite/granite-3.0-8b-instruct"):
        self.model_id = model_id
        self.granite_client = GraniteReasoningClient(model_id=model_id)

    def load_document(self, file_source: Any) -> Dict[str, Any]:
        """Loads and normalizes an ESG report via the parser pipeline."""
        return parse_document(file_source)

    def calculate_continuous_score(self, value: float, target: float, max_weight: float, higher_is_better: bool = True) -> float:
        """
        Computes a proportional, continuous score rather than an abrupt binary step.
        """
        if higher_is_better:
            ratio = value / target if target > 0 else 1.0
        else:
            ratio = target / value if value > 0 else 1.0

        ratio = max(0.0, min(1.0, ratio))
        return round(ratio * max_weight, 2)

    def analyze_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs in-depth risk analysis, dynamic Scope 3 calculations,
        and continuous weighted ESG readiness scoring.
        """
        emissions = data.get("emissions_metric_tons_co2e", {})
        energy = data.get("renewable_energy", {})
        waste = data.get("water_and_waste", {})
        gov = data.get("social_and_governance", {})

        s1 = emissions.get("scope_1_direct", 0.0)
        s2 = emissions.get("scope_2_indirect_market", 0.0)
        s3 = emissions.get("scope_3_value_chain", 0.0)
        total_ghg = emissions.get("total_ghg", (s1 + s2 + s3))
        s3_pct = (s3 / total_ghg * 100.0) if total_ghg > 0 else 0.0

        # Benchmark 1: Renewable Energy (Target: >= 60%, Weight: 30 pts)
        ren_pct = energy.get("renewable_share_pct", 0.0)
        ren_score = self.calculate_continuous_score(ren_pct, 60.0, 30.0)
        ren_diff = ren_pct - 60.0
        ren_status = "Compliant" if ren_diff >= 0 else "Needs Improvement"
        ren_risk = "Low" if ren_diff >= 0 else ("Moderate" if ren_diff >= -15.0 else "High")

        # Benchmark 2: YoY Emissions Reduction (Target: >= 5%, Weight: 30 pts)
        yoy_red = emissions.get("achieved_reduction_yoy_pct", 0.0)
        reduction_pass = yoy_red >= 5.0
        red_score = self.calculate_continuous_score(yoy_red, 5.0, 30.0)
        red_diff = yoy_red - 5.0
        red_status = "Compliant (On Track)" if reduction_pass else "Below Target (Lagging)"
        red_risk = "Low" if reduction_pass else ("Moderate" if yoy_red > 0 else "Severe")

        # Benchmark 3: Waste Landfill Diversion (Target: >= 75%, Weight: 20 pts)
        waste_div = waste.get("waste_diverted_from_landfill_pct", 0.0)
        waste_score = self.calculate_continuous_score(waste_div, 75.0, 20.0)
        waste_diff = waste_div - 75.0
        waste_status = "Compliant" if waste_diff >= 0 else "Needs Improvement"
        waste_risk = "Low" if waste_diff >= 0 else "Moderate"

        # Benchmark 4: Board Gender Diversity (Target: >= 40%, Weight: 20 pts)
        board_div = gov.get("female_board_representation_pct", 0.0)
        div_score = self.calculate_continuous_score(board_div, 40.0, 20.0)
        div_diff = board_div - 40.0
        div_status = "Compliant" if div_diff >= 0 else "Needs Improvement"
        div_risk = "Low" if div_diff >= 0 else "Moderate"

        total_score = round(ren_score + red_score + waste_score + div_score, 1)

        # Rating tier
        if total_score >= 85.0:
            rating = "[A] EXCELLENT (ESG LEADER)"
        elif total_score >= 70.0:
            rating = "[B] GOOD (PROGRESSING)"
        elif total_score >= 50.0:
            rating = "[C] MODERATE (AT RISK)"
        else:
            rating = "[D] CRITICAL (NON-COMPLIANT)"

        benchmarks = {
            "Renewable Energy Share (Target >= 60%)": {
                "value": f"{ren_pct:.1f}%",
                "variance": f"{ren_diff:+.1f}%",
                "score": f"{ren_score}/30",
                "status": ren_status,
                "risk": ren_risk
            },
            "YoY Emissions Reduction (Target >= 5%)": {
                "value": f"{yoy_red:.1f}%",
                "variance": f"{red_diff:+.1f}%",
                "score": f"{red_score}/30",
                "status": red_status,
                "risk": red_risk
            },
            "Waste Diversion (Target >= 75%)": {
                "value": f"{waste_div:.1f}%",
                "variance": f"{waste_diff:+.1f}%",
                "score": f"{waste_score}/20",
                "status": waste_status,
                "risk": waste_risk
            },
            "Board Diversity (Target >= 40%)": {
                "value": f"{board_div:.1f}%",
                "variance": f"{div_diff:+.1f}%",
                "score": f"{div_score}/20",
                "status": div_status,
                "risk": div_risk
            }
        }

        return {
            "company_name": data.get("company_name", "Unknown Entity"),
            "reporting_year": data.get("reporting_year", "N/A"),
            "standards": data.get("standards", ["GRI", "TCFD", "SASB"]),
            "esg_readiness_score": total_score,
            "rating_tier": rating,
            "reduction_pass": reduction_pass,
            "metrics": {
                "scope_1": s1,
                "scope_2": s2,
                "scope_3": s3,
                "total_ghg": total_ghg,
                "scope_3_pct": s3_pct,
                "renewable_pct": ren_pct,
                "renewable_mwh": energy.get("renewable_mwh", 0.0),
                "total_mwh": energy.get("total_mwh_consumed", 0.0),
                "re100_committed": bool(energy.get("re100_committed", False)),
                "water_withdrawn_m3": waste.get("total_water_withdrawal_m3", 0.0),
                "water_recycled_pct": waste.get("water_recycled_pct", 0.0),
                "waste_diverted_pct": waste_div,
                "female_board_rep_pct": board_div,
                "independent_directors_pct": gov.get("independent_directors_pct", 0.0),
                "gender_pay_equity_ratio": gov.get("gender_pay_equity_ratio", 1.0),
                "supplier_signoff_pct": gov.get("supplier_code_of_conduct_signoff_pct", 0.0),
                "achieved_yoy_pct": yoy_red,
                "target_reduction_2030_pct": emissions.get("target_reduction_2030_pct", 45.0)
            },
            "benchmarks": benchmarks
        }

    def generate_roadmap_items(self, analysis: Dict[str, Any]) -> list:
        """Returns structured strategic roadmap items synthesized by IBM Granite."""
        return self.granite_client.generate_structured_recommendations(analysis)

    def generate_audit_report(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Constructs a comprehensive, transparent audit report."""
        em = data.get("emissions_metric_tons_co2e", {})
        en = data.get("renewable_energy", {})
        m = analysis["metrics"]
        yoy_status_label = "ON TRACK" if analysis["reduction_pass"] else "LAGGING BEHIND TARGET"

        # Ask Granite client for synthesized strategic recommendations
        granite_advice = self.granite_client.generate_esg_insights(analysis)
        mode_label = "IBM WatsonX Live API" if self.granite_client.is_live else "Local IBM Granite 3.0 Reasoning Engine"

        report = f"""
================================================================================
[*] ECOGRANITE ADVISOR: CORPORATE ESG AUDIT REPORT
================================================================================
Analysis Engine : {mode_label}
Model Selected  : {self.model_id}
Entity Name     : {data.get('company_name')}
Audit Cycle     : FY {data.get('reporting_year')}
Frameworks Used : {', '.join(data.get('standards', ['GRI', 'TCFD', 'SASB']))}
--------------------------------------------------------------------------------

[+] COMPOSITE ESG READINESS SCORE: {analysis['esg_readiness_score']} / 100
Rating Level    : {analysis['rating_tier']}

1. GREENHOUSE GAS (GHG) EMISSIONS BREAKDOWN:
   - Scope 1 (Direct Operations)       : {m['scope_1']:>10,.1f} MT CO2e ({(m['scope_1']/m['total_ghg']*100.0 if m['total_ghg']>0 else 0):.1f}%)
   - Scope 2 (Purchased Electricity)   : {m['scope_2']:>10,.1f} MT CO2e ({(m['scope_2']/m['total_ghg']*100.0 if m['total_ghg']>0 else 0):.1f}%)
   - Scope 3 (Supply Chain & Value)    : {m['scope_3']:>10,.1f} MT CO2e ({m['scope_3_pct']:.1f}%)
   - Total Carbon Footprint            : {m['total_ghg']:>10,.1f} MT CO2e
   - 2030 Science-Based Target (SBTi)  : {em.get('target_reduction_2030_pct', 0.0):.1f}% reduction
   - YoY Progress Achieved             : {em.get('achieved_reduction_yoy_pct', 0.0):.1f}% ({yoy_status_label})

2. RENEWABLE ENERGY & TRANSITION:
   - Total Consumption                 : {en.get('total_mwh_consumed', 0.0):>10,.1f} MWh
   - Clean / Renewable Energy Sourced  : {en.get('renewable_mwh', 0.0):>10,.1f} MWh ({m['renewable_pct']:.1f}%)
   - RE100 Initiative Pledged          : {'YES' if en.get('re100_committed') else 'NO'}

3. ESG BENCHMARK COMPLIANCE TABLE:
   {'-'*76}
   {'Metric':<38} {'Value':<10} {'Score':<8} {'Status':<12} {'Risk'}
   {'-'*76}
"""
        for metric, d in analysis["benchmarks"].items():
            report += f"   {metric:<38} {d['value']:<10} {d['score']:<8} {d['status']:<12} {d['risk']}\n"

        report += f"   {'-'*76}\n\n"
        report += granite_advice
        report += "\n================================================================================\n"
        return report


def main():
    parser = argparse.ArgumentParser(description="EcoGranite-Advisor: Autonomous ESG Sustainability Auditor")
    parser.add_argument(
        "--input", "-i",
        default="sample_esg_report.json",
        help="Path to input ESG document (.json, .pdf, or .docx). Default: sample_esg_report.json"
    )
    parser.add_argument(
        "--model", "-m",
        default="ibm-granite/granite-3.0-8b-instruct",
        help="IBM Granite foundation model ID."
    )
    args = parser.parse_args()

    # Resolve path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = args.input if os.path.isabs(args.input) else os.path.join(base_dir, args.input)

    advisor = EcoGraniteAdvisor(model_id=args.model)
    data = advisor.load_document(input_file)
    analysis = advisor.analyze_compliance(data)
    report = advisor.generate_audit_report(data, analysis)

    print(report)


if __name__ == "__main__":
    main()
