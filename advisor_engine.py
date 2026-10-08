"""
EcoGranite-Advisor Engine
Comprehensive ESG Sustainability Auditor evaluating Scope 1-3 GHG emissions,
dynamic continuous scoring across E/S/G and Data Quality pillars, and risk-weighted compliance analysis.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, Optional, List

from esg_parser import parse_document
from granite_client import GraniteReasoningClient

logger = logging.getLogger(__name__)


class EcoGraniteAdvisor:
    def __init__(self, model_id: str = "ibm-granite/granite-3.0-8b-instruct"):
        self.model_id = model_id
        self.granite_client = GraniteReasoningClient(model_id=model_id)

    def load_document(self, file_source: Any) -> Dict[str, Any]:
        """Loads and normalizes an ESG report via the parser pipeline."""
        return parse_document(file_source)

    def calculate_continuous_score(
        self,
        value: Optional[float],
        target: float,
        max_weight: float,
        higher_is_better: bool = True
    ) -> float:
        """
        Computes a proportional, continuous score rather than an abrupt binary step.
        Handles missing data, lower-is-better boundaries, and zero targets.
        """
        if value is None:
            return 0.0

        if target <= 0.0:
            if higher_is_better:
                return max_weight if value >= target else 0.0
            else:
                return max_weight if value <= target else 0.0

        if higher_is_better:
            ratio = max(0.0, min(1.0, value / target))
            return round(ratio * max_weight, 2)
        else:
            # Lower is better: if value <= target, award full score.
            # If value > target, decay proportionally to target/value.
            if value <= target:
                return max_weight
            ratio = max(0.0, min(1.0, target / value))
            return round(ratio * max_weight, 2)

    def analyze_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs in-depth risk analysis, dynamic Scope 3 calculations,
        and continuous weighted ESG readiness scoring across 4 defensible pillars:
        - Environmental (60 pts)
        - Social (15 pts)
        - Governance (15 pts)
        - Data Quality & Reconciliation (10 pts)
        """
        # 1. Refusal on extraction failure
        if data.get("status") == "extraction_failed":
            return {
                "status": "extraction_failed",
                "can_audit": False,
                "company_name": data.get("company_name", "Unknown Entity"),
                "reporting_year": data.get("reporting_year", 2024),
                "esg_readiness_score": 0.0,
                "rating_tier": "[REFUSED] EXTRACTION FAILED",
                "errors": data.get("errors", ["Required ESG disclosures could not be extracted from document."]),
                "metrics": {},
                "benchmarks": {},
                "data_quality": {"reconciliation_status": "FAILED", "validation_warnings": data.get("errors", [])}
            }

        emissions = data.get("emissions_metric_tons_co2e", {})
        energy = data.get("renewable_energy", {})
        waste = data.get("water_and_waste", {})
        gov = data.get("social_and_governance", {})
        dq = data.get("data_quality", {})

        s1 = emissions.get("scope_1_direct")
        s2 = emissions.get("scope_2_indirect_market")
        s3 = emissions.get("scope_3_value_chain")
        total_ghg = emissions.get("total_ghg") or 0.0
        s3_pct = (s3 / total_ghg * 100.0) if (s3 is not None and total_ghg > 0) else 0.0

        # =========================================================================
        # PILLAR 1: ENVIRONMENTAL READINESS (60 Points)
        # =========================================================================
        # 1.1 Renewable Energy Share (Target: >= 60%, Weight: 25 pts)
        ren_pct = energy.get("renewable_share_pct")
        if ren_pct is not None:
            ren_score = self.calculate_continuous_score(ren_pct, 60.0, 25.0)
            ren_diff = ren_pct - 60.0
            ren_status = "Compliant" if ren_diff >= 0 else "Below Target"
            ren_risk = "Low" if ren_diff >= 0 else ("Moderate" if ren_diff >= -15.0 else "High")
            ren_val_str = f"{ren_pct:.1f}%"
            ren_var_str = f"{ren_diff:+.1f}%"
        else:
            ren_score = 0.0
            ren_status = "Unknown (Missing Data)"
            ren_risk = "Unrated"
            ren_val_str = "N/A"
            ren_var_str = "N/A"

        # 1.2 YoY Emissions Reduction (Target: >= 5%, Weight: 25 pts)
        yoy_red = emissions.get("achieved_reduction_yoy_pct")
        if yoy_red is not None:
            reduction_pass = yoy_red >= 5.0
            red_score = self.calculate_continuous_score(yoy_red, 5.0, 25.0)
            red_diff = yoy_red - 5.0
            red_status = "Compliant (On Track)" if reduction_pass else "Below Target (Lagging)"
            red_risk = "Low" if reduction_pass else ("Moderate" if yoy_red > 0 else "Severe")
            red_val_str = f"{yoy_red:.1f}%"
            red_var_str = f"{red_diff:+.1f}%"
        else:
            reduction_pass = False
            red_score = 0.0
            red_status = "Unknown (Missing Data)"
            red_risk = "Unrated"
            red_val_str = "N/A"
            red_var_str = "N/A"

        # 1.3 Waste Landfill Diversion (Target: >= 75%, Weight: 10 pts)
        waste_div = waste.get("waste_diverted_from_landfill_pct")
        if waste_div is not None:
            waste_score = self.calculate_continuous_score(waste_div, 75.0, 10.0)
            waste_diff = waste_div - 75.0
            waste_status = "Compliant" if waste_diff >= 0 else "Below Target"
            waste_risk = "Low" if waste_diff >= 0 else "Moderate"
            waste_val_str = f"{waste_div:.1f}%"
            waste_var_str = f"{waste_diff:+.1f}%"
        else:
            waste_score = 0.0
            waste_status = "Unknown (Missing Data)"
            waste_risk = "Unrated"
            waste_val_str = "N/A"
            waste_var_str = "N/A"

        env_subtotal = round(ren_score + red_score + waste_score, 1)

        # =========================================================================
        # PILLAR 2: SOCIAL READINESS (15 Points)
        # =========================================================================
        # 2.1 Gender Pay Equity Ratio (Target: >= 0.98, Weight: 7.5 pts)
        pay_equity = gov.get("gender_pay_equity_ratio")
        if pay_equity is not None:
            pay_score = self.calculate_continuous_score(pay_equity, 0.98, 7.5)
            pay_diff = pay_equity - 0.98
            pay_status = "Compliant" if pay_diff >= 0 else "Needs Improvement"
            pay_risk = "Low" if pay_diff >= 0 else "Moderate"
            pay_val_str = f"{pay_equity:.2f}"
            pay_var_str = f"{pay_diff:+.2f}"
        else:
            pay_score = 0.0
            pay_status = "Unknown (Missing Data)"
            pay_risk = "Unrated"
            pay_val_str = "N/A"
            pay_var_str = "N/A"

        # 2.2 Supplier Code of Conduct Sign-off (Target: >= 95%, Weight: 7.5 pts)
        supp_code = gov.get("supplier_code_of_conduct_signoff_pct")
        if supp_code is not None:
            supp_score = self.calculate_continuous_score(supp_code, 95.0, 7.5)
            supp_diff = supp_code - 95.0
            supp_status = "Compliant" if supp_diff >= 0 else "Below Target"
            supp_risk = "Low" if supp_diff >= 0 else "Moderate"
            supp_val_str = f"{supp_code:.1f}%"
            supp_var_str = f"{supp_diff:+.1f}%"
        else:
            supp_score = 0.0
            supp_status = "Unknown (Missing Data)"
            supp_risk = "Unrated"
            supp_val_str = "N/A"
            supp_var_str = "N/A"

        soc_subtotal = round(pay_score + supp_score, 1)

        # =========================================================================
        # PILLAR 3: GOVERNANCE READINESS (15 Points)
        # =========================================================================
        # 3.1 Board Gender Diversity (Target: >= 40%, Weight: 7.5 pts)
        board_div = gov.get("female_board_representation_pct")
        if board_div is not None:
            board_score = self.calculate_continuous_score(board_div, 40.0, 7.5)
            board_diff = board_div - 40.0
            board_status = "Compliant" if board_diff >= 0 else "Below Target"
            board_risk = "Low" if board_diff >= 0 else "Moderate"
            board_val_str = f"{board_div:.1f}%"
            board_var_str = f"{board_diff:+.1f}%"
        else:
            board_score = 0.0
            board_status = "Unknown (Missing Data)"
            board_risk = "Unrated"
            board_val_str = "N/A"
            board_var_str = "N/A"

        # 3.2 Independent Directors (Target: >= 75%, Weight: 7.5 pts)
        indep_dir = gov.get("independent_directors_pct")
        if indep_dir is not None:
            indep_score = self.calculate_continuous_score(indep_dir, 75.0, 7.5)
            indep_diff = indep_dir - 75.0
            indep_status = "Compliant" if indep_diff >= 0 else "Below Target"
            indep_risk = "Low" if indep_diff >= 0 else "Moderate"
            indep_val_str = f"{indep_dir:.1f}%"
            indep_var_str = f"{indep_diff:+.1f}%"
        else:
            indep_score = 0.0
            indep_status = "Unknown (Missing Data)"
            indep_risk = "Unrated"
            indep_val_str = "N/A"
            indep_var_str = "N/A"

        gov_subtotal = round(board_score + indep_score, 1)

        # =========================================================================
        # PILLAR 4: DATA QUALITY & RECONCILIATION (10 Points)
        # =========================================================================
        # 4.1 GHG Protocol Scope 1+2+3 Reconciliation (5 pts)
        reconciled = (dq.get("reconciliation_status") == "PASSED")
        recon_score = 5.0 if reconciled else 0.0

        # 4.2 Disclosure Completeness & Integrity (5 pts)
        comp_pct = dq.get("completeness_pct", 100.0)
        warnings = dq.get("validation_warnings", [])
        # Deduct 1.0 point per anomalous warning (boundary violation, NaN, etc.), minimum 0
        comp_score = max(0.0, round((comp_pct / 100.0 * 5.0) - (len(warnings) * 0.75), 1))

        dq_subtotal = round(recon_score + comp_score, 1)

        total_score = round(env_subtotal + soc_subtotal + gov_subtotal + dq_subtotal, 1)

        # =========================================================================
        # Rating Tier & Data Reconciliation Disqualification
        # =========================================================================
        reconciliation_failed = not reconciled
        if reconciliation_failed:
            rating = "[DISQUALIFIED] DATA INTEGRITY FAILURE (UNRECONCILED GHG TOTAL)"
        elif total_score >= 85.0:
            rating = "[A] EXCELLENT (ESG LEADER)"
        elif total_score >= 70.0:
            rating = "[B] GOOD (PROGRESSING)"
        elif total_score >= 50.0:
            rating = "[C] MODERATE (AT RISK)"
        else:
            rating = "[D] CRITICAL (NON-COMPLIANT)"

        benchmarks = {
            "Renewable Energy Share (Target >= 60%)": {
                "pillar": "Environmental",
                "value": ren_val_str,
                "variance": ren_var_str,
                "score": f"{ren_score}/25.0",
                "status": ren_status,
                "risk": ren_risk
            },
            "YoY Emissions Reduction (Target >= 5%)": {
                "pillar": "Environmental",
                "value": red_val_str,
                "variance": red_var_str,
                "score": f"{red_score}/25.0",
                "status": red_status,
                "risk": red_risk
            },
            "Waste Diversion (Target >= 75%)": {
                "pillar": "Environmental",
                "value": waste_val_str,
                "variance": waste_var_str,
                "score": f"{waste_score}/10.0",
                "status": waste_status,
                "risk": waste_risk
            },
            "Board Diversity (Target >= 40%)": {
                "pillar": "Governance",
                "value": board_val_str,
                "variance": board_var_str,
                "score": f"{board_score}/7.5",
                "status": board_status,
                "risk": board_risk
            },
            "Independent Directors (Target >= 75%)": {
                "pillar": "Governance",
                "value": indep_val_str,
                "variance": indep_var_str,
                "score": f"{indep_score}/7.5",
                "status": indep_status,
                "risk": indep_risk
            },
            "Gender Pay Equity (Target >= 0.98)": {
                "pillar": "Social",
                "value": pay_val_str,
                "variance": pay_var_str,
                "score": f"{pay_score}/7.5",
                "status": pay_status,
                "risk": pay_risk
            },
            "Supplier Code Sign-off (Target >= 95%)": {
                "pillar": "Social",
                "value": supp_val_str,
                "variance": supp_var_str,
                "score": f"{supp_score}/7.5",
                "status": supp_status,
                "risk": supp_risk
            },
            "GHG Scope Reconciliation (S1+S2+S3 = Total)": {
                "pillar": "Data Quality",
                "value": dq.get("reconciliation_status", "UNKNOWN"),
                "variance": f"{dq.get('total_ghg_variance', 0.0):.1f} MT",
                "score": f"{recon_score}/5.0",
                "status": "Compliant" if reconciled else "Failed Reconciliation",
                "risk": "Low" if reconciled else "Severe"
            }
        }

        return {
            "status": "success",
            "can_audit": True,
            "company_name": data.get("company_name", "Unknown Entity"),
            "reporting_year": data.get("reporting_year", "N/A"),
            "country": data.get("country", "India"),
            "jurisdiction": data.get("jurisdiction", "SEBI BRSR Core & Global Climate Standards"),
            "standards": data.get("standards", ["SEBI BRSR", "GRI", "TCFD", "SASB"]),
            "esg_readiness_score": total_score,
            "rating_tier": rating,
            "reduction_pass": reduction_pass,
            "pillars": {
                "environmental": {"score": env_subtotal, "max": 60.0},
                "social": {"score": soc_subtotal, "max": 15.0},
                "governance": {"score": gov_subtotal, "max": 15.0},
                "data_quality": {"score": dq_subtotal, "max": 10.0}
            },
            "metrics": {
                "scope_1": s1 if s1 is not None else 0.0,
                "scope_2": s2 if s2 is not None else 0.0,
                "scope_3": s3 if s3 is not None else 0.0,
                "total_ghg": total_ghg,
                "scope_3_pct": s3_pct,
                "renewable_pct": ren_pct if ren_pct is not None else 0.0,
                "renewable_mwh": energy.get("renewable_mwh", 0.0) or 0.0,
                "total_mwh": energy.get("total_mwh_consumed", 0.0) or 0.0,
                "re100_committed": bool(energy.get("re100_committed", False)),
                "water_withdrawn_m3": waste.get("total_water_withdrawal_m3", 0.0) or 0.0,
                "water_recycled_pct": waste.get("water_recycled_pct", 0.0) or 0.0,
                "waste_diverted_pct": waste_div if waste_div is not None else 0.0,
                "female_board_rep_pct": board_div if board_div is not None else 0.0,
                "independent_directors_pct": indep_dir if indep_dir is not None else 0.0,
                "gender_pay_equity_ratio": pay_equity if pay_equity is not None else 1.0,
                "supplier_signoff_pct": supp_code if supp_code is not None else 0.0,
                "achieved_yoy_pct": yoy_red if yoy_red is not None else 0.0,
                "target_reduction_2030_pct": emissions.get("target_reduction_2030_pct", 45.0) or 45.0
            },
            "benchmarks": benchmarks,
            "data_quality": dq
        }

    def generate_roadmap_items(self, analysis: Dict[str, Any]) -> list:
        """Returns structured strategic roadmap items synthesized by IBM Granite rule engine."""
        return self.granite_client.generate_structured_recommendations(analysis)

    def get_eight_step_roadmap(self) -> list:
        """Returns the standardized 8-step decarbonization workflow."""
        return self.granite_client.get_eight_step_roadmap()

    def generate_ai_insights(self, analysis: Dict[str, Any]) -> str:
        """Invokes IBM WatsonX / Local Granite model to generate narrative insights."""
        return self.granite_client.generate_esg_insights(analysis)

    def generate_audit_report(self, data: Dict[str, Any], analysis: Dict[str, Any], ai_insights: Optional[str] = None) -> str:
        """Constructs a comprehensive, transparent audit report."""
        if not analysis.get("can_audit", True):
            errors = analysis.get("errors", [])
            return f"""
================================================================================
[!] ECOGRANITE ADVISOR: AUDIT REFUSAL (EXTRACTION FAILED)
================================================================================
Entity Name  : {analysis.get('company_name', 'Unknown')}
Audit Status : FAILED PRE-AUDIT EXTRACTION

Reason: Required corporate disclosures could not be extracted with fidelity.
Extraction Errors:
""" + "\n".join(f" - {err}" for err in errors) + "\n================================================================================\n"

        em = data.get("emissions_metric_tons_co2e", {})
        en = data.get("renewable_energy", {})
        m = analysis["metrics"]
        dq = analysis.get("data_quality", {})
        pillars = analysis.get("pillars", {})
        yoy_status_label = "ON TRACK" if analysis["reduction_pass"] else "LAGGING BEHIND TARGET"

        # If insights were not pre-generated, generate now
        if ai_insights is None:
            ai_insights = self.generate_ai_insights(analysis)

        granite_meta = self.granite_client.get_runtime_metadata()
        mode_label = granite_meta.get("label", "IBM Granite 3.0 Reasoning Engine")

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

[+] COMPOSITE ESG SUSTAINABILITY READINESS SCORE: {analysis['esg_readiness_score']} / 100
Rating Level    : {analysis['rating_tier']}
Score Breakdown :
   - Environmental Pillar       : {pillars.get('environmental', {}).get('score', 0)} / {pillars.get('environmental', {}).get('max', 60)} pts
   - Social Pillar              : {pillars.get('social', {}).get('score', 0)} / {pillars.get('social', {}).get('max', 15)} pts
   - Governance Pillar          : {pillars.get('governance', {}).get('score', 0)} / {pillars.get('governance', {}).get('max', 15)} pts
   - Data Quality & Integrity   : {pillars.get('data_quality', {}).get('score', 0)} / {pillars.get('data_quality', {}).get('max', 10)} pts

DATA RECONCILIATION & INTEGRITY:
   - Scope 1+2+3 vs Total GHG   : {dq.get('reconciliation_status', 'UNKNOWN')} (Variance: {dq.get('total_ghg_variance', 0.0):.1f} MT)
   - Completeness Score         : {dq.get('completeness_pct', 0.0):.1f}%
"""
        if dq.get("validation_warnings"):
            report += "   - Validation Warnings        :\n"
            for w in dq["validation_warnings"]:
                report += f"     * [!] {w}\n"

        report += f"""
1. GREENHOUSE GAS (GHG) EMISSIONS BREAKDOWN:
   - Scope 1 (Direct Operations)       : {m['scope_1']:>10,.1f} MT CO2e ({(m['scope_1']/m['total_ghg']*100.0 if m['total_ghg']>0 else 0):.1f}%)
   - Scope 2 (Purchased Electricity)   : {m['scope_2']:>10,.1f} MT CO2e ({(m['scope_2']/m['total_ghg']*100.0 if m['total_ghg']>0 else 0):.1f}%)
   - Scope 3 (Supply Chain & Value)    : {m['scope_3']:>10,.1f} MT CO2e ({m['scope_3_pct']:.1f}%)
   - Total Carbon Footprint            : {m['total_ghg']:>10,.1f} MT CO2e
   - 2030 Science-Based Target (SBTi)  : {em.get('target_reduction_2030_pct', 0.0):.1f}% reduction
   - YoY Progress Achieved             : {em.get('achieved_reduction_yoy_pct', 0.0):.1f}% ({yoy_status_label})

2. RENEWABLE ENERGY & TRANSITION:
   - Total Consumption                 : {en.get('total_mwh_consumed', 0.0) or 0.0:>10,.1f} MWh
   - Clean / Renewable Energy Sourced  : {en.get('renewable_mwh', 0.0) or 0.0:>10,.1f} MWh ({m['renewable_pct']:.1f}%)
   - RE100 Initiative Pledged          : {'YES' if en.get('re100_committed') else 'NO'}

3. ESG BENCHMARK COMPLIANCE TABLE:
   {'-'*76}
   {'Metric':<38} {'Value':<10} {'Score':<8} {'Status':<12} {'Risk'}
   {'-'*76}
"""
        for metric, d in analysis["benchmarks"].items():
            report += f"   {metric:<38} {d['value']:<10} {d['score']:<8} {d['status']:<12} {d['risk']}\n"

        report += f"   {'-'*76}\n\n"
        report += ai_insights
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
