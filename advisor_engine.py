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
            "country": data.get("country", ""),
            "jurisdiction": data.get("jurisdiction", "Global Climate Standards"),
            "standards": data.get("standards", ["GRI", "TCFD", "SASB"]),
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

    def calculate_sector_weighted_score(self, analysis: Dict[str, Any], sector: str = "Technology & Software") -> Dict[str, Any]:
        """
        Calculates sector-specific materiality scores based on SASB SICS industry profiles.
        Supported sectors:
        - Technology & Software
        - Heavy Industry & Metals
        - Financial Institutions
        - General Enterprise
        """
        if not analysis.get("can_audit", True):
            return {"sector": sector, "weighted_score": 0.0, "weights": {}}

        m = analysis.get("metrics", {})
        ren_pct = m.get("renewable_pct", 0.0)
        yoy_red = m.get("achieved_yoy_pct", 0.0)
        waste_div = m.get("waste_diverted_pct", 0.0)
        water_rec = m.get("water_recycled_pct", 0.0)
        female_board = m.get("female_board_rep_pct", 0.0)
        indep_board = m.get("independent_directors_pct", 0.0)
        supplier_code = m.get("supplier_signoff_pct", 0.0)
        s3_pct = m.get("scope_3_pct", 60.0)

        # Base sub-scores normalized to 0-1.0
        score_ren = min(1.0, max(0.0, ren_pct / 60.0))
        score_yoy = min(1.0, max(0.0, yoy_red / 5.0))
        score_waste = min(1.0, max(0.0, waste_div / 75.0))
        score_water = min(1.0, max(0.0, water_rec / 50.0))
        score_board = min(1.0, max(0.0, female_board / 40.0))
        score_indep = min(1.0, max(0.0, indep_board / 75.0))
        score_supp = min(1.0, max(0.0, supplier_code / 95.0))
        # Scope 3 management: combination of supplier code signoff and target reduction
        score_s3 = (score_supp * 0.6) + (min(1.0, max(0.0, m.get("target_reduction_2030_pct", 45.0) / 45.0)) * 0.4)

        if sector == "Technology & Software":
            weights = {
                "Scope 3 Value Chain (45%)": 45.0,
                "Scope 2 Clean Power (20%)": 20.0,
                "Governance & Diversity (20%)": 20.0,
                "Water & Circular Waste (10%)": 10.0,
                "Scope 1 Direct (5%)": 5.0
            }
            comp_score = (
                (score_s3 * 45.0) +
                (score_ren * 20.0) +
                (((score_board + score_indep) / 2.0) * 20.0) +
                (((score_waste + score_water) / 2.0) * 10.0) +
                (score_yoy * 5.0)
            )
        elif sector == "Heavy Industry & Metals":
            weights = {
                "Scope 1 Direct Operations (35%)": 35.0,
                "Scope 2 Power & Energy (25%)": 25.0,
                "Scope 3 Value Chain (20%)": 20.0,
                "Water & Resource Circularity (15%)": 15.0,
                "Governance Accountability (5%)": 5.0
            }
            comp_score = (
                (score_yoy * 35.0) +
                (score_ren * 25.0) +
                (score_s3 * 20.0) +
                (((score_waste + score_water) / 2.0) * 15.0) +
                (score_board * 5.0)
            )
        elif sector == "Financial Institutions":
            weights = {
                "Scope 3 Financed Emissions (65%)": 65.0,
                "Board Independence & Risk (20%)": 20.0,
                "Scope 2 Energy Consumption (8%)": 8.0,
                "Water & Waste Efficiency (5%)": 5.0,
                "Scope 1 Operational Fleet (2%)": 2.0
            }
            comp_score = (
                (score_s3 * 65.0) +
                (score_indep * 20.0) +
                (score_ren * 8.0) +
                (score_waste * 5.0) +
                (score_yoy * 2.0)
            )
        else:
            weights = {
                "Environmental Pillar (60%)": 60.0,
                "Social Responsibility (15%)": 15.0,
                "Governance Architecture (15%)": 15.0,
                "Data Quality & Integrity (10%)": 10.0
            }
            comp_score = analysis.get("esg_readiness_score", 0.0)

        final_score = round(min(100.0, max(0.0, comp_score)), 1)
        return {
            "sector": sector,
            "weighted_score": final_score,
            "weights": weights,
            "alignment": "SASB SICS & ISSB IFRS S2 Materiality Standard"
        }

    def evaluate_double_materiality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        CSRD Double Materiality Matrix (ESRS 1 & ESRS 2).
        Calculates:
        1. Financial Risk Exposure (Outside-In): Carbon taxes, fossil volatility, regulatory penalties.
        2. Impact Materiality (Inside-Out): Planetary emissions magnitude, water depletion, social impact.
        """
        if not analysis.get("can_audit", True):
            return {"financial_risk": 0.0, "impact_materiality": 0.0, "quadrant": "Unrated"}

        m = analysis.get("metrics", {})
        tot_ghg = m.get("total_ghg", 0.0)
        ren_pct = m.get("renewable_pct", 0.0)
        s3_pct = m.get("scope_3_pct", 50.0)
        recon_status = analysis.get("data_quality", {}).get("reconciliation_status", "PASSED")

        # Outside-In Financial Risk (0 to 100):
        # Driven by high non-renewable exposure, reconciliation disqualification risk, and supply chain exposure
        f_risk = 20.0  # baseline regulatory risk
        if ren_pct < 60.0:
            f_risk += (60.0 - ren_pct) * 0.6  # carbon price exposure on brown grid power
        if s3_pct > 65.0:
            f_risk += 15.0  # value chain price volatility
        if recon_status != "PASSED":
            f_risk += 35.0  # statutory penalty & litigation risk
        financial_risk = round(min(100.0, max(10.0, f_risk)), 1)

        # Inside-Out Impact Materiality (0 to 100):
        # Driven by absolute GHG scale, resource circularity, water neutrality, and board diversity
        i_impact = 30.0
        if tot_ghg > 100000.0:
            i_impact += 35.0
        elif tot_ghg > 50000.0:
            i_impact += 20.0
        if m.get("waste_diverted_pct", 0.0) < 75.0:
            i_impact += 15.0
        if m.get("water_recycled_pct", 0.0) < 50.0:
            i_impact += 10.0
        impact_materiality = round(min(100.0, max(15.0, i_impact)), 1)

        # Quadrant classification
        if financial_risk >= 50.0 and impact_materiality >= 50.0:
            quadrant = "Critical Double Materiality Focus"
        elif financial_risk < 50.0 and impact_materiality >= 50.0:
            quadrant = "High Impact / Low Financial Risk (Planetary Stewardship)"
        elif financial_risk >= 50.0 and impact_materiality < 50.0:
            quadrant = "Low Impact / High Financial Risk (Enterprise Hedging)"
        else:
            quadrant = "Low Impact / Low Financial Risk (Routine Monitoring)"

        return {
            "financial_risk_score": financial_risk,
            "impact_materiality_score": impact_materiality,
            "quadrant": quadrant,
            "standard": "EU CSRD ESRS 1 & ESRS 2"
        }

    def generate_multi_framework_scorecard(self) -> List[Dict[str, Any]]:
        """
        Returns the multi-company framework scorecard summary across GRI, CSRD ESRS, and SEBI BRSR Core.
        """
        return [
            {
                "company_name": "EcoGlobal Enterprise Corp.",
                "audit_cycle": "FY 2024",
                "gri_baseline_score": "100 / 100 [A]",
                "csrd_esrs_score": "80 / 100 [A] (RE Share < 80%)",
                "sebi_brsr_status": "Compliant (BRSR Core Assured)",
                "primary_audit_priority": "Accelerate Scope 3 Tier-1 Supplier Telemetry"
            },
            {
                "company_name": "Siemens AG",
                "audit_cycle": "FY 2024",
                "gri_baseline_score": "100 / 100 [A]",
                "csrd_esrs_score": "95 / 100 [A] (84.0% RE Power)",
                "sebi_brsr_status": "Compliant (EU Taxonomy Aligned)",
                "primary_audit_priority": "Primary LCAs for Scope 3 Category 1 Raw Materials"
            },
            {
                "company_name": "Infosys Limited",
                "audit_cycle": "FY 2024",
                "gri_baseline_score": "100 / 100 [A]",
                "csrd_esrs_score": "90 / 100 [A] (74.8% RE Power)",
                "sebi_brsr_status": "BRSR Core Leader",
                "primary_audit_priority": "Campus Zero Liquid Discharge & VPPAs"
            }
        ]

    def calculate_framework_score(self, analysis: Dict[str, Any], framework: str = "GRI Baseline") -> Dict[str, Any]:
        """
        Real-time Statutory Framework Evaluator:
        Recalculates compliance readiness, penalties, and rating tier across:
        - GRI Baseline (Multi-stakeholder standard)
        - CSRD (ESRS Strict EU standard with RE >= 80% hurdle)
        - ISSB (IFRS S2 Financial Climate & Transition standard)
        """
        if not analysis.get("can_audit", True):
            return {
                "framework": framework,
                "score": 0.0,
                "tier": "[REFUSED] EXTRACTION FAILED",
                "penalties": ["Missing audit disclosures"],
                "focus": "N/A"
            }

        base_score = analysis.get("esg_readiness_score", 0.0)
        m = analysis.get("metrics", {})
        ren_pct = m.get("renewable_pct", 0.0)
        supp_code = m.get("supplier_signoff_pct", 0.0)
        target_2030 = m.get("target_reduction_2030_pct", 45.0)
        recon_status = analysis.get("data_quality", {}).get("reconciliation_status", "PASSED")

        penalties = []
        if recon_status != "PASSED":
            return {
                "framework": framework,
                "score": base_score,
                "tier": "[DISQUALIFIED] RECONCILIATION FAILURE",
                "penalties": ["GHG Protocol scope mismatch contradicts reported total"],
                "focus": "Data Reconciliation Integrity"
            }

        if "CSRD" in framework or "ESRS" in framework:
            fw_name = "CSRD (ESRS Strict)"
            score = base_score
            # ESRS E1 Climate Hurdle: RE share < 80% incurs a penalty under strict criteria
            if ren_pct < 80.0:
                pen_val = round((80.0 - ren_pct) * 0.5, 1)
                score -= pen_val
                penalties.append(f"ESRS E1 Penalty: Renewable power {ren_pct:.1f}% is below 80% EU hurdle (-{pen_val} pts)")
            # ESRS G1 Business Conduct: Supplier code of conduct < 95%
            if supp_code < 95.0:
                score -= 10.0
                penalties.append(f"ESRS G1 Penalty: Supplier Code sign-off {supp_code:.1f}% < 95% threshold (-10 pts)")
            
            score = round(max(0.0, min(100.0, score)), 1)
            tier = "[A] EXCELLENT (CSRD LEADER)" if score >= 85.0 else ("[B] GOOD (PROGRESSING)" if score >= 70.0 else ("[C] MODERATE" if score >= 50.0 else "[D] CRITICAL"))
            return {
                "framework": fw_name,
                "score": score,
                "tier": tier,
                "penalties": penalties,
                "focus": "Double Materiality, Value Chain Telemetry & 80% Clean Power Hurdle"
            }

        elif "ISSB" in framework or "IFRS" in framework:
            fw_name = "ISSB (IFRS S2)"
            score = base_score
            # IFRS S2 Climate-related Disclosures: Target must be aligned with 1.5°C pathway (>= 45% by 2030)
            if target_2030 < 45.0:
                score -= 15.0
                penalties.append(f"IFRS S2 Transition Risk: 2030 emissions reduction target {target_2030:.1f}% is below 45% SBTi 1.5°C threshold (-15 pts)")
            if m.get("scope_3_pct", 50.0) > 65.0 and supp_code < 90.0:
                score -= 10.0
                penalties.append("IFRS S2 Supply Chain Risk: Unmitigated Scope 3 exposure > 65% with < 90% supplier coverage (-10 pts)")

            score = round(max(0.0, min(100.0, score)), 1)
            tier = "[A] EXCELLENT (ISSB LEADER)" if score >= 85.0 else ("[B] GOOD (PROGRESSING)" if score >= 70.0 else ("[C] MODERATE" if score >= 50.0 else "[D] CRITICAL"))
            return {
                "framework": fw_name,
                "score": score,
                "tier": tier,
                "penalties": penalties,
                "focus": "Financial Capital Allocation, Transition Plans & Climate Value-at-Risk"
            }

        else:
            # GRI Baseline
            return {
                "framework": "GRI Baseline",
                "score": base_score,
                "tier": analysis.get("rating_tier", "[A] EXCELLENT (ESG LEADER)"),
                "penalties": penalties,
                "focus": "Multi-stakeholder Impact Transparency (GRI 300 Environmental & GRI 400 Social)"
            }


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
