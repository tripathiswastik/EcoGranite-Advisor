"""
EcoGranite-Advisor engine.

Scores ESG disclosures across four pillars (Environmental 60, Social 15,
Governance 15, Data Quality 10), applies sector / framework views, and
builds the text audit report.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from esg_parser import parse_document, validate_and_normalize_esg
from granite_client import GraniteReasoningClient, load_environment

DEFAULT_MODEL_ID = "ibm-granite/granite-3.0-8b-instruct"
RECONCILIATION_POINTS = 5.0
COMPLETENESS_POINTS = 5.0
WARNING_PENALTY_POINTS = 0.75  # deducted per validation warning
TIER_A, TIER_B, TIER_C = 85.0, 70.0, 50.0
CSRD_RENEWABLE_HURDLE_PCT = 80.0
CSRD_SUPPLIER_HURDLE_PCT = 95.0
ISSB_TARGET_HURDLE_PCT = 45.0
SCORECARD_SAMPLES = (
    ("EcoGlobal Enterprise Corp.", "sample_esg_report.json"),
    ("Siemens AG", "sample_esg_report_siemens.json"),
    ("Infosys Limited", "sample_esg_report_india.json"),
)
MISSING = "Unknown (Missing Data)"


@dataclass(frozen=True)
class MetricSpec:
    """Definition of one scored benchmark."""

    label: str
    pillar: str
    section: str
    key: str
    target: float
    weight: float
    is_percent: bool = True
    decimals: int = 1
    ok_status: str = "Compliant"
    below_status: str = "Below Target"


def _renewable_risk(value: float) -> str:
    diff = value - 60.0
    if diff >= 0:
        return "Low"
    return "Moderate" if diff >= -15.0 else "High"


def _yoy_risk(value: float) -> str:
    if value >= 5.0:
        return "Low"
    return "Moderate" if value > 0 else "Severe"


def _default_risk(value: float, target: float) -> str:
    return "Low" if value >= target else "Moderate"


SPECS: tuple[tuple[MetricSpec, Optional[Callable[[float], str]]], ...] = (
    (MetricSpec("Renewable Energy Share (Target >= 60%)", "Environmental", "renewable_energy",
                "renewable_share_pct", 60.0, 25.0), _renewable_risk),
    (MetricSpec("YoY Emissions Reduction (Target >= 5%)", "Environmental", "emissions_metric_tons_co2e",
                "achieved_reduction_yoy_pct", 5.0, 25.0,
                ok_status="Compliant (On Track)", below_status="Below Target (Lagging)"), _yoy_risk),
    (MetricSpec("Waste Diversion (Target >= 75%)", "Environmental", "water_and_waste",
                "waste_diverted_from_landfill_pct", 75.0, 10.0), None),
    (MetricSpec("Board Diversity (Target >= 40%)", "Governance", "social_and_governance",
                "female_board_representation_pct", 40.0, 7.5), None),
    (MetricSpec("Independent Directors (Target >= 75%)", "Governance", "social_and_governance",
                "independent_directors_pct", 75.0, 7.5), None),
    (MetricSpec("Gender Pay Equity (Target >= 0.98)", "Social", "social_and_governance",
                "gender_pay_equity_ratio", 0.98, 7.5, is_percent=False, decimals=2,
                below_status="Needs Improvement"), None),
    (MetricSpec("Supplier Code Sign-off (Target >= 95%)", "Social", "social_and_governance",
                "supplier_code_of_conduct_signoff_pct", 95.0, 7.5), None),
)

SECTOR_WEIGHTS: dict[str, dict[str, float]] = {
    "Technology & Software": {
        "Scope 3 Value Chain (45%)": 45.0, "Scope 2 Clean Power (20%)": 20.0,
        "Governance & Diversity (20%)": 20.0, "Water & Circular Waste (10%)": 10.0,
        "Scope 1 Direct (5%)": 5.0,
    },
    "Heavy Industry & Metals": {
        "Scope 1 Direct Operations (35%)": 35.0, "Scope 2 Power & Energy (25%)": 25.0,
        "Scope 3 Value Chain (20%)": 20.0, "Water & Resource Circularity (15%)": 15.0,
        "Governance Accountability (5%)": 5.0,
    },
    "Financial Institutions": {
        "Scope 3 Financed Emissions (65%)": 65.0, "Board Independence & Risk (20%)": 20.0,
        "Scope 2 Energy Consumption (8%)": 8.0, "Water & Waste Efficiency (5%)": 5.0,
        "Scope 1 Operational Fleet (2%)": 2.0,
    },
    "General Enterprise": {
        "Environmental Pillar (60%)": 60.0, "Social Responsibility (15%)": 15.0,
        "Governance Architecture (15%)": 15.0, "Data Quality & Integrity (10%)": 10.0,
    },
}


def _fraction(value: Optional[float], target: float) -> float:
    """Returns value/target clamped to 0..1 (0 if value is missing)."""
    if value is None or target <= 0:
        return 0.0
    return min(1.0, max(0.0, value / target))


def _tier(score: float, labels: tuple[str, str, str, str]) -> str:
    """Maps a score to a tier label using the shared 85/70/50 thresholds."""
    if score >= TIER_A:
        return labels[0]
    if score >= TIER_B:
        return labels[1]
    return labels[2] if score >= TIER_C else labels[3]


MAIN_TIERS = ("[A] EXCELLENT (ESG LEADER)", "[B] GOOD (PROGRESSING)",
              "[C] MODERATE (AT RISK)", "[D] CRITICAL (NON-COMPLIANT)")
CSRD_TIERS = ("[A] EXCELLENT (CSRD LEADER)", "[B] GOOD (PROGRESSING)", "[C] MODERATE", "[D] CRITICAL")
ISSB_TIERS = ("[A] EXCELLENT (ISSB LEADER)", "[B] GOOD (PROGRESSING)", "[C] MODERATE", "[D] CRITICAL")


def _fmt(value: Any, spec: str = ".1f", suffix: str = "") -> str:
    """Formats a number, or 'N/A' when it is None."""
    return "N/A" if value is None else f"{value:{spec}}{suffix}"


class EcoGraniteAdvisor:
    """Scores ESG disclosures and produces reports."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self.granite_client = GraniteReasoningClient(model_id=model_id)

    def load_document(self, file_source: Any) -> dict[str, Any]:
        """Loads and normalizes an ESG report via the parser pipeline."""
        return parse_document(file_source)

    def calculate_continuous_score(
        self, value: Optional[float], target: float, max_weight: float, higher_is_better: bool = True
    ) -> float:
        """Proportional score instead of a binary cutoff. Missing value scores 0."""
        if value is None:
            return 0.0
        if target <= 0.0:
            meets = value >= target if higher_is_better else value <= target
            return max_weight if meets else 0.0
        if higher_is_better:
            return round(_fraction(value, target) * max_weight, 2)
        if value <= target:
            return max_weight
        return round(min(1.0, target / value) * max_weight, 2)

    def _evaluate_metric(
        self, spec: MetricSpec, data: dict[str, Any], risk_fn: Optional[Callable[[float], str]]
    ) -> dict[str, Any]:
        """Scores one benchmark. Missing -> 'Unknown'; out-of-range % -> 'Invalid'."""
        value = (data.get(spec.section) or {}).get(spec.key)
        row = {"pillar": spec.pillar, "raw": value, "max": spec.weight}
        unit = "%" if spec.is_percent else ""
        if value is None:
            row.update(score=0.0, status=MISSING, risk="Unrated", value="N/A", variance="N/A")
        elif spec.is_percent and not 0.0 <= value <= 100.0:
            row.update(score=0.0, status="Invalid (Out of Range)", risk="Severe",
                       value=f"{value:.{spec.decimals}f}{unit}", variance="N/A")
        else:
            diff = value - spec.target
            risk = risk_fn(value) if risk_fn else _default_risk(value, spec.target)
            row.update(
                score=self.calculate_continuous_score(value, spec.target, spec.weight),
                status=spec.ok_status if diff >= 0 else spec.below_status,
                risk=risk,
                value=f"{value:.{spec.decimals}f}{unit}",
                variance=f"{diff:+.{spec.decimals}f}{unit}",
            )
        return row

    def _data_quality_points(self, dq: dict[str, Any]) -> tuple[float, float]:
        """Returns (reconciliation points, completeness points)."""
        recon = RECONCILIATION_POINTS if dq.get("reconciliation_status") == "PASSED" else 0.0
        completeness = float(dq.get("completeness_pct", 100.0))
        warnings = dq.get("validation_warnings", [])
        raw = completeness / 100.0 * COMPLETENESS_POINTS - len(warnings) * WARNING_PENALTY_POINTS
        return recon, max(0.0, round(raw, 1))

    @staticmethod
    def _rating(total: float, recon_status: str) -> str:
        if recon_status == "FAILED":
            return "[DISQUALIFIED] DATA INTEGRITY FAILURE (UNRECONCILED GHG TOTAL)"
        if recon_status != "PASSED" and total >= TIER_A:
            total = TIER_A - 0.1  # unverified totals cannot reach the leader tier
        return _tier(total, MAIN_TIERS)

    @staticmethod
    def _refusal(data: dict[str, Any]) -> dict[str, Any]:
        errors = data.get("errors", ["Required ESG disclosures could not be extracted from document."])
        return {
            "status": "extraction_failed", "can_audit": False,
            "company_name": data.get("company_name", "Unknown Entity"),
            "reporting_year": data.get("reporting_year", "N/A"),
            "esg_readiness_score": 0.0, "rating_tier": "[REFUSED] EXTRACTION FAILED",
            "errors": errors, "metrics": {}, "benchmarks": {},
            "data_quality": {"reconciliation_status": "FAILED", "validation_warnings": errors},
        }

    def analyze_compliance(self, data: dict[str, Any]) -> dict[str, Any]:
        """Scores a disclosure. Raw (un-normalized) dicts are normalized first."""
        if data.get("status") != "extraction_failed" and "data_quality" not in data:
            data = validate_and_normalize_esg(data)
        if data.get("status") == "extraction_failed":
            return self._refusal(data)

        emissions = data.get("emissions_metric_tons_co2e") or {}
        dq = data.get("data_quality") or {}
        rows = {spec.label: self._evaluate_metric(spec, data, risk) for spec, risk in SPECS}
        recon_pts, comp_pts = self._data_quality_points(dq)
        recon_status = dq.get("reconciliation_status", "UNVERIFIED")

        def pillar(name: str) -> float:
            return round(float(sum(r["score"] for r in rows.values() if r["pillar"] == name)), 1)

        pillars = {
            "environmental": {"score": pillar("Environmental"), "max": 60.0},
            "social": {"score": pillar("Social"), "max": 15.0},
            "governance": {"score": pillar("Governance"), "max": 15.0},
            "data_quality": {"score": round(recon_pts + comp_pts, 1), "max": 10.0},
        }
        total = round(sum(p["score"] for p in pillars.values()), 1)

        # Use {**..., "score": ...} instead of PEP 584 dict union for Python 3.8/3.7 compatibility
        benchmarks = {
            label: {
                **{k: r[k] for k in ("pillar", "value", "variance", "status", "risk")},
                "score": f"{r['score']}/{r['max']}"
            }
            for label, r in rows.items()
        }
        reconciled_labels = {
            "PASSED": ("Compliant", "Low"),
            "FAILED": ("Failed Reconciliation", "Severe"),
        }.get(recon_status, ("Unverified (No Reported Total)", "Moderate"))
        benchmarks["GHG Scope Reconciliation (S1+S2+S3 = Total)"] = {
            "pillar": "Data Quality", "value": recon_status,
            "variance": f"{dq.get('total_ghg_variance', 0.0):.1f} MT",
            "score": f"{recon_pts}/{RECONCILIATION_POINTS}",
            "status": reconciled_labels[0], "risk": reconciled_labels[1],
        }
        return self._build_result(data, emissions, rows, benchmarks, pillars, total, recon_status, dq)

    def _build_result(
        self, data: dict[str, Any], emissions: dict[str, Any], rows: dict[str, dict[str, Any]],
        benchmarks: dict[str, Any], pillars: dict[str, Any], total: float, recon_status: str,
        dq: dict[str, Any],
    ) -> dict[str, Any]:
        """Assembles the analysis dict returned by analyze_compliance."""
        energy = data.get("renewable_energy") or {}
        waste = data.get("water_and_waste") or {}
        gov = data.get("social_and_governance") or {}
        s3 = emissions.get("scope_3_value_chain")
        total_ghg = emissions.get("total_ghg") or 0.0
        raw = {spec.key: rows[spec.label]["raw"] for spec, _ in SPECS}
        yoy = raw["achieved_reduction_yoy_pct"]
        return {
            "status": "success", "can_audit": True,
            "company_name": data.get("company_name", "Unknown Entity"),
            "reporting_year": data.get("reporting_year") or "N/A",
            "country": data.get("country", ""),
            "jurisdiction": data.get("jurisdiction", "Global Climate Standards"),
            "standards": data.get("standards", []),
            "esg_readiness_score": total,
            "rating_tier": self._rating(total, recon_status),
            "reduction_pass": yoy is not None and yoy >= 5.0,
            "pillars": pillars,
            "metrics": {
                "scope_1": emissions.get("scope_1_direct") or 0.0,
                "scope_2": emissions.get("scope_2_indirect_market") or 0.0,
                "scope_3": s3 or 0.0,
                "total_ghg": total_ghg,
                "scope_3_pct": (s3 / total_ghg * 100.0) if (s3 is not None and total_ghg > 0) else 0.0,
                "renewable_pct": raw["renewable_share_pct"] or 0.0,
                "renewable_mwh": energy.get("renewable_mwh") or 0.0,
                "total_mwh": energy.get("total_mwh_consumed") or 0.0,
                "re100_committed": bool(energy.get("re100_committed", False)),
                "water_withdrawn_m3": waste.get("total_water_withdrawal_m3") or 0.0,
                "water_recycled_pct": waste.get("water_recycled_pct") or 0.0,
                "waste_diverted_pct": raw["waste_diverted_from_landfill_pct"] or 0.0,
                "female_board_rep_pct": raw["female_board_representation_pct"] or 0.0,
                "independent_directors_pct": raw["independent_directors_pct"] or 0.0,
                "gender_pay_equity_ratio": raw["gender_pay_equity_ratio"],  # None = not disclosed
                "supplier_signoff_pct": raw["supplier_code_of_conduct_signoff_pct"] or 0.0,
                "achieved_yoy_pct": yoy or 0.0,
                "target_reduction_2030_pct": emissions.get("target_reduction_2030_pct"),
                "csr_spend_pct_net_profit": gov.get("csr_spend_pct_net_profit"),
                "zero_liquid_discharge": waste.get("zero_liquid_discharge"),
            },
            "benchmarks": benchmarks,
            "data_quality": dq,
        }

    def generate_roadmap_items(self, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Returns structured roadmap items from the Granite rule engine."""
        return self.granite_client.generate_structured_recommendations(analysis)

    def get_eight_step_roadmap(self) -> list[dict[str, Any]]:
        """Returns the standardized 8-step decarbonization workflow."""
        return self.granite_client.get_eight_step_roadmap()

    def generate_ai_insights(self, analysis: dict[str, Any]) -> str:
        """Generates narrative insights via WatsonX or the local engine."""
        return self.granite_client.generate_esg_insights(analysis)

    @staticmethod
    def _refusal_report(analysis: dict[str, Any]) -> str:
        bar = "=" * 80
        lines = "\n".join(f" - {err}" for err in analysis.get("errors", []))
        return (
            f"\n{bar}\n[!] ECOGRANITE ADVISOR: AUDIT REFUSAL (EXTRACTION FAILED)\n{bar}\n"
            f"Entity Name  : {analysis.get('company_name', 'Unknown')}\n"
            "Audit Status : FAILED PRE-AUDIT EXTRACTION\n\n"
            "Reason: Required corporate disclosures could not be extracted with fidelity.\n"
            f"Extraction Errors:\n{lines}\n{bar}\n"
        )

    def generate_audit_report(
        self, data: dict[str, Any], analysis: dict[str, Any], ai_insights: Optional[str] = None
    ) -> str:
        """Builds the text audit report. Missing values print as N/A."""
        if not analysis.get("can_audit", True):
            return self._refusal_report(analysis)
        if ai_insights is None:
            ai_insights = self.generate_ai_insights(analysis)
        em = data.get("emissions_metric_tons_co2e") or {}
        en = data.get("renewable_energy") or {}
        m, dq, pillars = analysis["metrics"], analysis.get("data_quality", {}), analysis.get("pillars", {})
        mode_label = self.granite_client.get_runtime_metadata().get("label", "IBM Granite 3.0")
        total = m["total_ghg"]

        def share(value: float) -> float:
            return value / total * 100.0 if total > 0 else 0.0

        def pts(name: str) -> str:
            p = pillars.get(name, {})
            return f"{p.get('score', 0)} / {p.get('max', 0)} pts"

        lines = [
            "", "=" * 80, "[*] ECOGRANITE ADVISOR: CORPORATE ESG AUDIT REPORT", "=" * 80,
            f"Analysis Engine : {mode_label}",
            f"Model Selected  : {self.model_id}",
            f"Entity Name     : {data.get('company_name')}",
            f"Audit Cycle     : FY {analysis.get('reporting_year')}",
            f"Frameworks Used : {', '.join(analysis.get('standards') or ['Not disclosed'])}",
            "-" * 80, "",
            f"[+] COMPOSITE ESG SUSTAINABILITY READINESS SCORE: {analysis['esg_readiness_score']} / 100",
            f"Rating Level    : {analysis['rating_tier']}",
            "Score Breakdown :",
            f"   - Environmental Pillar       : {pts('environmental')}",
            f"   - Social Pillar              : {pts('social')}",
            f"   - Governance Pillar          : {pts('governance')}",
            f"   - Data Quality & Integrity   : {pts('data_quality')}",
            "", "DATA RECONCILIATION & INTEGRITY:",
            f"   - Scope 1+2+3 vs Total GHG   : {dq.get('reconciliation_status', 'UNKNOWN')} "
            f"(Variance: {dq.get('total_ghg_variance', 0.0):.1f} MT)",
            f"   - Completeness Score         : {dq.get('completeness_pct', 0.0):.1f}%",
        ]
        if dq.get("validation_warnings"):
            lines.append("   - Validation Warnings        :")
            lines.extend(f"     * [!] {w}" for w in dq["validation_warnings"])
        yoy_label = "ON TRACK" if analysis["reduction_pass"] else "LAGGING BEHIND TARGET"
        lines += [
            "", "1. GREENHOUSE GAS (GHG) EMISSIONS BREAKDOWN:",
            f"   - Scope 1 (Direct Operations)       : {m['scope_1']:>10,.1f} MT CO2e ({share(m['scope_1']):.1f}%)",
            f"   - Scope 2 (Purchased Electricity)   : {m['scope_2']:>10,.1f} MT CO2e ({share(m['scope_2']):.1f}%)",
            f"   - Scope 3 (Supply Chain & Value)    : {m['scope_3']:>10,.1f} MT CO2e ({m['scope_3_pct']:.1f}%)",
            f"   - Total Carbon Footprint            : {total:>10,.1f} MT CO2e",
            f"   - 2030 Science-Based Target (SBTi)  : {_fmt(em.get('target_reduction_2030_pct'), '.1f', '%')} reduction",
            f"   - YoY Progress Achieved             : {_fmt(em.get('achieved_reduction_yoy_pct'), '.1f', '%')} ({yoy_label})",
            "", "2. RENEWABLE ENERGY & TRANSITION:",
            f"   - Total Consumption                 : {en.get('total_mwh_consumed') or 0.0:>10,.1f} MWh",
            f"   - Clean / Renewable Energy Sourced  : {en.get('renewable_mwh') or 0.0:>10,.1f} MWh ({m['renewable_pct']:.1f}%)",
            f"   - RE100 Initiative Pledged          : {'YES' if en.get('re100_committed') else 'NO'}",
            "", "3. ESG BENCHMARK COMPLIANCE TABLE:", "   " + "-" * 76,
            f"   {'Metric':<38} {'Value':<10} {'Score':<8} {'Status':<12} {'Risk'}", "   " + "-" * 76,
        ]
        for name, row in analysis["benchmarks"].items():
            lines.append(f"   {name:<38} {row['value']:<10} {row['score']:<8} {row['status']:<12} {row['risk']}")
        lines += ["   " + "-" * 76, "", ai_insights, "=" * 80, ""]
        return "\n".join(lines)

    def calculate_sector_weighted_score(
        self, analysis: dict[str, Any], sector: str = "Technology & Software"
    ) -> dict[str, Any]:
        """Sector-weighted score. Raises ValueError for an unsupported sector.

        Sub-scores are proxies built from disclosed metrics (e.g. 'Scope 3'
        uses supplier sign-off and the 2030 target); see ``proxy_note``.
        """
        if sector not in SECTOR_WEIGHTS:
            raise ValueError(f"Unsupported sector '{sector}'. Choose from {list(SECTOR_WEIGHTS)}")
        weights = SECTOR_WEIGHTS[sector]
        if not analysis.get("can_audit", True):
            return {"sector": sector, "weighted_score": 0.0, "weights": {}}
        m = analysis.get("metrics", {})
        ren = _fraction(m.get("renewable_pct"), 60.0)
        yoy = _fraction(m.get("achieved_yoy_pct"), 5.0)
        waste = _fraction(m.get("waste_diverted_pct"), 75.0)
        water = _fraction(m.get("water_recycled_pct"), 50.0)
        board = _fraction(m.get("female_board_rep_pct"), 40.0)
        indep = _fraction(m.get("independent_directors_pct"), 75.0)
        supplier = _fraction(m.get("supplier_signoff_pct"), 95.0)
        target = _fraction(m.get("target_reduction_2030_pct"), 45.0)
        s3 = supplier * 0.6 + target * 0.4
        parts = {
            "Technology & Software": (s3, ren, (board + indep) / 2, (waste + water) / 2, yoy),
            "Heavy Industry & Metals": (yoy, ren, s3, (waste + water) / 2, board),
            "Financial Institutions": (s3, indep, ren, waste, yoy),
        }
        if sector == "General Enterprise":
            raw_score = analysis.get("esg_readiness_score", 0.0)
        else:
            raw_score = sum(frac * w for frac, w in zip(parts[sector], weights.values()))
        return {
            "sector": sector,
            "weighted_score": round(min(100.0, max(0.0, raw_score)), 1),
            "weights": weights,
            "alignment": "SASB SICS & ISSB IFRS S2 Materiality Standard",
            "proxy_note": "Sub-scores are proxies from available disclosures, not sector-specific metrics.",
        }

    def evaluate_double_materiality(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """CSRD double materiality: outside-in financial risk vs inside-out impact."""
        if not analysis.get("can_audit", True):
            return {"financial_risk_score": 0.0, "impact_materiality_score": 0.0, "quadrant": "Unrated"}
        m = analysis.get("metrics", {})
        recon = analysis.get("data_quality", {}).get("reconciliation_status", "UNVERIFIED")

        financial = 20.0
        if m.get("renewable_pct", 0.0) < 60.0:
            financial += (60.0 - m.get("renewable_pct", 0.0)) * 0.6
        if m.get("scope_3_pct", 0.0) > 65.0:
            financial += 15.0
        if recon != "PASSED":
            financial += 35.0
        financial = round(min(100.0, max(10.0, financial)), 1)

        impact = 30.0
        total = m.get("total_ghg", 0.0)
        impact += 35.0 if total > 100000.0 else (20.0 if total > 50000.0 else 0.0)
        impact += 15.0 if m.get("waste_diverted_pct", 0.0) < 75.0 else 0.0
        impact += 10.0 if m.get("water_recycled_pct", 0.0) < 50.0 else 0.0
        impact = round(min(100.0, max(15.0, impact)), 1)

        quadrants = {
            (True, True): "Critical Double Materiality Focus",
            (False, True): "High Impact / Low Financial Risk (Planetary Stewardship)",
            (True, False): "Low Impact / High Financial Risk (Enterprise Hedging)",
            (False, False): "Low Impact / Low Financial Risk (Routine Monitoring)",
        }
        return {
            "financial_risk_score": financial,
            "impact_materiality_score": impact,
            "quadrant": quadrants[(financial >= 50.0, impact >= 50.0)],
            "standard": "EU CSRD ESRS 1 & ESRS 2",
        }

    def generate_multi_framework_scorecard(self, base_dir: Optional[str] = None) -> list[dict[str, Any]]:
        """Computes the cross-framework scorecard from the bundled sample files.

        All scores are calculated live; nothing is hard-coded.
        """
        folder = base_dir or os.path.dirname(os.path.abspath(__file__))
        rows = []
        for display_name, filename in SCORECARD_SAMPLES:
            analysis = self.analyze_compliance(self.load_document(os.path.join(folder, filename)))
            gri = self.calculate_framework_score(analysis, "GRI Baseline")
            csrd = self.calculate_framework_score(analysis, "CSRD (ESRS Strict)")
            roadmap = self.generate_roadmap_items(analysis)
            rows.append({
                "company_name": display_name,
                "audit_cycle": f"FY {analysis.get('reporting_year')}",
                "gri_baseline_score": f"{gri['score']} / 100 {gri['tier'].split(' ')[0]}",
                "csrd_esrs_score": f"{csrd['score']} / 100 {csrd['tier'].split(' ')[0]}",
                "sebi_brsr_status": "Not assessed (no BRSR Core data model)",
                "primary_audit_priority": roadmap[0]["title"] if roadmap else "N/A",
            })
        return rows

    def calculate_framework_score(self, analysis: dict[str, Any], framework: str = "GRI Baseline") -> dict[str, Any]:
        """Recalculates readiness under GRI Baseline, CSRD (ESRS Strict) or ISSB (IFRS S2)."""
        if not analysis.get("can_audit", True):
            return {"framework": framework, "score": 0.0, "tier": "[REFUSED] EXTRACTION FAILED",
                    "penalties": ["Missing audit disclosures"], "focus": "N/A"}
        base = analysis.get("esg_readiness_score", 0.0)
        m = analysis.get("metrics", {})
        recon = analysis.get("data_quality", {}).get("reconciliation_status", "UNVERIFIED")
        if recon == "FAILED":
            return {"framework": framework, "score": base, "tier": "[DISQUALIFIED] RECONCILIATION FAILURE",
                    "penalties": ["GHG Protocol scope mismatch contradicts reported total"],
                    "focus": "Data Reconciliation Integrity"}
        if "CSRD" in framework or "ESRS" in framework:
            return self._csrd_score(base, m)
        if "ISSB" in framework or "IFRS" in framework:
            return self._issb_score(base, m)
        return {"framework": "GRI Baseline", "score": base, "tier": analysis.get("rating_tier", ""),
                "penalties": [],
                "focus": "Multi-stakeholder Impact Transparency (GRI 300 Environmental & GRI 400 Social)"}

    @staticmethod
    def _csrd_score(base: float, m: dict[str, Any]) -> dict[str, Any]:
        score, penalties = base, []
        renewable = m.get("renewable_pct", 0.0)
        if renewable < CSRD_RENEWABLE_HURDLE_PCT:
            penalty = round((CSRD_RENEWABLE_HURDLE_PCT - renewable) * 0.5, 1)
            score -= penalty
            penalties.append(f"ESRS E1 Penalty: Renewable power {renewable:.1f}% is below 80% EU hurdle (-{penalty} pts)")
        supplier = m.get("supplier_signoff_pct", 0.0)
        if supplier < CSRD_SUPPLIER_HURDLE_PCT:
            score -= 10.0
            penalties.append(f"ESRS G1 Penalty: Supplier Code sign-off {supplier:.1f}% < 95% threshold (-10 pts)")
        score = round(max(0.0, min(100.0, score)), 1)
        return {"framework": "CSRD (ESRS Strict)", "score": score, "tier": _tier(score, CSRD_TIERS),
                "penalties": penalties, "focus": "Double Materiality, Value Chain Telemetry & 80% Clean Power Hurdle"}

    @staticmethod
    def _issb_score(base: float, m: dict[str, Any]) -> dict[str, Any]:
        score, penalties = base, []
        target = m.get("target_reduction_2030_pct")
        if target is None or target < ISSB_TARGET_HURDLE_PCT:
            score -= 15.0
            shown = "not disclosed" if target is None else f"{target:.1f}%"
            penalties.append(f"IFRS S2 Transition Risk: 2030 emissions reduction target {shown} is below 45% SBTi 1.5°C threshold (-15 pts)")
        if m.get("scope_3_pct", 0.0) > 65.0 and m.get("supplier_signoff_pct", 0.0) < 90.0:
            score -= 10.0
            penalties.append("IFRS S2 Supply Chain Risk: Unmitigated Scope 3 exposure > 65% with < 90% supplier coverage (-10 pts)")
        score = round(max(0.0, min(100.0, score)), 1)
        return {"framework": "ISSB (IFRS S2)", "score": score, "tier": _tier(score, ISSB_TIERS),
                "penalties": penalties, "focus": "Financial Capital Allocation, Transition Plans & Climate Value-at-Risk"}


def _resolve_input_path(raw_path: str) -> str:
    """Relative paths resolve against the working directory, then the script folder."""
    if os.path.isabs(raw_path) or os.path.exists(raw_path):
        return raw_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), raw_path)


def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on error, 2 if extraction was refused."""
    parser = argparse.ArgumentParser(description="EcoGranite-Advisor: Autonomous ESG Sustainability Auditor")
    parser.add_argument("--input", "-i", default="sample_esg_report.json",
                        help="Path to input ESG document (.json, .pdf, or .docx).")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_ID, help="IBM Granite foundation model ID.")
    args = parser.parse_args()

    load_environment()
    advisor = EcoGraniteAdvisor(model_id=args.model)
    try:
        data = advisor.load_document(_resolve_input_path(args.input))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    analysis = advisor.analyze_compliance(data)
    print(advisor.generate_audit_report(data, analysis))
    return 0 if analysis.get("can_audit") else 2


if __name__ == "__main__":
    sys.exit(main())
