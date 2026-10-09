"""
IBM Granite Reasoning Client
Supports both live API inference (via WatsonX AI) and a deterministic offline
Granite Rule-Based Reasoning Engine when API keys are not supplied.
Exposes engine runtime metadata, catches and reports sanitized API errors, and generates structured cards.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GraniteReasoningClient:
    def __init__(self, model_id: str = "ibm-granite/granite-3.0-8b-instruct"):
        self.model_id = model_id
        # Only WATSONX_APIKEY enables live mode; HUGGINGFACE_API_KEY is not a WatsonX credential
        self.api_key = os.getenv("WATSONX_APIKEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.is_live = bool(self.api_key and self.project_id)
        self.last_error: Optional[str] = None
        self.last_reason: Optional[str] = None

    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Provides runtime engine status and sanitized fallback reasons for transparency."""
        if self.is_live and not self.last_error:
            return {
                "engine": "watsonx",
                "label": "IBM WatsonX Live API",
                "model_id": self.model_id,
                "is_fallback": False,
                "fallback_reason": None
            }
        else:
            reason = self.last_reason or ("No WatsonX credentials supplied (WATSONX_APIKEY / WATSONX_PROJECT_ID)" if not self.is_live else "Inference error encountered")
            return {
                "engine": "local",
                "label": "Local IBM Granite 3.0 Reasoning Engine",
                "model_id": self.model_id,
                "is_fallback": True,
                "fallback_reason": reason,
                "error": self.last_error
            }

    def generate_esg_insights(self, audit_summary: Dict[str, Any]) -> str:
        """
        Sends structured audit findings to IBM Granite or uses local deterministic reasoning.
        Captures and reports API failures with sanitized exception types to prevent credential leakage.
        """
        if self.is_live:
            try:
                result = self._call_watsonx_granite(audit_summary)
                self.last_error = None
                self.last_reason = None
                return result
            except Exception as e:
                logger.exception("WatsonX Granite inference failed")
                self.last_error = f"{type(e).__name__}"
                self.last_reason = f"WatsonX API Exception: {type(e).__name__}"

        return self._local_granite_reasoning(audit_summary)

    def _call_watsonx_granite(self, audit_summary: Dict[str, Any]) -> str:
        """Calls IBM WatsonX Granite Foundation Model using sanitized metric payloads."""
        from ibm_watsonx_ai.foundation_models import Model  # type: ignore
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # type: ignore

        parameters = {
            GenParams.MAX_NEW_TOKENS: 500,
            GenParams.TEMPERATURE: 0.2,
            GenParams.TOP_P: 0.9,
        }

        # Send computed metrics only — omit raw user text to prevent prompt injection
        metrics_payload = {
            "metrics": audit_summary.get("metrics", {}),
            "pillars": audit_summary.get("pillars", {}),
            "benchmarks": {
                k: {"status": v.get("status"), "score": v.get("score"), "risk": v.get("risk")}
                for k, v in audit_summary.get("benchmarks", {}).items()
            },
            "data_quality": {
                "reconciliation_status": audit_summary.get("data_quality", {}).get("reconciliation_status"),
                "completeness_pct": audit_summary.get("data_quality", {}).get("completeness_pct")
            }
        }

        prompt = (
            f"You are an expert ESG sustainability auditor. Analyze the following verified corporate metrics "
            f"and synthesize 3 concise, prioritized decarbonization recommendations:\n"
            f"{json.dumps(metrics_payload, indent=2)}"
        )

        model = Model(
            model_id=self.model_id,
            params=parameters,
            credentials={"apikey": self.api_key, "url": "https://us-south.ml.cloud.ibm.com"},
            project_id=self.project_id
        )

        response = model.generate_text(prompt=prompt)
        return str(response)

    def generate_structured_recommendations(self, audit_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Derives structured decarbonization action items for UI card presentation.
        Uses deterministic rule-based prioritization over observed disclosures.
        Handles missing disclosures by emitting data-gap notice cards.
        """
        metrics = audit_summary.get("metrics", {})
        benchmarks = audit_summary.get("benchmarks", {})

        s3_pct = metrics.get("scope_3_pct")
        s1 = metrics.get("scope_1", 0.0)
        s2 = metrics.get("scope_2", 0.0)
        s3 = metrics.get("scope_3", 0.0)
        tot_ghg = metrics.get("total_ghg", 0.0)
        ren_val = metrics.get("renewable_pct")
        water_rec_pct = metrics.get("water_recycled_pct")

        items = []

        # Card 1: Value Chain / Operations
        if tot_ghg == 0.0 and s1 == 0.0 and s2 == 0.0 and s3 == 0.0:
            items.append({
                "pillar": "GHG Accounting Gap",
                "title": "Establish Comprehensive GHG Baseline",
                "metric": "Emissions data undisclosed",
                "action": "Implement primary activity-data capture across Scope 1, 2, and 15 Scope 3 categories per GHG Protocol Corporate Standard.",
                "priority": "Critical Priority",
                "benchmark_case": "GHG Protocol Corporate Standard Baseline Guidance",
                "badge_color": "#EF4444"
            })
        elif s3_pct is not None and s3_pct > 50.0:
            items.append({
                "pillar": "Supply Chain & Scope 3",
                "title": "Accelerate Scope 3 Supplier Engagement (Abengoa Protocol)",
                "metric": f"{s3_pct:.1f}% of total footprint ({s3:,.0f} MT CO2e)",
                "action": "Enforce mandatory third-party verified vendor carbon accounting (Abengoa model) and execute Category 1 raw material LCA disaggregation (BASF model).",
                "priority": "High Priority",
                "benchmark_case": "Abengoa (Mandatory Supplier Verification) & BASF (Raw Material LCAs)",
                "badge_color": "#EF4444"
            })
        else:
            pct_direct = (100.0 - s3_pct) if s3_pct is not None else 100.0
            items.append({
                "pillar": "Direct Operations",
                "title": "Accelerate Scope 1 & 2 Electrification",
                "metric": f"{pct_direct:.1f}% direct emissions ({s1 + s2:,.0f} MT CO2e)",
                "action": "Electrify commercial fleet and replace fossil heat with industrial heat pumps.",
                "priority": "High Priority",
                "benchmark_case": "National Grid (Capital Allocation Internalization)",
                "badge_color": "#EF4444"
            })

        # Card 2: Renewable Energy
        ren_status = benchmarks.get("Renewable Energy Share (Target >= 60%)", {})
        if ren_val is None or "Missing" in ren_status.get("status", ""):
            items.append({
                "pillar": "Clean Power & Transition",
                "title": "Disclose Renewable Energy Procurement",
                "metric": "Renewable electricity data missing",
                "action": "Audit electricity contracts, track Energy Attribute Certificates (EACs/RECs), and formulate a transition roadmap toward RE100 criteria.",
                "priority": "High Priority",
                "benchmark_case": "RE100 Renewable Electricity Standard",
                "badge_color": "#F59E0B"
            })
        elif ren_status.get("status") != "Compliant":
            items.append({
                "pillar": "Clean Power & Transition",
                "title": "Close Renewable Electricity Deficit (SC Johnson CapEx Matrix)",
                "metric": f"Currently {ren_val:.1f}% (target: >=60.0%)",
                "action": "Execute long-term Virtual Power Purchase Agreements (VPPAs) and prioritize low-to-mid CapEx on-site solar storage.",
                "priority": "High Priority",
                "benchmark_case": "SC Johnson (CapEx vs. Impact Decarbonization Matrix)",
                "badge_color": "#F59E0B"
            })
        else:
            items.append({
                "pillar": "Clean Power Leadership",
                "title": "Advance 24/7 Carbon-Free Energy Matching",
                "metric": f"Exceeds RE threshold at {ren_val:.1f}%",
                "action": "Transition from annual RECs to hourly matching and battery storage orchestration.",
                "priority": "Medium Priority",
                "benchmark_case": "RE100 Technical Framework",
                "badge_color": "#10B981"
            })

        # Card 3: Circular Economy & Product Efficiency
        if water_rec_pct is None or water_rec_pct == 0.0:
            items.append({
                "pillar": "Water & Circularity",
                "title": "Implement Closed-Loop Water Recovery",
                "metric": "Water recycling data missing / nascent",
                "action": "Deploy advanced wastewater treatment, metering, and closed-loop effluent recycling systems to align with zero liquid discharge benchmarks.",
                "priority": "Medium Priority",
                "benchmark_case": "Closed-Loop Circular Effluent Recovery",
                "badge_color": "#3B82F6"
            })
        elif water_rec_pct < 50.0:
            items.append({
                "pillar": "Water & Circularity",
                "title": "Water Recycling & Closed-Loop Recovery",
                "metric": f"Current wastewater recycling at {water_rec_pct:.1f}% (below 50% target)",
                "action": "Scale closed-loop recovery technology and membrane bioreactors to surpass 50% circularity.",
                "priority": "Medium Priority",
                "benchmark_case": "Closed-Loop Circular Effluent Recovery",
                "badge_color": "#3B82F6"
            })
        else:
            items.append({
                "pillar": "Product & Downstream Efficiency",
                "title": "Downstream Energy Transformation (IKEA Model)",
                "metric": f"Strong water recycling at {water_rec_pct:.1f}%",
                "action": "Drive scale reductions in Category 11 use-phase product energy efficiency targeting +50% efficiency (IKEA model).",
                "priority": "Medium Priority",
                "benchmark_case": "IKEA (Category 11 Product Efficiency Scaling)",
                "badge_color": "#3B82F6"
            })

        return items

    def get_eight_step_roadmap(self) -> List[Dict[str, Any]]:
        """Provides the standardized EcoGranite 8-step enterprise decarbonization workflow."""
        return [
            {"step": 1, "title": "Define Business Goals & Align Stakeholders", "action": "Secure board-level sponsorship and link executive variable compensation to SBTi Scope 3 goals."},
            {"step": 2, "title": "Select Consolidation Boundary & Core Principles", "action": "Adopt uniform Operational Control, Financial Control, or Equity Share across operating assets."},
            {"step": 3, "title": "Map Value Chain & Screen 15 Category Hot Spots", "action": "Screen enterprise activities across all 15 Scope 3 categories to pinpoint high-impact nodes."},
            {"step": 4, "title": "Enforce Minimum Category Boundaries & Exclusions", "action": "Apply cradle-to-gate accounting (no capital goods amortization per Box 5.4) and justify exclusions."},
            {"step": 5, "title": "Collect Primary Data & Execute Allocation Formulas", "action": "Deploy supplier calculation templates; apply standardized T&D loss and investment allocation formulas."},
            {"step": 6, "title": "Establish Base Year & Recalculation Policies", "action": "Publish quantitative targets and define formal recalculation triggers for structural M&A changes."},
            {"step": 7, "title": "Deploy Supplier Engagement (Abengoa / SC Johnson)", "action": "Enforce 3rd-party verified vendor accounting via Code of Conduct; prioritize CapEx vs impact matrix."},
            {"step": 8, "title": "Public Disclosure & Secure Third-Party Assurance", "action": "Disclose 15 disaggregated categories under EcoGranite to unlock sustainability-linked debt financing."}
        ]

    def _local_granite_reasoning(self, audit_summary: Dict[str, Any]) -> str:
        """
        Deterministic reasoning following IBM Granite 3.0 prompting guidelines.
        Derives all figures dynamically from the supplied audit summary.
        """
        recommendations = []
        metrics = audit_summary.get("metrics", {})
        benchmarks = audit_summary.get("benchmarks", {})

        s3_pct = metrics.get("scope_3_pct")
        s1 = metrics.get("scope_1", 0.0)
        s2 = metrics.get("scope_2", 0.0)
        s3 = metrics.get("scope_3", 0.0)
        tot_ghg = metrics.get("total_ghg", 0.0)

        # Recommendation 1: Scope 3 Value Chain
        if tot_ghg == 0.0:
            recommendations.append(
                "1. Establish Complete Emissions Inventory: Emissions baseline is unverified. "
                "Engage internal operations and supply chain teams to quantify Scopes 1-3."
            )
        elif s3_pct is not None and s3_pct > 50.0:
            recommendations.append(
                f"1. Target Value Chain Decarbonization: Scope 3 represents {s3_pct:.1f}% of total footprint ({s3:,.1f} MT CO2e). "
                f"Deploy supplier carbon scorecards and transport route optimization."
            )
        else:
            pct_direct = (100.0 - s3_pct) if s3_pct is not None else 100.0
            recommendations.append(
                f"1. Operational Abatement: Direct Scope 1+2 operations drive {pct_direct:.1f}% of emissions ({s1 + s2:,.1f} MT CO2e). "
                f"Transition vehicle fleets and on-site heating to electric/heat pumps."
            )

        # Recommendation 2: Renewable Energy Transition
        ren_status = benchmarks.get("Renewable Energy Share (Target >= 60%)", {})
        ren_val = metrics.get("renewable_pct")
        if ren_val is None or "Missing" in ren_status.get("status", ""):
            recommendations.append(
                "2. Benchmark Clean Power Consumption: Renewable electricity share is undisclosed. "
                "Establish baseline tracking across all operational facilities."
            )
        elif ren_status.get("status") == "Compliant":
            recommendations.append(
                f"2. Maintain Clean Energy Momentum: Exceeded renewable target at {ren_val:.1f}%. "
                f"Aim for 24/7 carbon-free hourly matching."
            )
        else:
            recommendations.append(
                f"2. Accelerate Clean Energy Procurement: Currently at {ren_val:.1f}%, falling short of the 60% RE threshold. "
                f"Execute long-term Virtual Power Purchase Agreements (VPPAs)."
            )

        # Recommendation 3: Water Recycling / Resource Efficiency
        water_rec_pct = metrics.get("water_recycled_pct")
        if water_rec_pct is None or water_rec_pct == 0.0:
            recommendations.append(
                "3. Elevate Resource Circularity: Water recycling baseline is absent. "
                "Institute water audit protocols to benchmark circular recovery potential."
            )
        elif water_rec_pct < 50.0:
            recommendations.append(
                f"3. Elevate Water Circularity: Water recycling is currently at {water_rec_pct:.1f}% (below 50% circularity goal). "
                f"Invest in closed-loop effluent treatment and membrane filtration."
            )
        else:
            recommendations.append(
                f"3. Waste & Landfill Minimization: High water circularity at {water_rec_pct:.1f}%. "
                f"Focus on residual hazardous waste diversion."
            )

        header = "STRATEGIC RECOMMENDATIONS (IBM Granite 3.0 Reasoning):\n"
        if self.last_error:
            header = f"[Notice: WatsonX API offline ({self.last_reason}) — using local deterministic engine]\n" + header

        return header + "\n".join(f"   {rec}" for rec in recommendations)
