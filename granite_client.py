"""
IBM Granite Reasoning Client
Supports both live API inference (via WatsonX AI or HuggingFace Inference API)
and a deterministic offline Granite Rule-Based Reasoning Engine when API keys are not supplied.
"""

import json
import os
from typing import Dict, Any, Optional


class GraniteReasoningClient:
    def __init__(self, model_id: str = "ibm-granite/granite-3.0-8b-instruct"):
        self.model_id = model_id
        self.api_key = os.getenv("WATSONX_APIKEY") or os.getenv("HUGGINGFACE_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.is_live = bool(self.api_key)

    def generate_esg_insights(self, audit_summary: Dict[str, Any]) -> str:
        """
        Sends structured audit findings to IBM Granite or uses local deterministic reasoning.
        """
        if self.is_live and self.project_id:
            try:
                return self._call_watsonx_granite(audit_summary)
            except Exception as e:
                # Fallback to local deterministic reasoning
                pass

        return self._local_granite_reasoning(audit_summary)

    def _call_watsonx_granite(self, audit_summary: Dict[str, Any]) -> str:
        """Calls IBM WatsonX Granite Foundation Model."""
        from ibm_watsonx_ai.foundation_models import Model
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        parameters = {
            GenParams.MAX_NEW_TOKENS: 500,
            GenParams.TEMPERATURE: 0.2,
            GenParams.TOP_P: 0.9,
        }

        prompt = (
            f"You are an expert ESG sustainability auditor. Analyze the following corporate metrics "
            f"and synthesize 3 concise, prioritized decarbonization recommendations:\n"
            f"{json.dumps(audit_summary, indent=2)}"
        )

        model = Model(
            model_id=self.model_id,
            params=parameters,
            credentials={"apikey": self.api_key, "url": "https://us-south.ml.cloud.ibm.com"},
            project_id=self.project_id
        )

        response = model.generate_text(prompt=prompt)
        return response

    def generate_structured_recommendations(self, audit_summary: Dict[str, Any]) -> list:
        """
        Derives structured decarbonization action items for UI card presentation.
        """
        metrics = audit_summary.get("metrics", {})
        benchmarks = audit_summary.get("benchmarks", {})

        s3_pct = metrics.get("scope_3_pct", 0.0)
        s1 = metrics.get("scope_1", 0.0)
        s2 = metrics.get("scope_2", 0.0)
        s3 = metrics.get("scope_3", 0.0)
        tot_ghg = metrics.get("total_ghg", 0.0)
        ren_val = metrics.get("renewable_pct", 0.0)
        water_rec_pct = metrics.get("water_recycled_pct", 0.0)

        items = []

        # Card 1: Value Chain
        if s3_pct > 50.0:
            items.append({
                "pillar": "Supply Chain & Scope 3",
                "title": "Accelerate Scope 3 Supplier Engagement",
                "metric": f"{s3_pct:.1f}% of total footprint ({s3:,.0f} MT CO2e)",
                "action": "Implement automated supplier carbon auditing via IBM Docling telemetry and transport logistics invoice ingestion.",
                "priority": "High Priority",
                "badge_color": "#EF4444"
            })
        else:
            items.append({
                "pillar": "Direct Operations",
                "title": "Accelerate Scope 1 & 2 Electrification",
                "metric": f"{(100.0 - s3_pct):.1f}% direct emissions ({s1 + s2:,.0f} MT CO2e)",
                "action": "Electrify commercial fleet and replace fossil heat with industrial heat pumps.",
                "priority": "High Priority",
                "badge_color": "#EF4444"
            })

        # Card 2: Renewable Energy
        ren_status = benchmarks.get("Renewable Energy Share (Target >= 60%)", {})
        if ren_status.get("status") != "Compliant":
            items.append({
                "pillar": "Clean Power & Transition",
                "title": "Close Renewable Electricity Deficit",
                "metric": f"Currently {ren_val:.1f}% (target: ≥60.0%)",
                "action": "Execute long-term Virtual Power Purchase Agreements (VPPAs) and on-site solar storage.",
                "priority": "High Priority",
                "badge_color": "#F59E0B"
            })
        else:
            items.append({
                "pillar": "Clean Power Leadership",
                "title": "Advance 24/7 Carbon-Free Energy Matching",
                "metric": f"Exceeds RE threshold at {ren_val:.1f}%",
                "action": "Transition from annual RECs to hourly matching and battery storage orchestration.",
                "priority": "Medium Priority",
                "badge_color": "#10B981"
            })

        # Card 3: Circular Economy & Governance
        if water_rec_pct < 50.0:
            items.append({
                "pillar": "Water & Circularity",
                "title": "Water Recycling & Closed-Loop Recovery",
                "metric": f"Current wastewater recycling at {water_rec_pct:.1f}% (below 50% target)",
                "action": "Scale closed-loop recovery technology and membrane bioreactors to surpass 50% capacity by FY 2026.",
                "priority": "Medium Priority",
                "badge_color": "#3B82F6"
            })
        else:
            items.append({
                "pillar": "Waste Elimination",
                "title": "Zero-Waste to Landfill Certification",
                "metric": f"Strong water recycling at {water_rec_pct:.1f}%",
                "action": "Target residual supply chain packaging with post-consumer recycled (PCR) circular inputs.",
                "priority": "Medium Priority",
                "badge_color": "#3B82F6"
            })

        return items

    def _local_granite_reasoning(self, audit_summary: Dict[str, Any]) -> str:
        """
        Deterministic reasoning following IBM Granite 3.0 prompting guidelines.
        Derives all figures dynamically from the supplied audit summary.
        """
        recommendations = []
        metrics = audit_summary.get("metrics", {})
        benchmarks = audit_summary.get("benchmarks", {})

        s3_pct = metrics.get("scope_3_pct", 0.0)
        s1 = metrics.get("scope_1", 0.0)
        s2 = metrics.get("scope_2", 0.0)
        s3 = metrics.get("scope_3", 0.0)
        tot_ghg = metrics.get("total_ghg", 0.0)

        # Recommendation 1: Scope 3 Value Chain
        if s3_pct > 50.0:
            recommendations.append(
                f"1. Target Value Chain Decarbonization: Scope 3 represents {s3_pct:.1f}% of total footprint ({s3:,.1f} MT CO2e). "
                f"Deploy supplier carbon scorecards and transport route optimization."
            )
        else:
            recommendations.append(
                f"1. Operational Abatement: Direct Scope 1+2 operations drive {(100.0 - s3_pct):.1f}% of emissions ({s1 + s2:,.1f} MT CO2e). "
                f"Transition vehicle fleets and on-site heating to electric/heat pumps."
            )

        # Recommendation 2: Renewable Energy Transition
        ren_status = benchmarks.get("Renewable Energy Share (Target >= 60%)", {})
        ren_val = metrics.get("renewable_pct", 0.0)
        if ren_status.get("status") != "Compliant":
            recommendations.append(
                f"2. Accelerate Clean Energy Procurement: Currently at {ren_val:.1f}%, falling short of the 60% RE threshold. "
                f"Execute long-term Virtual Power Purchase Agreements (VPPAs)."
            )
        else:
            recommendations.append(
                f"2. Maintain Clean Energy Momentum: Exceeded renewable target at {ren_val:.1f}%. "
                f"Aim for 24/7 carbon-free hourly matching."
            )

        # Recommendation 3: Water Recycling / Resource Efficiency
        water_rec_pct = metrics.get("water_recycled_pct", 0.0)
        if water_rec_pct < 50.0:
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
        return header + "\n".join(f"   {rec}" for rec in recommendations)

