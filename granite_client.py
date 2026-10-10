"""
IBM Granite Reasoning Client
Supports live inference via IBM WatsonX AI and Hugging Face Inference API,
with a deterministic offline rule-based reasoning engine as a seamless fallback.
Exposes engine runtime metadata, catches and reports API errors, and generates structured cards.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


HF_MAX_TOKENS = 1000  # reasoning models spend part of the budget on thinking
WATSONX_MAX_TOKENS = 500
REQUEST_TIMEOUT_S = 20
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_FALLBACK_MODELS = ("ibm-granite/granite-4.2-8b", "ibm-granite/granite-4.2-3b")
MAX_RETRIES = 2
INITIAL_BACKOFF_S = 0.2


def load_environment() -> None:
    """Loads a local .env file (never overriding real environment variables).

    Call once from an entry point (app.py / advisor_engine.main), not at import
    time, so importing this module has no side effects and tests stay
    deterministic.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env loading")
        return
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path, override=False)


def build_prompt(audit_summary: dict[str, Any]) -> str:
    """Builds the LLM prompt from computed metrics only.

    Free text from uploaded files (e.g. company name) is excluded so it cannot
    inject instructions into the prompt.
    """
    safe_summary = {
        "rating_tier": audit_summary.get("rating_tier"),
        "metrics": audit_summary.get("metrics", {}),
        "benchmarks": audit_summary.get("benchmarks", {}),
    }
    return (
        "You are an expert ESG sustainability auditor. Analyze the following corporate metrics "
        "and synthesize 3 concise, prioritized decarbonization recommendations:\n"
        f"{json.dumps(safe_summary, indent=2)}"
    )


class GraniteReasoningClient:
    def __init__(self, model_id: str = "ibm-granite/granite-3.0-8b-instruct", hf_api_key: Optional[str] = None) -> None:
        self.model_id = model_id
        # WatsonX credentials
        self.api_key = os.getenv("WATSONX_APIKEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.service_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.is_watsonx_live = bool(self.api_key and self.project_id)

        # Hugging Face Inference credentials (explicit parameter takes precedence over env)
        raw_key = (hf_api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN") or "").strip()
        self.hf_key = raw_key or None
        self.is_hf_live = bool(self.hf_key)

        # is_live corresponds specifically to WatsonX live mode per review specifications
        self.is_live = self.is_watsonx_live

        self.last_error: Optional[str] = None
        self.last_reason: Optional[str] = None

    def get_runtime_metadata(self) -> dict[str, Any]:
        """Provides runtime engine status and fallback reasons for transparency."""
        if self.is_watsonx_live and not self.last_error:
            return {
                "engine": "watsonx",
                "label": "IBM WatsonX Live API",
                "model_id": self.model_id,
                "is_fallback": False,
                "fallback_reason": None
            }
        elif self.is_hf_live and not self.last_error:
            return {
                "engine": "huggingface",
                "label": f"Hugging Face Inference API ({self.model_id})",
                "model_id": self.model_id,
                "is_fallback": False,
                "fallback_reason": None
            }
        else:
            reason = self.last_reason or (
                "Local rule-based reasoning active"
                if not (self.is_watsonx_live or self.is_hf_live)
                else "Inference error encountered (fallback active)"
            )
            return {
                "engine": "local",
                "label": "Local IBM Granite 3.0 Reasoning Engine",
                "model_id": self.model_id,
                "is_fallback": True,
                "fallback_reason": reason,
                "error": self.last_error
            }

    def generate_esg_insights(self, audit_summary: dict[str, Any]) -> str:
        """
        Sends structured audit findings to IBM WatsonX, Hugging Face, or uses local deterministic reasoning.
        Captures and reports API failures without silently crashing.
        """
        # Provider 1: IBM WatsonX
        if self.is_watsonx_live:
            try:
                result = self._call_watsonx_granite(audit_summary)
                self.last_error = None
                self.last_reason = None
                return result
            except Exception as e:
                logger.exception("WatsonX Granite inference failed")
                self.last_error = type(e).__name__
                self.last_reason = f"WatsonX API Exception: {type(e).__name__} (see server log)"

        # Provider 2: Hugging Face Inference API
        elif self.is_hf_live:
            try:
                result = self._call_huggingface_granite(audit_summary)
                self.last_error = None
                self.last_reason = None
                return result
            except Exception as e:
                logger.exception("Hugging Face Granite inference failed")
                self.last_error = type(e).__name__
                msg = str(e)
                if msg.startswith("Hugging Face API"):
                    self.last_reason = msg
                else:
                    self.last_reason = f"Hugging Face API Exception: {type(e).__name__} (see server log)"

        # Fallback Provider: Local Rule-Based Engine
        return self._local_granite_reasoning(audit_summary)

    def _call_watsonx_granite(self, audit_summary: dict[str, Any]) -> str:
        """Calls IBM WatsonX Granite Foundation Model."""
        from ibm_watsonx_ai.foundation_models import Model
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        parameters = {
            GenParams.MAX_NEW_TOKENS: WATSONX_MAX_TOKENS,
            GenParams.TEMPERATURE: 0.2,
            GenParams.TOP_P: 0.9,
        }

        prompt = build_prompt(audit_summary)

        model = Model(
            model_id=self.model_id,
            params=parameters,
            credentials={"apikey": self.api_key, "url": self.service_url},
            project_id=self.project_id
        )

        return str(model.generate_text(prompt=prompt))

    def _call_huggingface_granite(self, audit_summary: dict[str, Any]) -> str:
        """Calls the Hugging Face Inference Router (OpenAI-compatible chat API).

        Tries the selected model first, then the Granite 4.2 fallbacks. Raises
        RuntimeError with a safe, token-free message on failure.
        """
        import requests  # type: ignore[import]

        headers = {"Authorization": f"Bearer {self.hf_key}", "Content-Type": "application/json"}
        messages = [
            {"role": "system", "content": "Synthesize concise, prioritized decarbonization recommendations."},
            {"role": "user", "content": build_prompt(audit_summary)},
        ]
        candidates = [self.model_id] + [m for m in HF_FALLBACK_MODELS if m != self.model_id]
        last_status = "Unknown"
        for model_name in candidates:
            payload = {"model": model_name, "messages": messages,
                       "max_tokens": HF_MAX_TOKENS, "temperature": 0.2}
            response = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S)
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                        time.sleep(INITIAL_BACKOFF_S * (2 ** attempt))
                        continue
                    break
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    if attempt < MAX_RETRIES:
                        time.sleep(INITIAL_BACKOFF_S * (2 ** attempt))
                        continue
                    raise
            if response is None:
                last_status = "No response"
                continue
            if response.status_code == 200:
                text = self._extract_chat_text(response.json())
                if text:
                    return text
                last_status = "empty response"
            elif response.status_code == 402:
                raise RuntimeError("Hugging Face API (402): Account has no inference credits. Fallback active.")
            elif response.status_code in (401, 403):
                raise RuntimeError("Hugging Face API (403): Token missing Inference Provider permissions. Fallback active.")
            else:
                last_status = f"status {response.status_code}"
        raise RuntimeError(f"Hugging Face API call failed ({last_status})")

    @staticmethod
    def _extract_chat_text(body: Any) -> str:
        """Returns the assistant text from a chat-completions body, or '' if absent."""
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""
        return content.strip() if isinstance(content, str) else ""

    def generate_structured_recommendations(self, audit_summary: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Derives structured decarbonization action items for UI card presentation.
        Uses deterministic rule-based prioritization over observed disclosures.
        """
        metrics = audit_summary.get("metrics", {})
        benchmarks = audit_summary.get("benchmarks", {})

        s3_pct = metrics.get("scope_3_pct", 0.0)
        s1 = metrics.get("scope_1", 0.0)
        s2 = metrics.get("scope_2", 0.0)
        s3 = metrics.get("scope_3", 0.0)
        ren_val = metrics.get("renewable_pct", 0.0)
        water_rec_pct = metrics.get("water_recycled_pct")

        items = []

        # Card 1: Value Chain / Operations
        if metrics.get("total_ghg", 0.0) <= 0:
            items.append({
                "pillar": "Data Quality",
                "title": "Disclose Scope 1-3 Emissions",
                "metric": "Total GHG footprint not available",
                "action": "Report Scope 1, 2 and 3 emissions to enable prioritization.",
                "priority": "High Priority",
                "benchmark_case": "GHG Protocol Corporate Standard",
                "badge_color": "#EF4444"
            })
        elif s3_pct > 50.0:
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
            items.append({
                "pillar": "Direct Operations",
                "title": "Accelerate Scope 1 & 2 Electrification",
                "metric": f"{(100.0 - s3_pct):.1f}% direct emissions ({s1 + s2:,.0f} MT CO2e)",
                "action": "Electrify commercial fleet and replace fossil heat with industrial heat pumps.",
                "priority": "High Priority",
                "benchmark_case": "National Grid (Capital Allocation Internalization)",
                "badge_color": "#EF4444"
            })

        # Card 2: Renewable Energy
        ren_status = benchmarks.get("Renewable Energy Share (Target >= 60%)", {})
        if str(ren_status.get("status", "")).startswith("Unknown"):
            items.append({
                "pillar": "Data Quality",
                "title": "Disclose Renewable Electricity Share",
                "metric": "Renewable share not disclosed",
                "action": "Report total and renewable MWh so the transition gap can be measured.",
                "priority": "High Priority",
                "benchmark_case": "RE100 Technical Framework",
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
        if water_rec_pct is None:
            items.append({
                "pillar": "Water & Circularity",
                "title": "Disclose Water Recycling & Conservation",
                "metric": "Water recycling not reported",
                "action": "Disclose campus and facility water recycling metrics to establish closed-loop circularity.",
                "priority": "Medium Priority",
                "benchmark_case": "Closed-Loop Circular Effluent Recovery",
                "badge_color": "#3B82F6"
            })
        elif water_rec_pct < 50.0:
            items.append({
                "pillar": "Water & Circularity",
                "title": "Water Recycling & Closed-Loop Recovery",
                "metric": f"Current wastewater recycling at {water_rec_pct:.1f}% (below 50% target)",
                "action": "Scale closed-loop recovery technology and membrane bioreactors to surpass 50% circularity within the next reporting cycle.",
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

    def get_eight_step_roadmap(self) -> list[dict[str, Any]]:
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

    def _local_granite_reasoning(self, audit_summary: dict[str, Any]) -> str:
        """
        Deterministic reasoning following IBM Granite 3.0 prompting guidelines.
        Derives all figures dynamically from the supplied audit summary.
        """
        cards = self.generate_structured_recommendations(audit_summary)
        recommendations = [
            f"{i}. {card['title']}: {card['metric']}. {card['action']}"
            for i, card in enumerate(cards, 1)
        ]

        header = "STRATEGIC RECOMMENDATIONS (IBM Granite 3.0 Reasoning):\n"
        if self.last_error:
            header = f"[Notice: Inference API offline ({self.last_reason}) — using local deterministic engine]\n" + header

        return header + "\n".join(f"   {rec}" for rec in recommendations)
