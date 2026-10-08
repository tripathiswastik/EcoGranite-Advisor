"""
EcoGranite-Advisor: High-Fidelity ESG Compliance & Decarbonization Dashboard
Autonomous Corporate Sustainability & Decarbonization Audit Engine.
Powered by IBM Granite 3.0 Reasoning & IBM Docling Document Parsing.
"""

import json
import logging
import os
import sys
from typing import Dict, Any

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError(
        f"Required UI packages missing ({e}). Run: pip install streamlit pandas plotly"
    )

from advisor_engine import EcoGraniteAdvisor
from esg_parser import parse_document, DOCLING_AVAILABLE

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="EcoGranite-Advisor | ESG Sustainability Auditor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# High-End Design System & Custom CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }

    /* Gradient Header Typography */
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.7px;
        background: linear-gradient(135deg, #0F172A 0%, #1E40AF 50%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        display: inline-block;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        color: #475569;
        margin-bottom: 1.4rem;
        line-height: 1.5;
    }

    /* Engine Status Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        vertical-align: middle;
        margin-left: 8px;
    }

    .badge-granite {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: #FFFFFF;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.35);
    }

    .badge-refusal {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: #FFFFFF;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.35);
    }

    /* Luxury Glassmorphic KPI Cards */
    .kpi-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem 1.1rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .kpi-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -4px rgba(15, 23, 42, 0.09), 0 4px 10px -2px rgba(15, 23, 42, 0.04);
        border-color: #CBD5E1;
    }

    .kpi-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .kpi-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .kpi-icon {
        font-size: 1.25rem;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: #F1F5F9;
    }

    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin-bottom: 0.4rem;
    }

    .kpi-sub {
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 6px;
    }

    .kpi-sub-positive {
        background: #ECFDF5;
        color: #059669;
    }

    .kpi-sub-negative {
        background: #FEF2F2;
        color: #DC2626;
    }

    .kpi-sub-neutral {
        background: #F8FAFC;
        color: #475569;
    }

    /* Recommendation Cards */
    .rec-card-modern {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease;
    }

    .rec-card-modern:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
    }

    .rec-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }

    .rec-badge-pillar {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 3px 8px;
        border-radius: 6px;
    }

    .rec-card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.12rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.35rem;
    }

    .rec-meta-box {
        font-size: 0.88rem;
        color: #334155;
        background: #F8FAFC;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border: 1px solid #F1F5F9;
    }

    .rec-action-text {
        font-size: 0.93rem;
        color: #1E293B;
        line-height: 1.5;
    }

    /* Stepper Workflow */
    .step-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }

    .step-num-pill {
        background: linear-gradient(135deg, #1E40AF, #3B82F6);
        color: white;
        font-weight: 800;
        font-size: 0.85rem;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)


def get_advisor(model_id: str) -> EcoGraniteAdvisor:
    """Instantiates the EcoGraniteAdvisor instance."""
    return EcoGraniteAdvisor(model_id=model_id)


def load_dataset_file(filename: str) -> Dict[str, Any]:
    """Helper to load a JSON dataset from disk."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"company_name": "Sample Entity", "status": "extraction_failed", "errors": ["File not found"]}


# ==========================================
# Sidebar: Controls & Document Ingestion
# ==========================================
with st.sidebar:
    st.markdown("### 🌍 EcoGranite Engine")
    st.caption("IBM Granite 3.0 + IBM Docling v2")
    st.markdown("---")

    # Foundation Model Selector
    model_choice = st.selectbox(
        "IBM Granite Foundation Model",
        options=[
            "ibm-granite/granite-3.0-8b-instruct",
            "ibm-granite/granite-3.0-2b-instruct",
            "ibm-granite/granite-guardian-3.0-8b"
        ],
        index=0,
        help="Select foundation model for automated synthesis and compliance reasoning."
    )

    advisor = get_advisor(model_choice)
    runtime_meta = advisor.granite_client.get_runtime_metadata()

    # Ingestion Source Selector
    st.markdown("#### 📂 Disclosure Data Source")
    data_source_mode = st.radio(
        "Select Disclosure Ingest Mode",
        options=[
            "📗 Sample: EcoGlobal Enterprise (Good/Compliant)",
            "🇮🇳 India: Infosys Limited (SEBI BRSR Core Mandate)",
            "🏢 Benchmark: Siemens AG (Global Enterprise - DEGREE)",
            "📙 Sample: CarbonHeavy Corp (Lagging/Poor)",
            "📕 Sample: Incoherent Disclosures (Reconciliation Failure)",
            "📁 Upload Custom File (.json, .pdf, .docx)"
        ],
        index=0
    )

    raw_data = None
    if data_source_mode == "📗 Sample: EcoGlobal Enterprise (Good/Compliant)":
        raw_data = load_dataset_file("sample_esg_report.json")
    elif data_source_mode == "🇮🇳 India: Infosys Limited (SEBI BRSR Core Mandate)":
        raw_data = load_dataset_file("sample_esg_report_india.json")
    elif data_source_mode == "🏢 Benchmark: Siemens AG (Global Enterprise - DEGREE)":
        raw_data = load_dataset_file("sample_esg_report_siemens.json")
    elif data_source_mode == "📙 Sample: CarbonHeavy Corp (Lagging/Poor)":
        raw_data = load_dataset_file("sample_esg_report_poor.json")
    elif data_source_mode == "📕 Sample: Incoherent Disclosures (Reconciliation Failure)":
        raw_data = load_dataset_file("sample_esg_report_invalid.json")
    else:
        uploaded_file = st.file_uploader(
            "Upload Corporate Disclosure (.json, .pdf, .docx)",
            type=["json", "pdf", "docx"],
            help="IBM Docling will parse multi-modal tables and text into the ESG auditing schema."
        )
        if uploaded_file is not None:
            try:
                with st.spinner(f"Ingesting {uploaded_file.name} via Docling Pipeline..."):
                    raw_data = advisor.load_document(uploaded_file)
                st.success(f"Parsed: {uploaded_file.name}")
            except Exception as err:
                st.error(f"Error parsing file: {err}")
                raw_data = {"status": "extraction_failed", "company_name": uploaded_file.name, "errors": [str(err)]}
        else:
            raw_data = load_dataset_file("sample_esg_report.json")

    # SASB SICS Sector Materiality Profile Selector
    st.markdown("#### 📐 SASB SICS Materiality Profile")
    comp_str = str(raw_data.get("company_name", "") if isinstance(raw_data, dict) else "")
    default_sec_idx = 0 if "Infosys" in comp_str else (1 if ("CarbonHeavy" in comp_str or "Siemens" in comp_str) else 3)
    sector_choice = st.selectbox(
        "Industry Weighting Archetype",
        options=[
            "Technology & Software",
            "Heavy Industry & Metals",
            "Financial Institutions",
            "General Enterprise"
        ],
        index=default_sec_idx,
        help="Applies SASB SICS / ISSB IFRS S2 industry-specific materiality weights dynamically."
    )

    # Sidebar Engine Diagnostics
    st.markdown("---")
    st.markdown("#### ⚙️ Engine Diagnostics")
    st.caption(f"Docling Bridge: **{'🟢 Native' if DOCLING_AVAILABLE else '🟡 Structural Parser'}**")
    if runtime_meta.get("is_fallback"):
        st.info(f"Granite Mode: **Local Reasoning**\n\n*{runtime_meta.get('fallback_reason')}*")
    else:
        st.success("Granite Mode: **IBM WatsonX Live API Connected**")


# ==========================================
# Pre-Audit Extraction Validation
# ==========================================
analysis = advisor.analyze_compliance(raw_data)

if not analysis.get("can_audit", True):
    # REFUSAL STATE: Refuse to invent fake numbers!
    st.markdown(
        f"<div class='hero-title'>EcoGranite-Advisor</div>"
        f"<span class='badge-pill badge-refusal'>Audit Refused</span>",
        unsafe_allow_html=True
    )
    st.error("🚨 **Pre-Audit Extraction Refusal**: The ingested document does not contain readable statutory ESG disclosures.")
    
    st.markdown(f"**Entity Reference**: `{analysis.get('company_name', 'Unknown')}`")
    st.markdown("### Missing / Failed Disclosure Fields:")
    for err in analysis.get("errors", []):
        st.markdown(f"- ❌ **{err}**")

    st.info(
        "💡 **Data Integrity Guarantee**: EcoGranite refuses to fabricate or substitute default metrics. "
        "Please provide a document containing verified Scope 1/2 GHG emissions and renewable energy disclosures."
    )
    st.stop()


# ==========================================
# Successful Audit Display
# ==========================================
company_name = analysis.get("company_name", "Unknown Entity")
reporting_year = analysis.get("reporting_year", 2024)
standards_list = analysis.get("standards", ["GRI", "TCFD", "SASB"])
country = analysis.get("country", "")
jurisdiction = analysis.get("jurisdiction", "")
m = analysis["metrics"]
benchmarks = analysis["benchmarks"]
pillars = analysis.get("pillars", {})
score = analysis["esg_readiness_score"]
rating_tier = analysis["rating_tier"]
dq = analysis.get("data_quality", {})

# Header
st.markdown(
    f"<div class='hero-title'>EcoGranite-Advisor: ESG Sustainability Auditor</div>"
    f"<span class='badge-pill badge-granite'>Granite 3.0 Reasoning</span>",
    unsafe_allow_html=True
)
country_flag = "🇮🇳 " if "India" in country else ("🇩🇪 " if "Germany" in country else "")
country_str = f" | {country_flag}{country}" if country else ""
st.markdown(
    f"<div class='hero-subtitle'>Corporate Sustainability & Decarbonization Audit Engine{country_str} | "
    f"<b>{company_name}</b> (FY {reporting_year}) — Frameworks: {', '.join(standards_list)}</div>",
    unsafe_allow_html=True
)

# Reconciliation Failure Banner
recon_status = dq.get("reconciliation_status", "UNKNOWN")
if recon_status != "PASSED":
    st.error(
        f"🚨 **Data Reconciliation Failure**: Scope 1+2+3 sum ({dq.get('calculated_total_ghg', 0):,.1f} MT) "
        f"does not reconcile with reported total GHG ({dq.get('reported_total_ghg', 0):,.1f} MT). "
        f"Variance: **{dq.get('total_ghg_variance', 0):,.1f} MT**. Rating disqualified from leader tier."
    )
elif dq.get("validation_warnings"):
    st.warning(f"⚠️ **Data Quality Notice**: {len(dq['validation_warnings'])} disclosure warning(s) detected. See Data Quality diagnostics.")

# ==========================================
# Luxury Glassmorphic KPI Row
# ==========================================
k1, k2, k3, k4 = st.columns(4)

with k1:
    is_leader = "EXCELLENT" in rating_tier or "GOOD" in rating_tier
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-top">
            <span class="kpi-label">Readiness Score</span>
            <span class="kpi-icon">🎯</span>
        </div>
        <div class="kpi-value">{score:.1f}<span style="font-size:1.1rem; color:#94A3B8;"> / 100</span></div>
        <span class="kpi-sub {'kpi-sub-positive' if is_leader else 'kpi-sub-negative'}">
            {rating_tier.split(']')[0] + ']' if ']' in rating_tier else rating_tier}
        </span>
    </div>
    """, unsafe_allow_html=True)

with k2:
    yoy_val = m.get("achieved_yoy_pct", 0.0)
    is_yoy_good = yoy_val >= 5.0
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-top">
            <span class="kpi-label">Total Footprint</span>
            <span class="kpi-icon">🏭</span>
        </div>
        <div class="kpi-value">{m['total_ghg']:,.0f}<span style="font-size:1.1rem; color:#94A3B8;"> MT</span></div>
        <span class="kpi-sub {'kpi-sub-positive' if is_yoy_good else 'kpi-sub-negative'}">
            {'-' if yoy_val > 0 else ''}{yoy_val:.1f}% YoY Reduction
        </span>
    </div>
    """, unsafe_allow_html=True)

with k3:
    ren_val = m.get("renewable_pct", 0.0)
    is_re100 = m.get("re100_committed", False)
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-top">
            <span class="kpi-label">Clean Energy</span>
            <span class="kpi-icon">⚡</span>
        </div>
        <div class="kpi-value">{ren_val:.1f}%</div>
        <span class="kpi-sub {'kpi-sub-positive' if is_re100 else 'kpi-sub-neutral'}">
            {'RE100 Pledged' if is_re100 else 'No RE100 Pledge'}
        </span>
    </div>
    """, unsafe_allow_html=True)

with k4:
    div_val = m.get("waste_diverted_pct", 0.0)
    target_diff = div_val - 75.0
    is_div_good = target_diff >= 0
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-top">
            <span class="kpi-label">Waste Diversion</span>
            <span class="kpi-icon">♻️</span>
        </div>
        <div class="kpi-value">{div_val:.1f}%</div>
        <span class="kpi-sub {'kpi-sub-positive' if is_div_good else 'kpi-sub-negative'}">
            {target_diff:+.1f}% vs 75% Target
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==========================================
# Main Navigation Tabs
# ==========================================
tab_exec, tab_ghg, tab_resources, tab_granite = st.tabs([
    "📊 Executive Scorecard & 4-Pillar Radar",
    "🏭 GHG Protocol Scope 1–3 Waterfall",
    "🌿 Resources & Governance Analytics",
    "🤖 IBM Granite Strategic Roadmap"
])

# ------------------------------------------
# TAB 1: Executive Scorecard & Radar Chart
# ------------------------------------------
with tab_exec:
    st.subheader("Statutory Compliance & 4-Pillar Scoring")
    st.caption("Proportional ESG audit scoring across Environmental (60 pts), Social (15 pts), Governance (15 pts), and Data Quality (10 pts).")

    if any("SEBI" in s or "BRSR" in s for s in standards_list) or "India" in country:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%); border: 1px solid #FDBA74; border-left: 6px solid #EA580C; border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 1.25rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 4px;">
                <span style="font-weight:800; color:#9A3412; font-size:1.05rem; font-family:'Outfit', sans-serif;">🇮🇳 SEBI BRSR Core Statutory Compliance Mandate</span>
                <span style="background:#EA580C; color:white; font-size:0.75rem; font-weight:700; padding:2px 10px; border-radius:12px;">Top 1,000 Listed Entities</span>
            </div>
            <div style="color:#7C2D12; font-size:0.88rem; line-height:1.5;">
                <b>Statutory Reference:</b> Aligned with Circular <i>SEBI/HO/CFD/CFD-SEC-2/P/CIR/2023/122</i> requiring reasonable assurance across 9 ESG attributes, mandatory Scope 1, 2, and 3 value chain disclosures for top 250 entities, Section 135 Companies Act CSR 2% spend mandate, and alignment with India's COP26 <i>Panchamrit</i> Net-Zero 2070 decarbonization pathway.
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_chart_left, col_chart_right = st.columns([1, 1])

    with col_chart_left:
        # Metallic Gauge Chart
        gauge_color = "#10B981" if ("EXCELLENT" in rating_tier) else ("#3B82F6" if ("GOOD" in rating_tier) else ("#F59E0B" if score >= 50.0 else "#EF4444"))
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Composite Readiness (0-100)", 'font': {'size': 18, 'color': '#1E3A8A', 'family': 'Outfit'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': gauge_color, 'thickness': 0.28},
                'steps': [
                    {'range': [0, 50], 'color': '#FEE2E2'},
                    {'range': [50, 70], 'color': '#FEF3C7'},
                    {'range': [70, 85], 'color': '#E0E7FF'},
                    {'range': [85, 100], 'color': '#D1FAE5'}
                ],
                'threshold': {
                    'line': {'color': "#0F172A", 'width': 3},
                    'thickness': 0.8,
                    'value': 85.0
                }
            }
        ))
        fig_gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_chart_right:
        # 4-Pillar Radar / Polar Chart
        env_pct = (pillars.get('environmental', {}).get('score', 0) / 60.0) * 100
        soc_pct = (pillars.get('social', {}).get('score', 0) / 15.0) * 100
        gov_pct = (pillars.get('governance', {}).get('score', 0) / 15.0) * 100
        dq_pct = (pillars.get('data_quality', {}).get('score', 0) / 10.0) * 100

        categories = ['Environmental (60 pts)', 'Social (15 pts)', 'Governance (15 pts)', 'Data Quality (10 pts)']
        values = [env_pct, soc_pct, gov_pct, dq_pct]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='#2563EB', width=2),
            name='Audited Entity'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[80, 80, 80, 80, 80],
            theta=categories + [categories[0]],
            mode='lines',
            line=dict(color='#10B981', width=1.5, dash='dash'),
            name='Leader Benchmark (80%)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=300,
            margin=dict(l=30, r=30, t=30, b=30),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Benchmark Audit Table
    st.markdown("### Statutory Benchmark Compliance Matrix")
    bench_rows = []
    for metric_name, d in benchmarks.items():
        bench_rows.append({
            "Pillar": d.get("pillar", "General"),
            "Indicator": metric_name,
            "Achieved Value": d["value"],
            "Variance": d["variance"],
            "Weighted Score": d["score"],
            "Compliance Status": d["status"],
            "Risk Rating": d["risk"]
        })
    df_bench = pd.DataFrame(bench_rows)
    try:
        st.dataframe(df_bench, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(df_bench, use_container_width=True)

    # ==========================================
    # v3.0: SASB SICS Sector Materiality Weighting
    # ==========================================
    st.markdown("---")
    st.markdown("### 📐 SASB SICS Sector-Specific Materiality Weighting")
    st.caption("Adjusts scoring weights dynamically based on industry operational characteristics (SASB SICS / ISSB IFRS S2).")

    sec_res = advisor.calculate_sector_weighted_score(analysis, sector=sector_choice)
    sec_col1, sec_col2 = st.columns([1, 1.2])

    with sec_col1:
        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:1.2rem; border-left:5px solid #3B82F6;">
            <div style="font-size:0.85rem; font-weight:700; color:#64748B; text-transform:uppercase;">Selected Sector Profile</div>
            <div style="font-size:1.35rem; font-weight:800; color:#0F172A; font-family:'Outfit', sans-serif;">{sec_res['sector']}</div>
            <div style="margin-top:0.6rem; font-size:1.6rem; font-weight:800; color:#2563EB;">
                {sec_res['weighted_score']:.1f}<span style="font-size:1rem; color:#64748B;"> / 100</span>
            </div>
            <div style="font-size:0.82rem; color:#475569; margin-top:0.3rem;">
                Standard: <i>{sec_res.get('alignment', 'SASB SICS')}</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sec_col2:
        df_sec_weights = pd.DataFrame([
            {"Material Category": cat, "Weight": f"{w:.0f}%"}
            for cat, w in sec_res.get("weights", {}).items()
        ])
        st.dataframe(df_sec_weights, use_container_width=True, hide_index=True)

    # ==========================================
    # v3.0: CSRD Double Materiality Matrix (ESRS 1 & 2)
    # ==========================================
    st.markdown("---")
    st.markdown("### 🌐 CSRD Double Materiality Matrix (ESRS 1 & ESRS 2)")
    st.caption("Dual-axis evaluation: Financial Risk Exposure (Outside-In) vs. Environmental & Social Impact (Inside-Out).")

    dm = advisor.evaluate_double_materiality(analysis)
    f_risk = dm["financial_risk_score"]
    i_impact = dm["impact_materiality_score"]

    col_dm_plot, col_dm_desc = st.columns([1.1, 0.9])

    with col_dm_plot:
        fig_dm = go.Figure()
        # Shaded background quadrants
        fig_dm.add_shape(type="rect", x0=50, y0=50, x1=100, y1=100, fillcolor="rgba(239, 68, 68, 0.1)", line=dict(width=0))
        fig_dm.add_shape(type="rect", x0=0, y0=50, x1=50, y1=100, fillcolor="rgba(245, 158, 11, 0.1)", line=dict(width=0))
        fig_dm.add_shape(type="rect", x0=50, y0=0, x1=100, y1=50, fillcolor="rgba(59, 130, 246, 0.1)", line=dict(width=0))
        fig_dm.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50, fillcolor="rgba(16, 185, 129, 0.1)", line=dict(width=0))

        # Quadrant threshold lines
        fig_dm.add_hline(y=50, line_dash="dash", line_color="#94A3B8")
        fig_dm.add_vline(x=50, line_dash="dash", line_color="#94A3B8")

        # Entity scatter point
        fig_dm.add_trace(go.Scatter(
            x=[i_impact],
            y=[f_risk],
            mode="markers+text",
            marker=dict(size=18, color="#1E40AF", line=dict(width=2, color="#FFFFFF")),
            text=[company_name.split()[0]],
            textposition="top center",
            name="Audited Entity"
        ))

        fig_dm.update_layout(
            title="Double Materiality Positioning",
            xaxis_title="Inside-Out Environmental & Social Impact (0-100)",
            yaxis_title="Outside-In Financial Risk Exposure (0-100)",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100]),
            height=320,
            margin=dict(l=30, r=30, t=40, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,250,252,0.6)'
        )
        st.plotly_chart(fig_dm, use_container_width=True)

    with col_dm_desc:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:1.1rem; height:100%;">
            <div style="font-size:0.85rem; font-weight:700; color:#64748B; text-transform:uppercase;">Materiality Quadrant</div>
            <div style="font-size:1.15rem; font-weight:800; color:#0F172A; margin:0.3rem 0;">{dm['quadrant']}</div>
            <div style="font-size:0.88rem; color:#334155; line-height:1.5;">
                • <b>Outside-In Financial Exposure</b>: <code>{f_risk:.1f}/100</code> (Evaluates carbon taxation, fossil price exposure, and compliance litigation).<br>
                • <b>Inside-Out Planetary Impact</b>: <code>{i_impact:.1f}/100</code> (Evaluates total GHG volume, water neutrality, and resource circularity).
            </div>
            <div style="margin-top:0.7rem; font-size:0.8rem; color:#059669; font-weight:600;">
                ✓ Compliant with EU CSRD ESRS 1 General Principles
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # v3.0: Multi-Company Framework Scorecard Summary
    # ==========================================
    st.markdown("---")
    st.markdown("### 🏆 Multi-Company Framework Scorecard Summary")
    st.caption("Cross-framework benchmark comparison across GRI Baseline, CSRD ESRS Strict, and SEBI BRSR Core.")

    scorecard_data = advisor.generate_multi_framework_scorecard()
    df_sc = pd.DataFrame(scorecard_data)
    df_sc.columns = ["Company Entity", "Audit Cycle", "GRI Baseline Score", "CSRD ESRS Strict Score", "SEBI BRSR Core Status", "Primary Audit Priority"]
    st.dataframe(df_sc, use_container_width=True, hide_index=True)

# ------------------------------------------
# TAB 2: GHG Scope 1-3 Waterfall & Analytics
# ------------------------------------------
with tab_ghg:
    st.subheader("Greenhouse Gas Protocol Scope 1–3 Architecture")
    st.caption("Operational boundary accounting from direct fuel combustion through upstream and downstream supply chains.")

    s1 = m["scope_1"]
    s2 = m["scope_2"]
    s3 = m["scope_3"]
    tot_ghg = m["total_ghg"]

    col_wf, col_pie = st.columns([1.1, 0.9])

    with col_wf:
        # Plotly Waterfall Chart
        fig_waterfall = go.Figure(go.Waterfall(
            name="GHG Protocol Accounting",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Scope 1 (Direct)", "Scope 2 (Electricity)", "Scope 3 (Value Chain)", "Total Corporate Footprint"],
            y=[s1, s2, s3, 0],
            text=[f"{s1:,.0f} MT", f"{s2:,.0f} MT", f"{s3:,.0f} MT", f"{tot_ghg:,.0f} MT"],
            textposition="outside",
            connector={"line": {"color": "#94A3B8"}},
            decreasing={"marker": {"color": "#10B981"}},
            increasing={"marker": {"color": "#3B82F6"}},
            totals={"marker": {"color": "#1E3A8A"}}
        ))
        fig_waterfall.update_layout(
            title="Scope 1–3 Cumulative Footprint Waterfall (MT CO2e)",
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with col_pie:
        # Interactive Hole Donut Chart
        df_scopes = pd.DataFrame({
            "Scope": ["Scope 1 (Direct)", "Scope 2 (Electricity)", "Scope 3 (Value Chain)"],
            "MT CO2e": [s1, s2, s3]
        })
        fig_donut = px.pie(
            df_scopes,
            values="MT CO2e",
            names="Scope",
            title="Carbon Share by Scope",
            hole=0.55,
            color_discrete_sequence=["#1E40AF", "#3B82F6", "#F59E0B"]
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Scope 3 Criticality Callout
    s3_pct = m["scope_3_pct"]
    if s3_pct >= 50.0:
        st.warning(
            f"⚠️ **Scope 3 Criticality**: Supply chain represents **{s3_pct:.1f}%** ({s3:,.0f} MT CO2e) "
            f"of total emissions. In accordance with the Scope 3 Standard, Tier 1 supplier audits and Category 1 LCAs are required."
        )

    # Figure 1: Real-World Enterprise Benchmark Comparison (Log Scale)
    st.markdown("---")
    st.markdown("#### 🌐 Figure 1: Enterprise GHG Emissions Comparison (Log Scale)")
    st.caption("Comparison of Scope 1, Scope 2, and Upstream Scope 3 emissions across EcoGlobal Enterprise, Infosys Limited (India SEBI BRSR), and Siemens AG (kt CO2e).")

    fig_bench_compare = go.Figure()
    fig_bench_compare.add_trace(go.Bar(
        name='Scope 1 (Direct)',
        x=['EcoGlobal Enterprise', 'Infosys Limited (India)', 'Siemens AG'],
        y=[14.25, 19.5, 220.5],
        marker_color='#004F71'
    ))
    fig_bench_compare.add_trace(go.Bar(
        name='Scope 2 (Electricity)',
        x=['EcoGlobal Enterprise', 'Infosys Limited (India)', 'Siemens AG'],
        y=[8.94, 42.8, 220.5],
        marker_color='#00829B'
    ))
    fig_bench_compare.add_trace(go.Bar(
        name='Scope 3 (Upstream / Value Chain)',
        x=['EcoGlobal Enterprise', 'Infosys Limited (India)', 'Siemens AG'],
        y=[46.3, 148.0, 416758.0],
        marker_color='#A5D6E3'
    ))

    fig_bench_compare.update_layout(
        barmode='group',
        yaxis_type="log",
        yaxis_title="kt CO2e (Log Scale)",
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,0.8)'
    )
    st.plotly_chart(fig_bench_compare, use_container_width=True)

# ------------------------------------------
# TAB 3: Resources & Ethical Governance
# ------------------------------------------
with tab_resources:
    st.subheader("Resource Conservation & Ethical Governance Indicators")

    col_res_l, col_res_r = st.columns([1, 1])

    with col_res_l:
        st.markdown("#### Clean Energy & Resource Sourcing")
        clean_mwh = m.get("renewable_mwh", 0.0)
        tot_mwh = m.get("total_mwh", 0.0)
        conv_mwh = max(0.0, tot_mwh - clean_mwh)

        fig_energy = go.Figure(data=[
            go.Bar(name='Renewable Clean Energy', x=['MWh Consumed'], y=[clean_mwh], marker_color='#10B981'),
            go.Bar(name='Conventional Grid Power', x=['MWh Consumed'], y=[conv_mwh], marker_color='#64748B')
        ])
        fig_energy.update_layout(
            barmode='stack',
            title="Energy Mix: Clean vs Conventional Grid Power (MWh)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_energy, use_container_width=True)

        w1, w2 = st.columns(2)
        with w1:
            st.metric("Total Water Withdrawal", f"{m.get('water_withdrawn_m3', 0.0):,.0f} m³")
        with w2:
            st.metric("Water Recycling Ratio", f"{m.get('water_recycled_pct', 0.0):.1f}%", delta="Target: ≥50%")

    with col_res_r:
        st.markdown("#### Governance & Social Accountability")
        female_board = m.get("female_board_rep_pct", 0.0)
        indep_board = m.get("independent_directors_pct", 0.0)
        supplier_code = m.get("supplier_signoff_pct", 0.0)

        st.markdown(f"**Board Gender Diversity**: `{female_board:.1f}%` (Benchmark: ≥40%)")
        st.progress(min(1.0, max(0.0, female_board / 100.0)))

        st.markdown(f"**Independent Board Directors**: `{indep_board:.1f}%` (Benchmark: ≥75%)")
        st.progress(min(1.0, max(0.0, indep_board / 100.0)))

        st.markdown(f"**Supplier Code of Conduct Sign-off**: `{supplier_code:.1f}%` (Benchmark: ≥95%)")
        st.progress(min(1.0, max(0.0, supplier_code / 100.0)))

        st.markdown(f"**Gender Pay Equity Ratio**: `{m.get('gender_pay_equity_ratio', 1.0):.2f} : 1.00`")

        if "India" in country or any("SEBI" in s for s in standards_list):
            st.markdown("---")
            st.markdown("#### 🇮🇳 Indian Statutory Mandates (Companies Act 2013 & SEBI)")
            ic1, ic2 = st.columns(2)
            with ic1:
                st.metric("Sec. 135 CSR Spend", "2.05% of Net Profit", delta="Mandate: ≥ 2.0%")
            with ic2:
                st.metric("Water Neutrality / ZLD", f"{m.get('water_recycled_pct', 0.0):.1f}% Recycled", delta="Zero Liquid Discharge")

        if dq.get("validation_warnings"):
            st.markdown("---")
            st.markdown("#### ⚠️ Data Quality Diagnostics")
            for w in dq["validation_warnings"]:
                st.caption(f"• {w}")

# ------------------------------------------
# TAB 4: IBM Granite Roadmap & SC Johnson Matrix
# ------------------------------------------
with tab_granite:
    st.subheader("🤖 IBM Granite AI Strategic Roadmap & SC Johnson Matrix")
    st.info(
        "Prioritized decarbonization action cards synthesized by rule-based prioritization, "
        "the SC Johnson CapEx vs. Impact matrix, and IBM Granite 3.0 foundation model prompt reasoning."
    )

    # SC Johnson CapEx vs. Impact Decarbonization Matrix Chart
    st.markdown("#### 📐 SC Johnson Decarbonization Opportunity Matrix")
    matrix_data = pd.DataFrame([
        {"Action": "Raw Material Decarbonization (BASF Model)", "CapEx": "Mid CapEx", "Impact": "Game-Changing", "Size": 25, "Color": "#10B981"},
        {"Action": "Tier 1 Supplier Verification (Abengoa Model)", "CapEx": "Low CapEx", "Impact": "Game-Changing", "Size": 30, "Color": "#059669"},
        {"Action": "Downstream Product Efficiency (IKEA Model)", "CapEx": "Mid CapEx", "Impact": "Game-Changing", "Size": 28, "Color": "#3B82F6"},
        {"Action": "Virtual Power Purchase Agreements (VPPAs)", "CapEx": "Low CapEx", "Impact": "Incremental", "Size": 18, "Color": "#F59E0B"},
        {"Action": "Fleet & Heat Pump Electrification", "CapEx": "High CapEx", "Impact": "Game-Changing", "Size": 22, "Color": "#6366F1"},
        {"Action": "Closed-Loop Wastewater Bioreactors", "CapEx": "Low CapEx", "Impact": "Incremental", "Size": 15, "Color": "#06B6D4"}
    ])

    fig_matrix = px.scatter(
        matrix_data,
        x="CapEx",
        y="Impact",
        text="Action",
        size="Size",
        color="Action",
        category_orders={"CapEx": ["Low CapEx", "Mid CapEx", "High CapEx"], "Impact": ["Incremental", "Game-Changing"]},
        title="SC Johnson Opportunity Matrix: CapEx vs. Decarbonization Impact"
    )
    fig_matrix.update_traces(textposition='top center', textfont=dict(size=11, family='Plus Jakarta Sans'))
    fig_matrix.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,0.8)'
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    # Dynamic Priority Action Cards
    st.markdown("#### 🎯 Priority Action Cards")
    roadmap_items = advisor.generate_roadmap_items(analysis)

    for idx, item in enumerate(roadmap_items, 1):
        st.markdown(f"""
        <div class="rec-card-modern">
            <div class="rec-header-row">
                <span class="rec-badge-pillar" style="background:#EFF6FF; color:{item.get('badge_color', '#3B82F6')};">
                    {item.get('pillar', 'Strategic Pillar')} &bull; {item.get('priority', 'Priority')}
                </span>
                <span style="font-size:0.8rem; font-weight:600; color:#64748B;">Benchmark: {item.get('benchmark_case', 'Global Standard')}</span>
            </div>
            <div class="rec-card-title">{idx}. {item.get('title')}</div>
            <div class="rec-meta-box">📊 <b>Observed Metrics</b>: {item.get('metric')}</div>
            <div class="rec-action-text">🎯 <b>Strategic Action Directive</b>: {item.get('action')}</div>
        </div>
        """, unsafe_allow_html=True)

    # 8-Step Decarbonization Workflow Stepper
    with st.expander("🧭 EcoGranite 8-Step Enterprise Decarbonization Workflow (Interactive Stepper)"):
        if hasattr(advisor, "get_eight_step_roadmap"):
            eight_steps = advisor.get_eight_step_roadmap()
        elif hasattr(advisor, "granite_client") and hasattr(advisor.granite_client, "get_eight_step_roadmap"):
            eight_steps = advisor.granite_client.get_eight_step_roadmap()
        else:
            from granite_client import GraniteReasoningClient
            eight_steps = GraniteReasoningClient().get_eight_step_roadmap()
        for s in eight_steps:
            st.markdown(f"""
            <div class="step-box">
                <div class="step-num-pill">{s['step']}</div>
                <div>
                    <div style="font-weight:700; color:#0F172A; font-size:0.95rem;">{s['title']}</div>
                    <div style="font-size:0.88rem; color:#475569; margin-top:2px;">{s['action']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Corporate Benchmark Case Studies
    with st.expander("🏢 Corporate Benchmark Case Studies (Siemens AG, Infosys India, Abengoa, SC Johnson, BASF, IKEA)"):
        st.markdown("""
        - **Infosys Limited (SEBI BRSR Core & Carbon Neutrality)**: Achieved voluntary carbon neutrality in 2020 (30 years ahead of Paris Agreement targets); operates with 74.8% renewable electricity through on-site solar and off-site PPAs, 65.4% campus water recycling (ZLD), and automated supplier ESG onboarding across Tier 1 digital supply networks under SEBI BRSR Core circular guidelines.
        - **Siemens AG (DEGREE Sustainability Framework)**: Achieved interim 2025 Scope 1 & 2 targets (-66% reduction, 441 kt CO2e) one year in advance; 84% renewable electricity; 32.6% female top management representation (exceeded); 20% executive ESG Stock Award index integration; 694M MT customer avoided emissions.
        - **Abengoa (Mandatory Supplier Verification)**: Enforces standardized GHG calculation templates across supply chain tiers, requiring third-party verified emissions integrated into its mandatory Social Responsibility Code of Conduct.
        - **SC Johnson (CapEx vs. Impact Decarbonization Matrix)**: Classifies emissions reduction options across Capital Expenditure levels (Low/Mid/High CapEx) against strategic impact (Game-Changing vs Incremental) and implementation horizons.
        - **BASF (Category 1 Raw Material LCA Disaggregation)**: Evaluated Category 1 purchased goods, revealing that 93% of emissions stemmed directly from raw material extraction; deployed primary LCAs covering ~90% of purchased products by weight.
        - **IKEA (Category 11 Product Energy Transformation)**: Determined use of sold products drove 20% of net footprint (~6M MT CO2e); executed a +50% product energy efficiency transformation surpassing direct operational impacts.
        - **National Grid (Capital Allocation Internalization)**: Quantified full value chain impacts, internalizing carbon costs into utility investment decision-making.
        """)

    # Granite Narrative Synthesis (Decoupled execution)
    st.markdown("#### 🧠 Granite Narrative Synthesis")
    session_key = f"granite_insights_{company_name}_{reporting_year}"
    if session_key not in st.session_state:
        st.session_state[session_key] = None

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        if st.button("✨ Synthesize Granite AI Narrative", use_container_width=True):
            with st.spinner("Invoking IBM Granite reasoning engine..."):
                st.session_state[session_key] = advisor.generate_ai_insights(analysis)

    with col_info:
        if st.session_state[session_key]:
            st.success("Narrative generated and cached for session.")

    if st.session_state[session_key]:
        st.text_area("IBM Granite Synthesis Output", value=st.session_state[session_key], height=180)

    # Full text audit report export
    full_text_report = advisor.generate_audit_report(raw_data, analysis, ai_insights=st.session_state[session_key])
    with st.expander("📄 View Full Audit Report (Text/CLI format)"):
        st.code(full_text_report, language="text")

    # Download action buttons
    st.markdown("#### 📥 Export Audit Deliverables")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        st.download_button(
            label="Download ESG Audit Report (.txt)",
            data=full_text_report,
            file_name=f"EcoGranite_Audit_{company_name.replace(' ', '_')}_{reporting_year}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with dl_col2:
        st.download_button(
            label="Download Normalized Schema (.json)",
            data=json.dumps(analysis, indent=2),
            file_name=f"EcoGranite_Normalized_{company_name.replace(' ', '_')}_{reporting_year}.json",
            mime="application/json",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.caption("Powered by IBM Granite 3.0 & IBM Docling | EcoGranite-Advisor v2.0 | Built for IBM SkillsBuild AI Builders Challenge")
