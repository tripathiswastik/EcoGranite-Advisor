"""
EcoGranite-Advisor: Streamlit ESG Compliance & Decarbonization Dashboard
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

# Page Configuration
st.set_page_config(
    page_title="EcoGranite-Advisor | ESG Sustainability Auditor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Theme & Custom CSS
st.markdown("""
<style>
    /* Main typography & headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
        line-height: 1.4;
    }
    
    /* Engine status pill */
    .engine-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }

    .recon-pass {
        color: #059669;
        font-weight: bold;
    }

    .recon-fail {
        color: #DC2626;
        font-weight: bold;
    }
    
    /* Roadmap recommendation card */
    .rec-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .rec-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.3rem;
    }
    .rec-metric {
        font-size: 0.9rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.3rem;
    }
    .rec-action {
        font-size: 0.95rem;
        color: #1E293B;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Cached Advisor Factory
# ==========================================
@st.cache_resource
def get_advisor(model_id: str) -> EcoGraniteAdvisor:
    """Instantiates and caches the EcoGraniteAdvisor instance."""
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
            "📙 Sample: CarbonHeavy Corp (Lagging/Poor)",
            "📕 Sample: Incoherent Disclosures (Reconciliation Failure)",
            "📁 Upload Custom File (.json, .pdf, .docx)"
        ],
        index=0
    )

    raw_data = None
    if data_source_mode == "📗 Sample: EcoGlobal Enterprise (Good/Compliant)":
        raw_data = load_dataset_file("sample_esg_report.json")
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
    # REFUSAL STATE: Do NOT show fake scores or invented numbers!
    st.markdown(
        f"<div class='main-header'>EcoGranite-Advisor: ESG Compliance Dashboard"
        f"<span class='engine-badge' style='background:#DC2626;'>Audit Refused</span></div>",
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
m = analysis["metrics"]
benchmarks = analysis["benchmarks"]
pillars = analysis.get("pillars", {})
score = analysis["esg_readiness_score"]
rating_tier = analysis["rating_tier"]
dq = analysis.get("data_quality", {})

# Dynamic Header
st.markdown(
    f"<div class='main-header'>EcoGranite-Advisor: ESG Sustainability Auditor"
    f"<span class='engine-badge'>Granite 3.0 Powered</span></div>",
    unsafe_allow_html=True
)
st.markdown(
    f"<div class='sub-header'>Corporate Sustainability & Decarbonization Audit Engine | "
    f"<b>{company_name}</b> (FY {reporting_year}) — Frameworks: {', '.join(standards_list)}</div>",
    unsafe_allow_html=True
)

# Data Quality & Reconciliation Notice Banner
recon_status = dq.get("reconciliation_status", "UNKNOWN")
if recon_status != "PASSED":
    st.error(
        f"🚨 **Data Reconciliation Failure**: Scope 1+2+3 sum ({dq.get('calculated_total_ghg', 0):,.1f} MT) "
        f"does not match reported total GHG ({dq.get('reported_total_ghg', 0):,.1f} MT). "
        f"Variance: **{dq.get('total_ghg_variance', 0):,.1f} MT**. Rating disqualified from leader tier."
    )
elif dq.get("validation_warnings"):
    st.warning(f"⚠️ **Data Quality Notice**: {len(dq['validation_warnings'])} disclosure warning(s) detected. See Data Quality tab.")

# ==========================================
# KPI Header Cards (4 Pillars)
# ==========================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Sustainability Readiness",
        value=f"{score:.1f} / 100",
        delta=rating_tier,
        delta_color="normal" if "EXCELLENT" in rating_tier or "GOOD" in rating_tier else "inverse"
    )

with kpi2:
    yoy_val = m.get("achieved_yoy_pct", 0.0)
    st.metric(
        label="Total Carbon Footprint",
        value=f"{m['total_ghg']:,.0f} MT CO2e",
        delta=f"-{yoy_val:.1f}% YoY Reduction" if yoy_val > 0 else f"{yoy_val:.1f}% YoY",
        delta_color="normal" if yoy_val >= 5.0 else "inverse"
    )

with kpi3:
    ren_val = m.get("renewable_pct", 0.0)
    pledge_status = "RE100 Pledged" if m.get("re100_committed") else "No RE100 Pledge"
    st.metric(
        label="Renewable Electricity",
        value=f"{ren_val:.1f}%",
        delta=pledge_status,
        delta_color="normal" if m.get("re100_committed") else "off"
    )

with kpi4:
    div_val = m.get("waste_diverted_pct", 0.0)
    target_diff = div_val - 75.0
    st.metric(
        label="Waste Landfill Diversion",
        value=f"{div_val:.1f}%",
        delta=f"{target_diff:+.1f}% vs 75% Target",
        delta_color="normal" if target_diff >= 0 else "inverse"
    )

st.markdown("---")

# ==========================================
# Main Navigation Tabs
# ==========================================
tab_exec, tab_ghg, tab_resources, tab_granite = st.tabs([
    "📊 Executive Scorecard & Pillars",
    "🏭 GHG Scope 1–3 Breakdown",
    "🌿 Resources & Governance",
    "🤖 IBM Granite Strategic Roadmap"
])

# ------------------------------------------
# TAB 1: Executive Scorecard & Pillars
# ------------------------------------------
with tab_exec:
    st.subheader("Statutory Compliance & 4-Pillar Scoring")
    st.caption("Comprehensive ESG scoring across Environmental (60), Social (15), Governance (15), and Data Quality (10).")

    col_gauge, col_bench = st.columns([1, 1])

    with col_gauge:
        # Dynamic Gauge Chart
        gauge_color = "#10B981" if ("EXCELLENT" in rating_tier) else ("#3B82F6" if ("GOOD" in rating_tier) else ("#F59E0B" if score >= 50.0 else "#EF4444"))
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Composite Readiness (0-100)", 'font': {'size': 18, 'color': '#1E3A8A'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
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
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Pillar breakdown metrics
        st.markdown("#### Pillar Breakdown")
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            st.metric("Environmental", f"{pillars.get('environmental', {}).get('score', 0)} / 60")
        with p_c2:
            st.metric("Social", f"{pillars.get('social', {}).get('score', 0)} / 15")
        with p_c3:
            st.metric("Governance", f"{pillars.get('governance', {}).get('score', 0)} / 15")
        with p_c4:
            st.metric("Data Quality", f"{pillars.get('data_quality', {}).get('score', 0)} / 10")

    with col_bench:
        st.markdown("### Statutory Benchmark Breakdown")
        bench_rows = []
        for metric_name, d in benchmarks.items():
            bench_rows.append({
                "Pillar": d.get("pillar", "General"),
                "Indicator": metric_name,
                "Achieved": d["value"],
                "Variance": d["variance"],
                "Score": d["score"],
                "Audit Status": d["status"],
                "Risk": d["risk"]
            })
        df_bench = pd.DataFrame(bench_rows)
        try:
            st.dataframe(df_bench, use_container_width=True, hide_index=True)
        except TypeError:
            st.dataframe(df_bench, use_container_width=True)

# ------------------------------------------
# TAB 2: GHG Scope 1-3 Footprint Analysis
# ------------------------------------------
with tab_ghg:
    st.subheader("Greenhouse Gas (GHG) Protocol Scope 1–3 Analysis")
    st.caption("Emissions reconciliation and value chain footprint assessment.")

    s1 = m["scope_1"]
    s2 = m["scope_2"]
    s3 = m["scope_3"]
    tot_ghg = m["total_ghg"]

    col_donut, col_bar = st.columns([1, 1])

    df_scopes = pd.DataFrame({
        "Scope Category": ["Scope 1 (Direct Operations)", "Scope 2 (Electricity)", "Scope 3 (Value Chain)"],
        "MT CO2e": [s1, s2, s3],
        "Percentage": [
            (s1 / tot_ghg * 100) if tot_ghg > 0 else 0,
            (s2 / tot_ghg * 100) if tot_ghg > 0 else 0,
            (s3 / tot_ghg * 100) if tot_ghg > 0 else 0
        ]
    })

    with col_donut:
        fig_donut = px.pie(
            df_scopes,
            values="MT CO2e",
            names="Scope Category",
            title="Carbon Footprint Distribution by Scope",
            hole=0.45,
            color_discrete_sequence=["#1E40AF", "#3B82F6", "#F59E0B"]
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            df_scopes,
            x="Scope Category",
            y="MT CO2e",
            text_auto=',.0f',
            title="Absolute Emissions Breakdown (Metric Tons CO2e)",
            color="Scope Category",
            color_discrete_sequence=["#1E40AF", "#3B82F6", "#F59E0B"]
        )
        fig_bar.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Reconciliation and Scope 3 Callouts
    s3_pct = m["scope_3_pct"]
    if s3_pct >= 50.0:
        st.warning(
            f"⚠️ **Scope 3 Criticality**: Supply chain represents **{s3_pct:.1f}%** ({s3:,.0f} MT CO2e) "
            f"of total emissions. Supplier engagement is critical to reach the 2030 SBTi reduction target."
        )

# ------------------------------------------
# TAB 3: Resources & Governance
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
            title="Energy Mix: Clean vs Conventional Power (MWh)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_energy, use_container_width=True)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.metric("Total Water Withdrawal", f"{m.get('water_withdrawn_m3', 0.0):,.0f} m³")
        with col_w2:
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

        if dq.get("validation_warnings"):
            st.markdown("---")
            st.markdown("#### ⚠️ Data Quality Diagnostics")
            for w in dq["validation_warnings"]:
                st.caption(f"• {w}")

# ------------------------------------------
# TAB 4: IBM Granite Strategic Roadmap
# ------------------------------------------
with tab_granite:
    st.subheader("🤖 IBM Granite AI Strategic Roadmap")
    st.info(
        "Prioritized decarbonization action cards synthesized by rule-based prioritization "
        "and IBM Granite 3.0 foundation model prompt reasoning."
    )

    roadmap_items = advisor.generate_roadmap_items(analysis)

    for idx, item in enumerate(roadmap_items, 1):
        with st.container():
            st.markdown(f"""
            <div class='rec-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:0.8rem; font-weight:700; color:{item.get("badge_color", "#3B82F6")}; text-transform:uppercase;'>
                        {item.get('pillar', 'Strategic Pillar')} &bull; {item.get('priority', 'Priority')}
                    </span>
                </div>
                <div class='rec-title'>{idx}. {item.get('title')}</div>
                <div class='rec-metric'>📊 <b>Observed Disclosure</b>: {item.get('metric')}</div>
                <div class='rec-action'>🎯 <b>Target Action Plan</b>: {item.get('action')}</div>
            </div>
            """, unsafe_allow_html=True)

    # 8-Step Standard Decarbonization Workflow
    eight_steps = advisor.get_eight_step_roadmap()
    with st.expander("🧭 EcoGranite 8-Step Enterprise Decarbonization Roadmap (Methodology Workflow)"):
        for s in eight_steps:
            st.markdown(f"**Step {s['step']}: {s['title']}**\n\n> {s['action']}")

    # Benchmark case studies
    with st.expander("🏢 Global Corporate Decarbonization Benchmarks (Abengoa, SC Johnson, BASF, IKEA, National Grid)"):
        st.markdown("""
        - **Abengoa (Mandatory Supplier Verification)**: Enforces standardized GHG calculation templates across supply chain tiers, requiring third-party verified emissions integrated into its mandatory Social Responsibility Code of Conduct.
        - **SC Johnson (CapEx vs. Impact Decarbonization Matrix)**: Classifies emissions reduction options across Capital Expenditure levels (Low/Mid/High CapEx) against strategic impact (Game-Changing vs Incremental) and implementation horizons.
        - **BASF (Category 1 Raw Material LCA Disaggregation)**: Evaluated Category 1 purchased goods, revealing that 93% of emissions stemmed directly from raw material extraction; deployed primary LCAs covering ~90% of purchased products by weight.
        - **IKEA (Category 11 Product Energy Transformation)**: Determined use of sold products drove 20% of net footprint (~6M MT CO2e); executed a +50% product energy efficiency transformation surpassing direct operational impacts.
        - **National Grid (Capital Allocation Internalization)**: Quantified full value chain impacts, internalizing carbon costs into utility investment decision-making.
        """)

    # Narrative AI Generation Button (Decoupled execution)
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
