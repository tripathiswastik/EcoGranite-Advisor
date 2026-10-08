"""
EcoGranite-Advisor: Streamlit ESG Compliance & Decarbonization Dashboard
Autonomous Corporate Sustainability & Decarbonization Audit Engine.
Powered by IBM Granite 3.0 Reasoning & IBM Docling Document Parsing.
"""

import json
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
    page_title="EcoGranite-Advisor | ESG Compliance Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Theme & Custom CSS
st.markdown("""
<style>
    /* Main typography & headers */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
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
    
    /* Dynamic Metric Card styling */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        border-left: 5px solid #10B981;
        transition: transform 0.15s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
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
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.4rem;
    }
    .rec-metric {
        font-size: 0.9rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.4rem;
    }
    .rec-action {
        font-size: 0.95rem;
        color: #1E293B;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_dataset() -> Dict[str, Any]:
    """Loads the sample corporate ESG disclosure dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(base_dir, "sample_esg_report.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Standalone fallback if sample_esg_report.json is missing
    return {
        "company_name": "EcoGlobal Enterprise Corp.",
        "reporting_year": 2024,
        "standards": ["GRI", "TCFD", "SASB"],
        "emissions_metric_tons_co2e": {
            "scope_1_direct": 14250.0,
            "scope_2_indirect_market": 8940.0,
            "scope_3_value_chain": 46300.0,
            "total_ghg": 69490.0,
            "target_reduction_2030_pct": 45.0,
            "achieved_reduction_yoy_pct": 8.4
        },
        "renewable_energy": {
            "total_mwh_consumed": 38400.0,
            "renewable_mwh": 26880.0,
            "renewable_share_pct": 70.0,
            "re100_committed": True
        },
        "water_and_waste": {
            "total_water_withdrawal_m3": 124000.0,
            "water_recycled_pct": 42.0,
            "waste_generated_tons": 3200.0,
            "waste_diverted_from_landfill_pct": 86.5
        },
        "social_and_governance": {
            "female_board_representation_pct": 44.0,
            "gender_pay_equity_ratio": 0.98,
            "independent_directors_pct": 80.0,
            "supplier_code_of_conduct_signoff_pct": 98.2
        }
    }


# ==========================================
# Sidebar: Controls & Document Ingestion
# ==========================================
with st.sidebar:
    st.markdown("### 🌍 EcoGranite Engine")
    st.caption("IBM Granite 3.0 + IBM Docling v2")
    st.markdown("---")

    # Foundation Model Selector
    model_choice = st.selectbox(
        "IBM Granite Model",
        options=[
            "ibm-granite/granite-3.0-8b-instruct",
            "ibm-granite/granite-3.0-2b-instruct",
            "ibm-granite/granite-guardian-3.0-8b"
        ],
        index=0,
        help="Select foundation model for automated synthesis and compliance reasoning."
    )

    # Ingestion Source
    st.markdown("#### 📂 Disclosure Ingestion")
    docling_status = "🟢 Ready" if DOCLING_AVAILABLE else "🟡 Fallback Parser"
    st.caption(f"Docling Ingest Pipeline: **{docling_status}**")

    uploaded_file = st.file_uploader(
        "Upload Corporate Disclosure (.json, .pdf, .docx)",
        type=["json", "pdf", "docx"],
        help="IBM Docling will parse multi-modal tables and text into the ESG auditing schema."
    )

    # Initialize advisor engine
    advisor = EcoGraniteAdvisor(model_id=model_choice)

    # Process input data
    raw_data = None
    if uploaded_file is not None:
        try:
            with st.spinner(f"Ingesting {uploaded_file.name} via Docling Pipeline..."):
                raw_data = advisor.load_document(uploaded_file)
            st.success(f"Loaded: {uploaded_file.name}")
        except Exception as err:
            st.error(f"Error parsing file: {err}")
            raw_data = load_default_dataset()
    else:
        raw_data = load_default_dataset()

    # Reset button
    if st.button("🔄 Reset to Sample Disclosure", use_container_width=True):
        raw_data = load_default_dataset()
        st.rerun()

    # Sidebar Entity Summary
    st.markdown("---")
    st.markdown("#### 🏢 Audited Entity")
    company_name = raw_data.get("company_name", "Unknown Corp")
    reporting_year = raw_data.get("reporting_year", 2024)
    standards_list = raw_data.get("standards", ["GRI", "TCFD", "SASB"])

    st.markdown(f"**Entity**: `{company_name}`")
    st.markdown(f"**Reporting Cycle**: `FY {reporting_year}`")
    st.markdown(f"**Frameworks**: `{', '.join(standards_list)}`")

    # API Key status
    has_api = bool(os.getenv("WATSONX_APIKEY") or os.getenv("HUGGINGFACE_API_KEY"))
    if has_api:
        st.success("API Mode: IBM WatsonX Live API Connected")
    else:
        st.info("API Mode: Local Granite 3.0 Reasoning Engine (Deterministic)")


# ==========================================
# Compliance Analysis Execution
# ==========================================
analysis = advisor.analyze_compliance(raw_data)
m = analysis["metrics"]
benchmarks = analysis["benchmarks"]
score = analysis["esg_readiness_score"]
rating_tier = analysis["rating_tier"]

# Dynamic Header
st.markdown(
    f"<div class='main-header'>EcoGranite-Advisor: ESG Compliance Dashboard"
    f"<span class='engine-badge'>Granite 3.0 Powered</span></div>",
    unsafe_allow_html=True
)
st.markdown(
    f"<div class='sub-header'>Autonomous Corporate Sustainability & Decarbonization Audit Engine | "
    f"<b>{company_name}</b> (FY {reporting_year}) — Frameworks: {', '.join(standards_list)}</div>",
    unsafe_allow_html=True
)

# ==========================================
# KPI Header Cards (Continuous Scoring)
# ==========================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    delta_tier = rating_tier.split()[0] + " Tier"
    st.metric(
        label="Composite ESG Score",
        value=f"{score:.1f} / 100",
        delta=rating_tier,
        delta_color="normal" if score >= 70.0 else "inverse"
    )

with kpi2:
    yoy_val = m.get("achieved_yoy_pct", 0.0)
    st.metric(
        label="Total Carbon Footprint",
        value=f"{m['total_ghg']:,.0f} MT CO2e",
        delta=f"-{yoy_val:.1f}% YoY Reduction",
        delta_color="normal" if yoy_val >= 5.0 else "inverse"
    )

with kpi3:
    ren_val = m.get("renewable_pct", 0.0)
    pledge_status = "RE100 Pledged" if m.get("re100_committed") else "No RE100 Pledge"
    st.metric(
        label="Renewable Electricity Share",
        value=f"{ren_val:.1f}%",
        delta=pledge_status,
        delta_color="normal" if m.get("re100_committed") else "off"
    )

with kpi4:
    div_val = m.get("waste_diverted_pct", 0.0)
    target_diff = div_val - 75.0
    st.metric(
        label="Waste Diverted from Landfill",
        value=f"{div_val:.1f}%",
        delta=f"{target_diff:+.1f}% vs 75% Target",
        delta_color="normal" if target_diff >= 0 else "inverse"
    )

st.markdown("---")

# ==========================================
# Main Navigation Tabs
# ==========================================
tab_exec, tab_ghg, tab_resources, tab_granite = st.tabs([
    "📊 Executive Scorecard",
    "🏭 GHG Scope 1–3 Analysis",
    "🌿 Resources & Governance",
    "🤖 IBM Granite Strategic Roadmap"
])

# ------------------------------------------
# TAB 1: Executive Scorecard
# ------------------------------------------
with tab_exec:
    st.subheader("Statutory Compliance & Continuous Scoring")
    st.caption("Proportional ESG readiness scoring based on CSRD, GRI, and SEC climate disclosure rules.")

    col_gauge, col_bench = st.columns([1, 1])

    with col_gauge:
        # Dynamic Gauge Chart
        gauge_color = "#10B981" if score >= 85.0 else ("#3B82F6" if score >= 70.0 else ("#F59E0B" if score >= 50.0 else "#EF4444"))
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Composite Readiness Score (0-100)", 'font': {'size': 18, 'color': '#1E3A8A'}},
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
        fig_gauge.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_bench:
        st.markdown("### Statutory Benchmark Breakdown")
        bench_rows = []
        for metric_name, d in benchmarks.items():
            bench_rows.append({
                "Statutory Indicator": metric_name,
                "Achieved": d["value"],
                "Variance": d["variance"],
                "Weighted Score": d["score"],
                "Audit Status": d["status"],
                "Risk Rating": d["risk"]
            })
        df_bench = pd.DataFrame(bench_rows)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

# ------------------------------------------
# TAB 2: GHG Scope 1-3 Footprint Analysis
# ------------------------------------------
with tab_ghg:
    st.subheader("Greenhouse Gas (GHG) Scope 1–3 Emissions Breakdown")
    st.caption("Accounting aligned with GHG Protocol Corporate Standard across direct, electricity, and supply chain.")

    s1 = m["scope_1"]
    s2 = m["scope_2"]
    s3 = m["scope_3"]
    tot_ghg = m["total_ghg"]

    col_donut, col_bar = st.columns([1, 1])

    df_scopes = pd.DataFrame({
        "Scope Category": ["Scope 1 (Direct Operations)", "Scope 2 (Purchased Electricity)", "Scope 3 (Value Chain & Upstream)"],
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
        fig_donut.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
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
        fig_bar.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Scope 3 Callout
    s3_pct = m["scope_3_pct"]
    if s3_pct >= 50.0:
        st.warning(
            f"⚠️ **Scope 3 Criticality**: Supply chain accounts for **{s3_pct:.1f}%** ({s3:,.0f} MT CO2e) "
            f"of corporate emissions. Upstream supplier engagement is essential to hit the 2030 SBTi reduction target."
        )
    else:
        st.info(
            f"ℹ️ **Operational Dominance**: Direct operations (Scope 1+2) drive **{(100 - s3_pct):.1f}%** "
            f"({s1 + s2:,.0f} MT CO2e). Focus capital expenditure on process electrification and heating transition."
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
        st.progress(min(1.0, female_board / 100.0))

        st.markdown(f"**Independent Board Directors**: `{indep_board:.1f}%` (Benchmark: ≥75%)")
        st.progress(min(1.0, indep_board / 100.0))

        st.markdown(f"**Supplier Code of Conduct Sign-off**: `{supplier_code:.1f}%`")
        st.progress(min(1.0, supplier_code / 100.0))

        st.markdown(f"**Gender Pay Equity Ratio**: `{m.get('gender_pay_equity_ratio', 1.0):.2f} : 1.00`")

# ------------------------------------------
# TAB 4: IBM Granite Strategic Roadmap
# ------------------------------------------
with tab_granite:
    st.subheader("🤖 IBM Granite AI Decarbonization Roadmap")
    st.info(
        "The strategic recommendations below are dynamically generated by IBM Granite 3.0 "
        "reasoning over the ingested disclosures, GHG Protocol Scope 1-3 breakdown, and benchmark variances."
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

    # Full text audit report expander
    full_text_report = advisor.generate_audit_report(raw_data, analysis)
    with st.expander("📄 View Full Synthesized Audit Report (Text/CLI format)"):
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
