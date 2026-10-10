# 🌍 EcoGranite-Advisor: Autonomous ESG Sustainability Auditor

[![Challenge](https://img.shields.io/badge/Challenge-IBM%20SkillsBuild%20AI%20Builders-052F5F.svg?style=flat&logo=IBM)](https://github.com/IBM-SkillsBuild-AI-Builders-Challenge)
[![Model](https://img.shields.io/badge/Model-IBM%20Granite%203.0-blue.svg)](https://huggingface.co/ibm-granite)
[![Parser](https://img.shields.io/badge/Parser-IBM%20Docling%20Bridge-green.svg)](https://github.com/DS4SD/docling)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(38%2F38)-brightgreen.svg)](https://github.com/tripathiswastik/EcoGranite-Advisor)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Author](https://img.shields.io/badge/Author-Swastik%20Tripathi-blueviolet.svg)](https://github.com/tripathiswastik)

An intelligent, autonomous ESG (Environmental, Social, and Governance) corporate sustainability audit engine developed for the **IBM SkillsBuild AI Builders Challenge**. 

EcoGranite-Advisor ingests corporate sustainability disclosures, provides an **IBM Docling** extraction bridge for complex PDF reports, performs risk-weighted continuous compliance auditing across **4 Defensible Pillars (Environmental, Social, Governance, Data Quality)**, provides an interactive **Streamlit Web Dashboard (`app.py`)**, and generates decarbonization roadmaps powered by **IBM Granite 3.0** (with both live WatsonX API support and a deterministic local fallback).

---

## 📌 Problem Statement

Corporate ESG & climate disclosures are filed in massive, multi-column PDF reports containing intricate emission tables, supply chain metrics, and varying reporting standards (**GRI**, **SASB**, **TCFD**). 

Traditional text extraction tools flatten these tables, misaligning GHG rows with carbon quantities. Furthermore:
1. **Silent Data Fabrication**: Naive parsers often substitute default or sample numbers when extraction fails, concealing missing disclosures.
2. **Reconciliation Discrepancies**: Reported total GHG numbers frequently contradict the sum of Scope 1 + Scope 2 + Scope 3 disclosures.
3. **Binary Cutoff Fallacies**: Rigid thresholds penalize near-compliant companies as total failures while rewarding boundary cases without risk weighting.

---

## 💡 Solution Architecture

```text
+------------------------------------------------------+
|            Corporate ESG Report (PDF/JSON)           |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
|       IBM Docling Ingestion Engine (esg_parser.py)   |
|  • Layout & nested table extraction                  |
|  • Refusal on extraction failure (never fakes data)  |
|  • GHG Scope 1+2+3 reconciliation & anomaly checks   |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
|       4-Pillar Continuous Scoring Engine             |
|  • Environmental (60) | Social (15) | Governance (15)|
|  • Data Quality & Reconciliation Integrity (10)      |
|  • Disqualification flag on reconciliation failure   |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
|         IBM Granite 3.0 (granite_client.py)          |
|  • WatsonX live API inference / Deterministic engine |
|  • Rule-based priority cards + Narrative synthesis   |
|  • Transparent runtime metadata & error capture      |
+------------------------------------------------------+
                           |
         +-----------------+-----------------+
         |                                   |
         v                                   v
+------------------------+ +-----------------------------------+
|      CLI Auditor       | |    Streamlit Dashboard (app.py)   |
|  Text & Export Reports | | Multi-Sample Switcher & Analytics |
+------------------------+ +-----------------------------------+
```

---

## 🌟 Key Features

1. **🖥️ Interactive Streamlit Web Dashboard (`app.py`)**:
   - Executive ESG Readiness Score gauge with dynamic risk tiers ([A] Leader to [D] Critical).
   - Instant dataset switcher: Good (EcoGlobal), Poor (CarbonHeavy), Invalid (Incoherent Reconciliation Failure), and Custom File Upload.
   - GHG Protocol Scope 1–3 interactive donut and bar breakdown charts.
   - Resource circularity stacked energy charts, water recycling, and governance indicators.
   - IBM Granite AI Strategic Roadmap with prioritized action cards and audit report exporter.
2. **🛡️ Data Integrity & GHG Protocol Reconciliation**:
   - Evaluates whether Reported Total GHG = Scope 1 + Scope 2 + Scope 3.
   - Refuses to award `[A] EXCELLENT` tier if emissions data fails reconciliation.
   - Flags anomalies (> 100% diversity, renewable MWh > total MWh) in validation warnings rather than silently clamping.
3. **📄 Pre-Audit Extraction Refusal (Never Fakes Data)**:
   - If an uploaded PDF or document does not contain readable ESG disclosures, the engine clearly reports `extraction_failed` with the list of missing fields and refuses to issue a compliance score.
4. **🧠 Dual-Mode IBM Granite 3.0 Engine (`granite_client.py`)**:
   - **Live Mode**: Calls IBM WatsonX AI (`ibm-watsonx-ai`) foundation models when credentials are provided.
   - **Local Mode**: Uses deterministic prompt reasoning deriving all figures dynamically from the payload.
   - Captures and logs WatsonX errors transparently without crashing.
5. **📈 4-Pillar Continuous Scoring Model**:
   - **Environmental (60 pts)**: Renewable energy (25), YoY reduction (25), Waste diversion (10).
   - **Social (15 pts)**: Gender pay equity (7.5), Supplier code sign-off (7.5).
   - **Governance (15 pts)**: Board gender diversity (7.5), Independent directors (7.5).
6. **🇮🇳 Indian Statutory ESG Compliance Architecture**:
   - **SEBI BRSR Core**: Aligned with Circular *SEBI/HO/CFD/CFD-SEC-2/P/CIR/2023/122* mandating reasonable assurance across 9 ESG attributes and Scope 1–3 disclosures for top listed entities.
   - **Companies Act Section 135**: Audits mandatory 2% CSR spend on average net profits.
   - **SEBI LODR Regulation 17(1)**: Verifies independent director thresholds (≥50%) and gender diversity.
   - **Benchmark Dataset (`sample_esg_report_india.json`)**: Models Infosys Limited with 74.8% renewable electricity, 65.4% campus water recycling (Zero Liquid Discharge), and supply chain Scope 3 disclosures.
7. **📐 SASB SICS Sector-Specific Materiality Weighting (v3.0)**:
   - Eliminates distortions between capital-heavy manufacturing and low-direct-emission technology enterprises:
     - **Technology & Software**: Scope 3 Value Chain (45%), Scope 2 Power (20%), Governance (20%), Water & Waste (10%), Scope 1 (5%).
     - **Heavy Industry & Metals**: Scope 1 Direct (35%), Scope 2 Power (25%), Scope 3 (20%), Water & Waste (15%), Governance (5%).
     - **Financial Institutions**: Scope 3 Cat 15 Financed Emissions (65%), Board Independence (20%), Scope 2 (8%), Water (5%), Scope 1 (2%).
8. **🌐 CSRD Double Materiality Matrix (ESRS 1 & ESRS 2 - v3.0)**:
   - Evaluates dual materiality axes: **Outside-In Financial Risk Exposure** alongside **Inside-Out Environmental & Social Impact**.
9. **🏆 Multi-Company Framework Scorecard Summary (v3.0)**:
   - Standardized cross-framework audit scorecard comparing **EcoGlobal**, **Siemens AG**, and **Infosys Limited** across GRI Baseline, CSRD ESRS Strict, and SEBI BRSR Core.
10. **🧪 Comprehensive Adversarial Test Suite (`test_advisor.py`)**:
   - 32 automated unit tests (100% passing) validating continuous scoring, reconciliation, refusal, and v3.0 sector/double materiality models.

---

### 📊 Multi-Company Framework Scorecard Summary

| Company Entity | Audit Cycle | GRI Baseline Score | CSRD ESRS Strict Score | SEBI BRSR Core Status | Primary Audit Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EcoGlobal Enterprise Corp.** | FY 2024 | **100.0 / 100 [A]** | **95.0 / 100 [A]** *(RE Share < 80%)* | **Compliant** *(BRSR Core Assured)* | Target Value Chain Decarbonization |
| **Siemens AG** | FY 2024 | **98.6 / 100 [A]** | **98.6 / 100 [A]** | **Compliant** *(EU Taxonomy Aligned)* | Target Value Chain Decarbonization |
| **Infosys Limited** | FY 2024 | **99.5 / 100 [A]** | **98.0 / 100 [A]** | **BRSR Core Leader** | Target Value Chain Decarbonization |

---

## 📂 Repository Structure

```text
EcoGranite-Advisor/
├── app.py                           # Streamlit Interactive Web Dashboard
├── advisor_engine.py                # Main CLI entrypoint & continuous compliance auditor
├── esg_parser.py                    # IBM Docling PDF/DOCX bridge, reconciliation & validator
├── granite_client.py                # Dual IBM Granite client (WatsonX API + Local Reasoning)
├── test_advisor.py                  # Adversarial unit test suite (32 tests)
├── test_app_smoke.py                # Dashboard smoke and AppTest suite
├── pyproject.toml                   # Ruff & mypy configuration
├── sample_esg_report.json           # Sample 1: Compliant Leader ESG dataset (EcoGlobal)
├── sample_esg_report_india.json     # Sample 2: India SEBI BRSR Core dataset (Infosys Limited)
├── sample_esg_report_siemens.json   # Sample 3: Global Enterprise benchmark (Siemens AG)
├── sample_esg_report_poor.json      # Sample 4: Poor / Lagging ESG dataset (CarbonHeavy)
├── sample_esg_report_invalid.json   # Sample 5: Incoherent / Reconciliation failure dataset
├── ECOGRANITE_FRAMEWORK.md          # 5-Pillar Comprehensive ESG Decarbonization Specification
├── .env.example                     # Environment template for WatsonX & Hugging Face credentials
├── requirements.txt                 # Minimal dependencies for Docling, WatsonX, HF, and Streamlit
└── README.md                        # Challenge documentation & architecture specification
```

---

## 🔑 LLM Provider Setup & Configuration

EcoGranite-Advisor supports three tiered execution modes for IBM Granite foundation models:

1. **Hugging Face Inference Router** (`HUGGINGFACE_API_KEY` or `HF_TOKEN`):
   - Direct cloud inference for IBM Granite models (e.g. `ibm-granite/granite-4.2-8b`, `ibm-granite/granite-4.2-3b`, `ibm-granite/granite-3.0-8b-instruct`).
   - Requires fine-grained token with `inference.serverless.write` permission.
2. **IBM WatsonX AI** (`WATSONX_APIKEY` & `WATSONX_PROJECT_ID`):
   - Direct enterprise foundation model inference via IBM WatsonX Cloud SDK (`ibm-watsonx-ai`).
3. **Local Deterministic Reasoning Engine (Offline Fallback)**:
   - Always available with zero API keys or network connection required.
   - Evaluates disclosures against GHG Protocol math and generates auditable prioritized recommendations.

To configure API providers, copy the template and provide your credentials:
```bash
cp .env.example .env
```

---

## 🚀 Quick Start Guide

### 1. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501` to upload disclosures, inspect Scope 1–3 charts, and review Granite roadmaps.*

### 2. Run the Advisor Engine (CLI)
```bash
# Run with sample report
python advisor_engine.py --input sample_esg_report.json

# Run with any custom report
python advisor_engine.py --input path/to/your_report.json
```

### 3. Run the Automated Test Suite
```bash
# Run all unit and smoke tests
python -m unittest discover
```

### 4. Example Audit Report Output
```text
================================================================================
[*] ECOGRANITE ADVISOR: CORPORATE ESG AUDIT REPORT
================================================================================
Analysis Engine : Local IBM Granite 3.0 Reasoning Engine
Model Selected  : ibm-granite/granite-3.0-8b-instruct
Entity Name     : EcoGlobal Enterprise Corp.
Audit Cycle     : FY 2024
Frameworks Used : GRI, TCFD, SASB
--------------------------------------------------------------------------------

[+] COMPOSITE ESG READINESS SCORE: 100.0 / 100
Rating Level    : [A] EXCELLENT (ESG LEADER)

1. GREENHOUSE GAS (GHG) EMISSIONS BREAKDOWN:
   - Scope 1 (Direct Operations)       :   14,250.0 MT CO2e (20.5%)
   - Scope 2 (Purchased Electricity)   :    8,940.0 MT CO2e (12.9%)
   - Scope 3 (Supply Chain & Value)    :   46,300.0 MT CO2e (66.6%)
   - Total Carbon Footprint            :   69,490.0 MT CO2e
   - 2030 Science-Based Target (SBTi)  : 45.0% reduction
   - YoY Progress Achieved             : 8.4% (ON TRACK)

2. RENEWABLE ENERGY & TRANSITION:
   - Total Consumption                 :   38,400.0 MWh
   - Clean / Renewable Energy Sourced  :   26,880.0 MWh (70.0%)
   - RE100 Initiative Pledged          : YES

3. ESG BENCHMARK COMPLIANCE TABLE:
   ----------------------------------------------------------------------------
   Metric                                 Value      Score    Status       Risk
   ----------------------------------------------------------------------------
   Renewable Energy Share (Target >= 60%) 70.0%      25.0/25.0 Compliant    Low
   YoY Emissions Reduction (Target >= 5%) 8.4%       25.0/25.0 Compliant (On Track) Low
   Waste Diversion (Target >= 75%)        86.5%      10.0/10.0 Compliant    Low
   Board Diversity (Target >= 40%)        44.0%      7.5/7.5  Compliant    Low
   Independent Directors (Target >= 75%)  80.0%      7.5/7.5  Compliant    Low
   Gender Pay Equity (Target >= 0.98)     0.98       7.5/7.5  Compliant    Low
   Supplier Code Sign-off (Target >= 95%) 98.2%      7.5/7.5  Compliant    Low
   GHG Scope Reconciliation (S1+S2+S3)    PASSED     5.0/5.0  Compliant    Low
   ----------------------------------------------------------------------------

STRATEGIC RECOMMENDATIONS (IBM Granite 3.0 Reasoning):
   1. Target Value Chain Decarbonization: Scope 3 represents 66.6% of total footprint (46,300.0 MT CO2e). Deploy supplier carbon scorecards and transport route optimization.
   2. Maintain Clean Energy Momentum: Exceeded renewable target at 70.0%. Aim for 24/7 carbon-free hourly matching.
   3. Elevate Water Circularity: Water recycling is currently at 42.0% (below 50% circularity goal). Invest in closed-loop effluent treatment and membrane filtration.
================================================================================
```

---

## ⚖️ Methodology & Regulatory Assurance Notice

> [!IMPORTANT]
> **Audit Assurance Boundary:** EcoGranite readiness scores represent **internal diagnostic screening heuristics** and maturity proxies designed to identify reporting gaps, reconciliation discrepancies, and transition hurdles.
> 
> - **Not Legal Certification:** Scores do not constitute formal legal compliance certification or third-party reasonable assurance under EU CSRD, SEBI BRSR Core, or ISSB mandates.
> - **Accounting Assumptions:** Scope 2 accounting defaults to market-based reporting with contractual instruments; consolidation operates under the operational control boundary.
> - **Missing Data Integrity:** Missing metrics remain explicitly classified as `"Unknown (Missing Data)"` with zero credit; they are never defaulted to zero or fabricated.
> - **Deterministic Separation:** All numeric audit scores and pillar calculations are computed deterministically before AI calls; generative AI advice does not alter underlying scores.

---

## 📦 Clean Release Packaging

To build clean, publication-ready distribution archives free of `.git` internals, `.env` secrets, virtual environments, or compiled bytecode:

```bash
python build_release.py
```

The resulting zip file is written to `dist/EcoGranite-Advisor-v4.2.zip` and verified automatically against credential leakage rules.

---

## 🛠️ Technology Stack

- **Reasoning LLM:** [IBM Granite 4.2 / 3.0](https://huggingface.co/ibm-granite) / WatsonX AI Foundation Models
- **Document Intelligence:** [IBM Docling (DS4SD)](https://github.com/DS4SD/docling)
- **Runtime:** Python 3.8+
- **Reporting Standards:** GRI (Global Reporting Initiative), TCFD, SASB

---

## 👤 Author

- **Swastik Tripathi** — [GitHub (@tripathiswastik)](https://github.com/tripathiswastik)

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
