# 🌍 EcoGranite-Advisor: Autonomous ESG Sustainability Auditor

[![Challenge](https://img.shields.io/badge/Challenge-IBM%20SkillsBuild%20AI%20Builders-052F5F.svg?style=flat&logo=IBM)](https://github.com/IBM-SkillsBuild-AI-Builders-Challenge)
[![Model](https://img.shields.io/badge/Model-IBM%20Granite%203.0-blue.svg)](https://huggingface.co/ibm-granite)
[![Parser](https://img.shields.io/badge/Parser-IBM%20Docling%20Bridge-green.svg)](https://github.com/DS4SD/docling)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(6%2F6)-brightgreen.svg)](https://github.com/tripathiswastik/EcoGranite-Advisor)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Author](https://img.shields.io/badge/Author-Swastik%20Tripathi-blueviolet.svg)](https://github.com/tripathiswastik)

An intelligent, autonomous ESG (Environmental, Social, and Governance) corporate sustainability audit engine developed for the **IBM SkillsBuild AI Builders Challenge**. 

EcoGranite-Advisor ingests corporate sustainability disclosures, provides an **IBM Docling** extraction bridge for complex PDF reports, performs risk-weighted continuous compliance auditing, and generates decarbonization roadmaps powered by **IBM Granite 3.0** (with both live WatsonX API support and a deterministic local fallback).

---

## 📌 Problem Statement

Corporate ESG & climate disclosures are filed in massive, multi-column PDF reports containing intricate emission tables, supply chain metrics, and varying reporting standards (**GRI**, **SASB**, **TCFD**). 

Traditional text extraction tools flatten these tables, misaligning GHG rows with carbon quantities. Furthermore, manual compliance auditing against statutory benchmarks (e.g., SBTi targets, RE100 commitments) is labor-intensive and prone to human error.

---

## 💡 Solution Architecture

```text
+------------------------------------+
|     Corporate ESG Report (PDF/JSON)|
+------------------------------------+
                  |
                  v
+------------------------------------+
|  IBM Docling Engine (esg_parser.py)|
|  • Layout & nested table parsing   |
|  • Normalization & zero-clamping   |
+------------------------------------+
                  |
                  v
+------------------------------------+
|      Continuous Scoring Engine     |
|  • Proportional weighted scoring   |
|  • Risk & variance calculations    |
+------------------------------------+
                  |
                  v
+------------------------------------+
| IBM Granite 3.0 (granite_client.py)|
|  • WatsonX live API inference      |
|  • Dynamic, data-derived roadmap   |
+------------------------------------+
                  |
                  v
+------------------------------------+
|    Executive ESG Audit Report      |
+------------------------------------+
```

---

## 🌟 Key Features

1. **📄 IBM Docling Integration Bridge (`esg_parser.py`)**:
   - Accepts both `.pdf` / `.docx` files via IBM Docling's `DocumentConverter` and pre-processed `.json` disclosures.
   - Normalizes missing fields with safe defaults and clamps negative/zero values.
2. **🧠 Dual-Mode IBM Granite 3.0 Engine (`granite_client.py`)**:
   - **Live Mode**: Calls IBM WatsonX AI (`ibm-watsonx-ai`) foundation models when `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` are configured.
   - **Local Mode**: Uses deterministic Granite 3.0 prompt logic deriving all recommendations from the supplied metrics.
3. **📈 Continuous, Proportional Scoring Model**:
   - Replaced abrupt binary cutoff jumps with continuous weighted scoring across 4 pillars (Renewable Energy, YoY Abatement, Waste Diversion, Board Diversity).
4. **🔍 Dynamic, Data-Derived Recommendations**:
   - Scope 3 percentages, water circularity goals, and YoY on-track flags are calculated from the actual input payload.
5. **💻 Flexible CLI Tool**:
   - Accepts arbitrary input files using `--input <file>` rather than hardcoding a demo fixture.
6. **🧪 Comprehensive Test Suite (`test_advisor.py`)**:
   - 6 automated unit tests validating boundary cases, negative values, and scoring accuracy.

---

## 📂 Repository Structure

```text
EcoGranite-Advisor/
├── advisor_engine.py          # Main CLI entrypoint & continuous compliance auditor
├── esg_parser.py              # IBM Docling PDF/DOCX bridge & data normalizer
├── granite_client.py          # Dual IBM Granite client (WatsonX API + Local Reasoning)
├── test_advisor.py            # Unit test suite covering scoring and edge cases
├── sample_esg_report.json     # Sample corporate disclosure dataset
├── requirements.txt           # Dependencies for Docling, WatsonX, and Streamlit
└── README.md                  # Challenge documentation & architecture specification
```

---

## 🚀 Quick Start Guide

### 1. Run the Advisor Engine
```bash
# Run with sample report
python advisor_engine.py --input sample_esg_report.json

# Run with any custom report
python advisor_engine.py --input path/to/your_report.json
```

### 2. Run the Test Suite
```bash
python -m unittest test_advisor.py
```

### 3. Example Audit Report Output
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
   Renewable Energy Share (Target >= 60%) 70.0%      30.0/30  Compliant    Low
   YoY Emissions Reduction (Target >= 5%) 8.4%       30.0/30  Compliant (On Track) Low
   Waste Diversion (Target >= 75%)        86.5%      20.0/20  Compliant    Low
   Board Diversity (Target >= 40%)        44.0%      20.0/20  Compliant    Low
   ----------------------------------------------------------------------------

STRATEGIC RECOMMENDATIONS (IBM Granite 3.0 Reasoning):
   1. Target Value Chain Decarbonization: Scope 3 represents 66.6% of total footprint (46,300.0 MT CO2e). Deploy supplier carbon scorecards and transport route optimization.
   2. Maintain Clean Energy Momentum: Exceeded renewable target at 70.0%. Aim for 24/7 carbon-free hourly matching.
   3. Elevate Water Circularity: Water recycling is currently at 42.0% (below 50% circularity goal). Invest in closed-loop effluent treatment and membrane filtration.
================================================================================
```

---

## 🛠️ Technology Stack

- **Reasoning LLM:** [IBM Granite 3.0](https://huggingface.co/ibm-granite) / WatsonX AI Foundation Models
- **Document Intelligence:** [IBM Docling (DS4SD)](https://github.com/DS4SD/docling)
- **Runtime:** Python 3.8+
- **Reporting Standards:** GRI (Global Reporting Initiative), TCFD, SASB

---

## 👤 Author

- **Swastik Tripathi** — [GitHub (@tripathiswastik)](https://github.com/tripathiswastik)

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
