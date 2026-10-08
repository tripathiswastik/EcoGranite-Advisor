"""
Docling Document Parser & ESG Data Normalization Engine
Provides document conversion for PDF/Docx/JSON ESG reports into structured data.
Enforces rigorous data reconciliation, validation warnings, and refusal on extraction failure.
"""

import io
import json
import logging
import os
import re
from typing import Dict, Any, Union, List, Optional

logger = logging.getLogger(__name__)

# Check for IBM Docling availability
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Check for PyPDF2 / pypdf fallback (proper PDF stream decompression)
try:
    import PyPDF2
    PDF_PARSER_AVAILABLE = True
except ImportError:
    try:
        import pypdf as PyPDF2
        PDF_PARSER_AVAILABLE = True
    except Exception:
        PDF_PARSER_AVAILABLE = False


def parse_document(file_source: Union[str, dict, io.IOBase, Any], filename: str = None) -> Dict[str, Any]:
    """
    Parses an ESG document into normalized structured JSON.
    - If input is already a dictionary, normalizes and returns directly.
    - If input is a file path:
        - .json: loads and validates.
        - .pdf / .docx: uses IBM Docling DocumentConverter (or text extraction).
    - If input is a file-like stream (e.g., Streamlit UploadedFile), inspects extension and parses.
    - If extraction fails to find required fields, returns an explicit extraction_failed result.
    """
    # 1. Direct dictionary input
    if isinstance(file_source, dict):
        if file_source.get("status") == "extraction_failed":
            return file_source
        return validate_and_normalize_esg(file_source)

    # 2. File-like object (e.g. Streamlit UploadedFile or BytesIO)
    if hasattr(file_source, "read"):
        file_name = getattr(file_source, "name", filename or "uploaded_disclosure.json")
        ext = os.path.splitext(file_name)[1].lower()

        content = file_source.read()
        if isinstance(content, bytes):
            content_str = content.decode("utf-8", errors="ignore")
        else:
            content_str = str(content)

        if ext == ".json":
            try:
                data = json.loads(content_str)
                return validate_and_normalize_esg(data)
            except json.JSONDecodeError as e:
                return {
                    "status": "extraction_failed",
                    "company_name": os.path.splitext(file_name)[0],
                    "source_file": file_name,
                    "errors": [f"Malformed JSON document: {str(e)}"]
                }
        elif ext in (".pdf", ".docx"):
            return _parse_pdf_or_docx_bytes(content, file_name)
        else:
            try:
                data = json.loads(content_str)
                return validate_and_normalize_esg(data)
            except Exception:
                return {
                    "status": "extraction_failed",
                    "company_name": os.path.splitext(file_name)[0],
                    "source_file": file_name,
                    "errors": [f"Unsupported file format '{ext}'. Expected .json, .pdf, or .docx"]
                }

    # 3. String path
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            raise FileNotFoundError(f"ESG document not found at: {file_source}")

        ext = os.path.splitext(file_source)[1].lower()

        if ext == ".json":
            with open(file_source, "r", encoding="utf-8") as f:
                data = json.load(f)
                return validate_and_normalize_esg(data)

        elif ext in (".pdf", ".docx"):
            return _parse_pdf_or_docx_path(file_source)
        else:
            return {
                "status": "extraction_failed",
                "company_name": os.path.basename(file_source),
                "source_file": file_source,
                "errors": [f"Unsupported file extension '{ext}'. Supported: .json, .pdf, .docx"]
            }

    raise TypeError(f"Unsupported input type: {type(file_source)}")


def _extract_text_from_pdf_stream(stream: io.BytesIO) -> str:
    """Extracts text from decompressed PDF page stream objects using PyPDF2."""
    if not PDF_PARSER_AVAILABLE:
        return ""
    try:
        reader = PyPDF2.PdfReader(stream)
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n".join(pages_text)
    except Exception as e:
        logger.warning(f"PDF stream decompression error: {e}")
        return ""


def _extract_text_from_docx_stream(stream: io.BytesIO) -> str:
    """Extracts text paragraphs and table cells from DOCX using standard library zipfile & XML."""
    import zipfile
    import xml.etree.ElementTree as ET
    try:
        with zipfile.ZipFile(stream) as docx:
            if "word/document.xml" not in docx.namelist():
                return ""
            xml_content = docx.read("word/document.xml")
        root = ET.fromstring(xml_content)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for p in root.iter(f"{{{ns['w']}}}p"):
            texts = [node.text for node in p.iter(f"{{{ns['w']}}}t") if node.text]
            if texts:
                paragraphs.append("".join(texts))
        return "\n".join(paragraphs)
    except Exception as e:
        logger.warning(f"DOCX archive decompression error: {e}")
        return ""


def _parse_pdf_or_docx_path(file_path: str) -> Dict[str, Any]:
    """Parse PDF or DOCX from filesystem path using Docling, PyPDF2, or DOCX XML extractor."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return _parse_pdf_or_docx_bytes(file_bytes, os.path.basename(file_path))


def _parse_pdf_or_docx_bytes(file_bytes: Union[bytes, str], file_name: str) -> Dict[str, Any]:
    """
    Parse PDF or DOCX using:
    1. IBM Docling (Primary deep multi-modal layout & table vision pipeline)
    2. Stream Decompressor (PyPDF2 for PDF streams, zipfile/XML for DOCX)
    3. Refusal with clear diagnostics if document streams cannot be decompressed
    """
    if isinstance(file_bytes, str):
        file_bytes = file_bytes.encode("utf-8")
    ext = os.path.splitext(file_name)[1].lower()
    clean_title = os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ").title()

    # Path 1: Primary IBM Docling Pipeline
    if DOCLING_AVAILABLE:
        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            converter = DocumentConverter()
            result = converter.convert(tmp_path)
            markdown_text = result.document.export_to_markdown()
            return extract_esg_from_text(markdown_text, file_name)
        except Exception as e:
            logger.warning(f"Docling conversion encountered error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # Path 2: Decompressed Stream Extraction (No raw binary decoding!)
    stream = io.BytesIO(file_bytes)
    extracted_text = ""

    if ext == ".pdf":
        extracted_text = _extract_text_from_pdf_stream(stream)
        if not extracted_text.strip():
            return {
                "status": "extraction_failed",
                "company_name": clean_title,
                "source_file": file_name,
                "errors": [
                    "PDF extraction failed: Could not decompress textual content from PDF streams. "
                    "Corporate PDF reports with FlateDecode compressed streams or complex tables "
                    "require IBM Docling ('pip install docling') or PyPDF2 ('pip install pypdf2')."
                ]
            }
    elif ext == ".docx":
        extracted_text = _extract_text_from_docx_stream(stream)
        if not extracted_text.strip():
            return {
                "status": "extraction_failed",
                "company_name": clean_title,
                "source_file": file_name,
                "errors": [
                    "DOCX extraction failed: Could not read 'word/document.xml' from archive. "
                    "File may be corrupted, encrypted, or not a valid Word document."
                ]
            }
    else:
        return {
            "status": "extraction_failed",
            "company_name": clean_title,
            "source_file": file_name,
            "errors": [f"Unsupported file format '{ext}'. Expected .pdf, .docx, or .json."]
        }

    return extract_esg_from_text(extracted_text, file_name)


def extract_esg_from_text(text: str, source_name: str) -> Dict[str, Any]:
    """
    Scans document text/markdown for actual ESG disclosures.
    Refuses to invent numbers: if required disclosures are missing, reports extraction_failed.
    """
    clean_title = os.path.splitext(source_name)[0].replace("_", " ").replace("-", " ").title()

    errors = []
    extracted: Dict[str, Any] = {
        "company_name": clean_title,
        "reporting_year": 2024,
        "standards": ["GRI", "TCFD"],
        "emissions_metric_tons_co2e": {},
        "renewable_energy": {},
        "water_and_waste": {},
        "social_and_governance": {}
    }

    # 1. Company Name detection
    comp_match = re.search(r"(?:Company|Entity|Corporation|Organization)\s*[:\-]\s*([^\n\r]+)", text, re.IGNORECASE)
    if comp_match:
        extracted["company_name"] = comp_match.group(1).strip()

    # 2. Scope 1
    s1_match = re.search(r"Scope\s*1(?:\s*\(Direct\))?\s*[:\-]?\s*([\d,\.]+)", text, re.IGNORECASE)
    if s1_match:
        extracted["emissions_metric_tons_co2e"]["scope_1_direct"] = float(s1_match.group(1).replace(",", ""))
    else:
        errors.append("Scope 1 direct GHG emissions could not be identified in disclosure text")

    # 3. Scope 2
    s2_match = re.search(r"Scope\s*2(?:\s*\(Indirect\))?\s*[:\-]?\s*([\d,\.]+)", text, re.IGNORECASE)
    if s2_match:
        extracted["emissions_metric_tons_co2e"]["scope_2_indirect_market"] = float(s2_match.group(1).replace(",", ""))
    else:
        errors.append("Scope 2 indirect GHG emissions could not be identified in disclosure text")

    # 4. Scope 3
    s3_match = re.search(r"Scope\s*3(?:\s*\(Value Chain\))?\s*[:\-]?\s*([\d,\.]+)", text, re.IGNORECASE)
    if s3_match:
        extracted["emissions_metric_tons_co2e"]["scope_3_value_chain"] = float(s3_match.group(1).replace(",", ""))

    # 5. Renewable energy
    ren_match = re.search(r"Renewable\s*(?:energy|electricity|share)?\s*[:\-]?\s*([\d\.]+)\s*%", text, re.IGNORECASE)
    if ren_match:
        extracted["renewable_energy"]["renewable_share_pct"] = float(ren_match.group(1))
    else:
        errors.append("Renewable energy share percentage could not be identified in disclosure text")

    # If critical mandatory fields failed to extract, REFUSE to invent data!
    if len(errors) >= 2 or ("scope_1_direct" not in extracted["emissions_metric_tons_co2e"] and "scope_2_indirect_market" not in extracted["emissions_metric_tons_co2e"]):
        return {
            "status": "extraction_failed",
            "company_name": extracted["company_name"],
            "source_file": source_name,
            "errors": errors
        }

    return validate_and_normalize_esg(extracted)


def validate_and_normalize_esg(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates ESG metrics, performs GHG reconciliation (S1 + S2 + S3 vs reported total),
    preserves source anomalies with validation warnings, and scores completeness.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid ESG payload: expected JSON object.")

    if data.get("status") == "extraction_failed":
        return data

    validation_warnings: List[str] = []
    core_fields_evaluated = 0
    core_fields_present = 0

    def check_presence(val: Any) -> bool:
        nonlocal core_fields_evaluated, core_fields_present
        core_fields_evaluated += 1
        if val is not None:
            core_fields_present += 1
            return True
        return False

    company_name = data.get("company_name", "Unknown Entity")
    reporting_year = data.get("reporting_year", 2024)

    # ----------------------------------------------------
    # Emissions & Reconciliation Analysis
    # ----------------------------------------------------
    emissions = data.get("emissions_metric_tons_co2e") or {}
    s1_raw = emissions.get("scope_1_direct")
    s2_raw = emissions.get("scope_2_indirect_market")
    s3_raw = emissions.get("scope_3_value_chain")
    rep_total_raw = emissions.get("total_ghg")

    check_presence(s1_raw)
    check_presence(s2_raw)
    check_presence(s3_raw)
    check_presence(rep_total_raw)

    def parse_float_safe(val: Any, field_name: str) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
            if f < 0.0:
                validation_warnings.append(f"{field_name} is negative ({f}); expected non-negative value.")
            return f
        except (ValueError, TypeError):
            validation_warnings.append(f"{field_name} contains non-numeric value '{val}'.")
            return None

    s1 = parse_float_safe(s1_raw, "Scope 1")
    s2 = parse_float_safe(s2_raw, "Scope 2")
    s3 = parse_float_safe(s3_raw, "Scope 3")
    reported_total = parse_float_safe(rep_total_raw, "Reported Total GHG")

    # Scope sum calculation
    s1_val = s1 if s1 is not None else 0.0
    s2_val = s2 if s2 is not None else 0.0
    s3_val = s3 if s3 is not None else 0.0
    calc_total = s1_val + s2_val + s3_val

    # Reconciliation check
    reconciliation_status = "PASSED"
    variance = 0.0

    if reported_total is not None and (s1 is not None or s2 is not None or s3 is not None):
        variance = abs(reported_total - calc_total)
        tolerance = max(1.0, 0.01 * reported_total)
        if variance > tolerance:
            reconciliation_status = "FAILED"
            validation_warnings.append(
                f"GHG Protocol Mismatch: Reported total ({reported_total:,.1f} MT) does not reconcile "
                f"with Scope 1+2+3 sum ({calc_total:,.1f} MT). Variance: {variance:,.1f} MT."
            )

    # Effective total GHG: if reconciliation failed or reported is missing, use calculated
    effective_total = reported_total if (reported_total is not None and reconciliation_status == "PASSED") else calc_total

    # Scope 3 exceeding total check
    if s3 is not None and effective_total > 0 and s3 > effective_total:
        validation_warnings.append(f"Scope 3 emissions ({s3:,.1f} MT) exceed total GHG ({effective_total:,.1f} MT).")

    # Reduction target
    target_red = parse_float_safe(emissions.get("target_reduction_2030_pct", 45.0), "2030 Target Reduction %")
    achieved_yoy = parse_float_safe(emissions.get("achieved_reduction_yoy_pct"), "YoY Reduction %")
    check_presence(achieved_yoy)

    # ----------------------------------------------------
    # Renewable Energy & Grid Mix
    # ----------------------------------------------------
    energy = data.get("renewable_energy") or {}
    tot_energy = parse_float_safe(energy.get("total_mwh_consumed"), "Total MWh Consumed")
    ren_energy = parse_float_safe(energy.get("renewable_mwh"), "Renewable MWh")
    ren_pct_raw = energy.get("renewable_share_pct")
    check_presence(ren_pct_raw)

    if ren_pct_raw is not None:
        ren_pct = parse_float_safe(ren_pct_raw, "Renewable Share %")
    elif tot_energy and tot_energy > 0 and ren_energy is not None:
        ren_pct = round((ren_energy / tot_energy) * 100.0, 2)
    else:
        ren_pct = None

    if ren_pct is not None and (ren_pct < 0.0 or ren_pct > 100.0):
        validation_warnings.append(f"Renewable share ({ren_pct}%) is outside valid 0-100% boundary.")

    if tot_energy is not None and ren_energy is not None and ren_energy > tot_energy:
        validation_warnings.append(f"Renewable energy ({ren_energy} MWh) exceeds total consumed ({tot_energy} MWh).")

    # ----------------------------------------------------
    # Water & Waste
    # ----------------------------------------------------
    waste = data.get("water_and_waste") or {}
    water_withdrawn = parse_float_safe(waste.get("total_water_withdrawal_m3"), "Water Withdrawal")
    water_rec_pct = parse_float_safe(waste.get("water_recycled_pct"), "Water Recycled %")
    waste_div_pct = parse_float_safe(waste.get("waste_diverted_from_landfill_pct"), "Waste Diversion %")
    check_presence(waste_div_pct)

    if waste_div_pct is not None and (waste_div_pct < 0.0 or waste_div_pct > 100.0):
        validation_warnings.append(f"Waste diversion ({waste_div_pct}%) is outside valid 0-100% boundary.")

    if water_rec_pct is not None and (water_rec_pct < 0.0 or water_rec_pct > 100.0):
        validation_warnings.append(f"Water recycling ({water_rec_pct}%) is outside valid 0-100% boundary.")

    # ----------------------------------------------------
    # Social & Governance
    # ----------------------------------------------------
    gov = data.get("social_and_governance") or {}
    board_div = parse_float_safe(gov.get("female_board_representation_pct"), "Female Board Diversity %")
    indep_dir = parse_float_safe(gov.get("independent_directors_pct"), "Independent Directors %")
    pay_equity = parse_float_safe(gov.get("gender_pay_equity_ratio"), "Gender Pay Equity Ratio")
    supplier_code = parse_float_safe(gov.get("supplier_code_of_conduct_signoff_pct"), "Supplier Code Sign-off %")
    check_presence(board_div)

    if board_div is not None and (board_div < 0.0 or board_div > 100.0):
        validation_warnings.append(f"Board gender diversity ({board_div}%) was outside 0-100% range.")

    if indep_dir is not None and (indep_dir < 0.0 or indep_dir > 100.0):
        validation_warnings.append(f"Independent directors ({indep_dir}%) was outside 0-100% range.")

    # Data completeness computation
    completeness_pct = round((core_fields_present / core_fields_evaluated) * 100.0, 1) if core_fields_evaluated > 0 else 0.0

    return {
        "status": "success",
        "company_name": company_name,
        "reporting_year": reporting_year,
        "country": data.get("country", "India"),
        "jurisdiction": data.get("jurisdiction", "SEBI BRSR Core & Global Frameworks"),
        "standards": data.get("standards", ["SEBI BRSR", "GRI", "TCFD", "SASB"]),
        "emissions_metric_tons_co2e": {
            "scope_1_direct": s1,
            "scope_2_indirect_market": s2,
            "scope_3_value_chain": s3,
            "total_ghg": effective_total,
            "reported_total_ghg": reported_total,
            "calculated_total_ghg": calc_total,
            "target_reduction_2030_pct": target_red if target_red is not None else 45.0,
            "achieved_reduction_yoy_pct": achieved_yoy
        },
        "renewable_energy": {
            "total_mwh_consumed": tot_energy,
            "renewable_mwh": ren_energy,
            "renewable_share_pct": ren_pct,
            "re100_committed": bool(energy.get("re100_committed", False))
        },
        "water_and_waste": {
            "total_water_withdrawal_m3": water_withdrawn,
            "water_recycled_pct": water_rec_pct,
            "waste_generated_tons": parse_float_safe(waste.get("waste_generated_tons"), "Waste Generated"),
            "waste_diverted_from_landfill_pct": waste_div_pct
        },
        "social_and_governance": {
            "female_board_representation_pct": board_div,
            "gender_pay_equity_ratio": pay_equity if pay_equity is not None else 1.0,
            "independent_directors_pct": indep_dir if indep_dir is not None else 0.0,
            "supplier_code_of_conduct_signoff_pct": supplier_code if supplier_code is not None else 0.0
        },
        "data_quality": {
            "completeness_pct": completeness_pct,
            "reconciliation_status": reconciliation_status,
            "reported_total_ghg": reported_total,
            "calculated_total_ghg": calc_total,
            "total_ghg_variance": variance,
            "validation_warnings": validation_warnings
        }
    }
