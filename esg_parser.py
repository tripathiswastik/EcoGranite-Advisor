"""
Docling Document Parser & ESG Data Normalization Engine
Provides document conversion for PDF/Docx/JSON ESG reports into structured data.
Enforces rigorous data reconciliation, validation warnings, and refusal on extraction failure.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

import importlib

# Size safety limits
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB max upload
MAX_XML_BYTES = 50 * 1024 * 1024     # 50 MB max uncompressed XML
_NUMBER_PATTERN = r"(\d[\d,]*(?:\.\d+)?)"

# Check for IBM Docling availability
try:
    _docling_mod = importlib.import_module("docling.document_converter")
    DocumentConverter = getattr(_docling_mod, "DocumentConverter")
    DOCLING_AVAILABLE = True
except Exception:
    DocumentConverter = None
    DOCLING_AVAILABLE = False

# Check for PyPDF2 / pypdf fallback
try:
    PyPDF2 = importlib.import_module("PyPDF2")
    PDF_PARSER_AVAILABLE = True
except Exception:
    try:
        PyPDF2 = importlib.import_module("pypdf")
        PDF_PARSER_AVAILABLE = True
    except Exception:
        PyPDF2 = None
        PDF_PARSER_AVAILABLE = False


class _Collector:
    """Helper to safely parse numeric inputs, check validity, and collect warnings."""

    def __init__(self, warnings: List[str]):
        self.warnings = warnings
        self.evaluated = 0
        self.present = 0

    def check_presence(self, val: Any) -> bool:
        self.evaluated += 1
        if val is not None:
            self.present += 1
            return True
        return False

    def number(self, val: Any, field_name: str) -> Optional[float]:
        if val is None:
            return None
        # In Python, bool is an instance of int, so reject booleans explicitly
        if isinstance(val, bool):
            self.warnings.append(f"{field_name} contains non-numeric boolean value '{val}'.")
            return None
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                self.warnings.append(f"{field_name} contains non-finite numeric value '{val}'.")
                return None
            if f < 0.0:
                self.warnings.append(f"{field_name} is negative ({f}); expected non-negative value.")
            return f
        except (ValueError, TypeError):
            self.warnings.append(f"{field_name} contains non-numeric value '{val}'.")
            return None


def parse_float_safe(val: Any, field_name: str, warnings: Optional[List[str]] = None) -> Optional[float]:
    """Standalone safe float parser rejecting NaN, inf, and booleans."""
    w: List[str] = warnings if warnings is not None else []
    return _Collector(w).number(val, field_name)


def parse_document(file_source: Union[str, dict, io.IOBase, Any], filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses an ESG document into normalized structured JSON.
    - If input is already a dictionary, normalizes and returns directly.
    - If input is a file path:
        - .json: loads and validates.
        - .pdf / .docx: uses IBM Docling DocumentConverter (or stream extraction).
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
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        if len(content_bytes) > MAX_UPLOAD_BYTES:
            return {
                "status": "extraction_failed",
                "company_name": os.path.splitext(file_name)[0],
                "source_file": file_name,
                "errors": [f"File size ({len(content_bytes)} bytes) exceeds maximum upload limit of {MAX_UPLOAD_BYTES} bytes."]
            }

        if ext == ".json":
            try:
                content_str = content_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                return {
                    "status": "extraction_failed",
                    "company_name": os.path.splitext(file_name)[0],
                    "source_file": file_name,
                    "errors": [f"Strict UTF-8 decoding failed for JSON document: {str(e)}"]
                }
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
            return _parse_pdf_or_docx_bytes(content_bytes, file_name)
        else:
            try:
                content_str = content_bytes.decode("utf-8")
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

        if os.path.getsize(file_source) > MAX_UPLOAD_BYTES:
            return {
                "status": "extraction_failed",
                "company_name": os.path.basename(file_source),
                "source_file": file_source,
                "errors": [f"File size exceeds maximum upload limit of {MAX_UPLOAD_BYTES} bytes."]
            }

        ext = os.path.splitext(file_source)[1].lower()

        if ext == ".json":
            try:
                with open(file_source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return validate_and_normalize_esg(data)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return {
                    "status": "extraction_failed",
                    "company_name": os.path.basename(file_source),
                    "source_file": file_source,
                    "errors": [f"Failed to decode or parse JSON file: {str(e)}"]
                }
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
    """Extracts text from decompressed PDF page stream objects using PyPDF2 / pypdf."""
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
    """
    Extracts text paragraphs and table cells from DOCX using standard library zipfile & XML.
    Defends against zip bombs and entity expansion attacks (XXE).
    """
    import xml.etree.ElementTree as ET
    import zipfile
    try:
        with zipfile.ZipFile(stream) as docx:
            total_uncompressed = sum(info.file_size for info in docx.infolist())
            if total_uncompressed > MAX_XML_BYTES:
                logger.warning(f"DOCX uncompressed size ({total_uncompressed} bytes) exceeds limit of {MAX_XML_BYTES} bytes.")
                return ""

            if "word/document.xml" not in docx.namelist():
                return ""
            xml_content = docx.read("word/document.xml")

        # Reject entity declarations to prevent XML entity expansion attacks
        if b"<!DOCTYPE" in xml_content or b"<!ENTITY" in xml_content:
            logger.warning("DOCX document.xml contains entity declaration or doctype; refusing to parse.")
            return ""

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

    # Path 2: Decompressed Stream Extraction
    stream = io.BytesIO(file_bytes)

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
        # Check for entity declarations directly in docx stream before parsing
        stream.seek(0)
        import zipfile
        try:
            with zipfile.ZipFile(stream) as docx:
                if "word/document.xml" in docx.namelist():
                    raw_xml = docx.read("word/document.xml")
                    if b"<!DOCTYPE" in raw_xml or b"<!ENTITY" in raw_xml:
                        return {
                            "status": "extraction_failed",
                            "company_name": clean_title,
                            "source_file": file_name,
                            "errors": ["DOCX extraction refused: Document contains prohibited XML entity declarations."]
                        }
        except Exception:
            pass
        stream.seek(0)
        extracted_text = _extract_text_from_docx_stream(stream)
        if not extracted_text.strip():
            return {
                "status": "extraction_failed",
                "company_name": clean_title,
                "source_file": file_name,
                "errors": [
                    "DOCX extraction failed: Could not read 'word/document.xml' from archive. "
                    "File may be corrupted, encrypted, contain prohibited entity declarations, or exceed size limits."
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


def _find_number(pattern: str, text: str) -> Optional[float]:
    """Helper to extract a single numeric capture using a robust regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        clean = match.group(1).replace(",", "").strip()
        f = float(clean)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, IndexError):
        return None


def extract_esg_from_text(text: str, source_name: str) -> Dict[str, Any]:
    """
    Scans document text/markdown for actual ESG disclosures.
    Refuses to invent numbers: missing disclosures remain None or result in extraction_failed.
    """
    clean_title = os.path.splitext(source_name)[0].replace("_", " ").replace("-", " ").title()

    errors = []
    extracted: Dict[str, Any] = {
        "company_name": clean_title,
        "reporting_year": None,
        "standards": [],
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
    s1 = _find_number(rf"Scope\s*1(?:\s*\(Direct\))?\s*[:\-]?\s*{_NUMBER_PATTERN}", text)
    if s1 is not None:
        extracted["emissions_metric_tons_co2e"]["scope_1_direct"] = s1
    else:
        errors.append("Scope 1 direct GHG emissions could not be identified in disclosure text")

    # 3. Scope 2
    s2 = _find_number(rf"Scope\s*2(?:\s*\(Indirect\))?\s*[:\-]?\s*{_NUMBER_PATTERN}", text)
    if s2 is not None:
        extracted["emissions_metric_tons_co2e"]["scope_2_indirect_market"] = s2
    else:
        errors.append("Scope 2 indirect GHG emissions could not be identified in disclosure text")

    # 4. Scope 3
    s3 = _find_number(rf"Scope\s*3(?:\s*\(Value\s*Chain\))?\s*[:\-]?\s*{_NUMBER_PATTERN}", text)
    if s3 is not None:
        extracted["emissions_metric_tons_co2e"]["scope_3_value_chain"] = s3

    # 5. Renewable energy
    ren = _find_number(rf"Renewable\s*(?:energy|electricity|share)?\s*[:\-]?\s*{_NUMBER_PATTERN}\s*%", text)
    if ren is not None:
        extracted["renewable_energy"]["renewable_share_pct"] = ren
    else:
        errors.append("Renewable energy share percentage could not be identified in disclosure text")

    # If critical mandatory fields failed to extract, refuse to invent data
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
    Never fabricates default values for missing metrics.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid ESG payload: expected JSON object.")

    if data.get("status") == "extraction_failed":
        return data

    company_name = data.get("company_name", "Unknown Entity")

    # Check for empty payload or disclosure missing all Scope 1, 2, and 3 data
    emissions_raw = data.get("emissions_metric_tons_co2e")
    if not isinstance(emissions_raw, dict) or not emissions_raw:
        # Check if emissions are defined at root
        s1_check = data.get("scope_1_direct")
        s2_check = data.get("scope_2_indirect_market")
        s3_check = data.get("scope_3_value_chain")
        if s1_check is None and s2_check is None and s3_check is None:
            return {
                "status": "extraction_failed",
                "company_name": company_name,
                "source_file": data.get("source_file", "unknown"),
                "errors": ["Required GHG emissions data (Scope 1, 2, or 3) missing from disclosure payload."]
            }
        emissions_raw = data

    validation_warnings: List[str] = []
    collector = _Collector(validation_warnings)

    s1_raw = emissions_raw.get("scope_1_direct")
    s2_raw = emissions_raw.get("scope_2_indirect_market")
    s3_raw = emissions_raw.get("scope_3_value_chain")
    rep_total_raw = emissions_raw.get("total_ghg")

    collector.check_presence(s1_raw)
    collector.check_presence(s2_raw)
    collector.check_presence(s3_raw)
    collector.check_presence(rep_total_raw)

    s1 = collector.number(s1_raw, "Scope 1")
    s2 = collector.number(s2_raw, "Scope 2")
    s3 = collector.number(s3_raw, "Scope 3")
    reported_total = collector.number(rep_total_raw, "Reported Total GHG")

    # If all 3 scopes are absent or invalid, payload cannot be audited
    if s1 is None and s2 is None and s3 is None:
        return {
            "status": "extraction_failed",
            "company_name": company_name,
            "source_file": data.get("source_file", "unknown"),
            "errors": ["Required GHG emissions data (Scope 1, 2, or 3) missing from disclosure payload."]
        }

    # Sum of known scopes
    s1_val = s1 if s1 is not None else 0.0
    s2_val = s2 if s2 is not None else 0.0
    s3_val = s3 if s3 is not None else 0.0
    calc_total = s1_val + s2_val + s3_val

    # Reconciliation determination:
    # If reported total is missing -> UNVERIFIED (earns 0 pts, cannot be tier A)
    # If reported total is present -> compare against calculated sum with tolerance
    if reported_total is None:
        reconciliation_status = "UNVERIFIED"
        variance = 0.0
        effective_total = calc_total
    else:
        variance = abs(reported_total - calc_total)
        tolerance = max(1.0, 0.01 * reported_total)
        if variance > tolerance:
            reconciliation_status = "FAILED"
            validation_warnings.append(
                f"GHG Protocol Mismatch: Reported total ({reported_total:,.1f} MT) does not reconcile "
                f"with Scope 1+2+3 sum ({calc_total:,.1f} MT). Variance: {variance:,.1f} MT."
            )
            effective_total = calc_total
        else:
            reconciliation_status = "PASSED"
            effective_total = reported_total

    # Scope 3 exceeding total check
    if s3 is not None and effective_total > 0 and s3 > effective_total:
        validation_warnings.append(f"Scope 3 emissions ({s3:,.1f} MT) exceed total GHG ({effective_total:,.1f} MT).")

    # Emissions targets (no hardcoded defaults)
    target_red = collector.number(emissions_raw.get("target_reduction_2030_pct"), "2030 Target Reduction %")
    achieved_yoy = collector.number(emissions_raw.get("achieved_reduction_yoy_pct"), "YoY Reduction %")
    collector.check_presence(achieved_yoy)

    # ----------------------------------------------------
    # Renewable Energy & Grid Mix
    # ----------------------------------------------------
    energy = data.get("renewable_energy") or {}
    tot_energy = collector.number(energy.get("total_mwh_consumed"), "Total MWh Consumed")
    ren_energy = collector.number(energy.get("renewable_mwh"), "Renewable MWh")
    ren_pct_raw = energy.get("renewable_share_pct")
    collector.check_presence(ren_pct_raw)

    if ren_pct_raw is not None:
        ren_pct = collector.number(ren_pct_raw, "Renewable Share %")
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
    water_withdrawn = collector.number(waste.get("total_water_withdrawal_m3"), "Water Withdrawal")
    water_rec_pct = collector.number(waste.get("water_recycled_pct"), "Water Recycled %")
    waste_div_pct = collector.number(waste.get("waste_diverted_from_landfill_pct"), "Waste Diversion %")
    waste_gen = collector.number(waste.get("waste_generated_tons"), "Waste Generated")
    collector.check_presence(waste_div_pct)

    if waste_div_pct is not None and (waste_div_pct < 0.0 or waste_div_pct > 100.0):
        validation_warnings.append(f"Waste diversion ({waste_div_pct}%) is outside valid 0-100% boundary.")

    if water_rec_pct is not None and (water_rec_pct < 0.0 or water_rec_pct > 100.0):
        validation_warnings.append(f"Water recycling ({water_rec_pct}%) is outside valid 0-100% boundary.")

    # Passthrough for zero liquid discharge
    zld_raw = waste.get("zero_liquid_discharge") if waste.get("zero_liquid_discharge") is not None else data.get("zero_liquid_discharge")
    zld = bool(zld_raw) if zld_raw is not None else None

    # ----------------------------------------------------
    # Social & Governance (No fabricated defaults)
    # ----------------------------------------------------
    gov = data.get("social_and_governance") or {}
    board_div = collector.number(gov.get("female_board_representation_pct"), "Female Board Diversity %")
    indep_dir = collector.number(gov.get("independent_directors_pct"), "Independent Directors %")
    pay_equity = collector.number(gov.get("gender_pay_equity_ratio"), "Gender Pay Equity Ratio")
    supplier_code = collector.number(gov.get("supplier_code_of_conduct_signoff_pct"), "Supplier Code Sign-off %")
    collector.check_presence(board_div)

    if board_div is not None and (board_div < 0.0 or board_div > 100.0):
        validation_warnings.append(f"Board gender diversity ({board_div}%) was outside 0-100% range.")

    if indep_dir is not None and (indep_dir < 0.0 or indep_dir > 100.0):
        validation_warnings.append(f"Independent directors ({indep_dir}%) was outside 0-100% range.")

    if supplier_code is not None and (supplier_code < 0.0 or supplier_code > 100.0):
        validation_warnings.append(f"Supplier code sign-off ({supplier_code}%) was outside 0-100% range.")

    # Passthrough for CSR spend % of net profit
    csr_spend_raw = gov.get("csr_spend_pct_net_profit") if gov.get("csr_spend_pct_net_profit") is not None else data.get("csr_spend_pct_net_profit")
    csr_spend = collector.number(csr_spend_raw, "CSR Spend %")

    # Data completeness computation
    denom = collector.evaluated if collector.evaluated > 0 else 1
    completeness_pct = round((collector.present / denom) * 100.0, 1) if collector.evaluated > 0 else 0.0

    return {
        "status": "success",
        "company_name": company_name,
        "reporting_year": data.get("reporting_year"),
        "country": data.get("country", ""),
        "jurisdiction": data.get("jurisdiction", ""),
        "standards": data.get("standards", []),
        "emissions_metric_tons_co2e": {
            "scope_1_direct": s1,
            "scope_2_indirect_market": s2,
            "scope_3_value_chain": s3,
            "total_ghg": effective_total,
            "reported_total_ghg": reported_total,
            "calculated_total_ghg": calc_total,
            "target_reduction_2030_pct": target_red,
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
            "waste_generated_tons": waste_gen,
            "waste_diverted_from_landfill_pct": waste_div_pct,
            "zero_liquid_discharge": zld
        },
        "social_and_governance": {
            "female_board_representation_pct": board_div,
            "gender_pay_equity_ratio": pay_equity,
            "independent_directors_pct": indep_dir,
            "supplier_code_of_conduct_signoff_pct": supplier_code,
            "csr_spend_pct_net_profit": csr_spend
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
