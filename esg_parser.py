"""
ESG document parser and normalization engine.

Converts JSON / PDF / DOCX ESG disclosures into one normalized schema.
Rules enforced here:
  * Never invent numbers. Missing values stay None.
  * Anomalies (negative, NaN, out of range) become validation warnings.
  * Documents without usable emissions data are refused ("extraction_failed").
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # reject larger uploads
MAX_DOCX_XML_BYTES = 50 * 1024 * 1024  # reject zip bombs inside DOCX
RECONCILIATION_TOLERANCE_PCT = 0.01
RECONCILIATION_MIN_TOLERANCE_MT = 1.0
KNOWN_STANDARDS = ("GRI", "TCFD", "SASB", "SBTi", "BRSR", "CSRD", "ISSB")
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Optional dependency: IBM Docling
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except (ImportError, OSError):
    DocumentConverter = None
    DOCLING_AVAILABLE = False

# Optional dependency: pypdf (preferred) or PyPDF2
try:
    from pypdf import PdfReader
    PDF_PARSER_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader  # type: ignore[no-redef]
        PDF_PARSER_AVAILABLE = True
    except ImportError:
        PdfReader = None  # type: ignore[assignment,misc]
        PDF_PARSER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def parse_document(file_source: Any, filename: Optional[str] = None) -> dict[str, Any]:
    """Parse an ESG document into normalized structured data.

    Accepts a dict, a file path (str), or a file-like object with ``read()``.
    Raises FileNotFoundError for a missing path and TypeError for other types.
    Unreadable content returns {"status": "extraction_failed", ...}.
    """
    if isinstance(file_source, dict):
        if file_source.get("status") == "extraction_failed":
            return file_source
        return validate_and_normalize_esg(file_source)
    if hasattr(file_source, "read"):
        return _parse_stream(file_source, filename)
    if isinstance(file_source, str):
        return _parse_path(file_source)
    raise TypeError(f"Unsupported input type: {type(file_source)}")


def _failure(file_name: str, errors: list[str]) -> dict[str, Any]:
    """Builds a standard extraction_failed result."""
    return {
        "status": "extraction_failed",
        "company_name": _clean_title(file_name),
        "source_file": file_name,
        "errors": errors,
    }


def _clean_title(file_name: str) -> str:
    """Turns 'my-report_2024.pdf' into 'My Report 2024'."""
    stem = os.path.splitext(os.path.basename(file_name))[0]
    return stem.replace("_", " ").replace("-", " ").title()


MAX_JSON_DEPTH = 15


def _check_depth(obj: Any, depth: int = 0) -> bool:
    """Returns True if object depth is within acceptable limit."""
    if depth > MAX_JSON_DEPTH:
        return False
    if isinstance(obj, dict):
        return all(_check_depth(v, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_check_depth(v, depth + 1) for v in obj)
    return True


def _parse_json_bytes(content: bytes, file_name: str) -> dict[str, Any]:
    """Decodes and normalizes JSON bytes; returns extraction_failed on bad input."""
    try:
        data = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _failure(file_name, [f"Malformed JSON document: {exc}"])
    if not isinstance(data, dict):
        return _failure(file_name, ["JSON root must be an object, not a list or scalar."])
    if not _check_depth(data):
        return _failure(file_name, ["JSON document exceeds maximum nesting depth limit."])
    return validate_and_normalize_esg(data)


def _parse_stream(stream: Any, filename: Optional[str]) -> dict[str, Any]:
    """Parses an uploaded file-like object (e.g. Streamlit UploadedFile)."""
    file_name = getattr(stream, "name", None) or filename or "uploaded_disclosure.json"
    ext = os.path.splitext(file_name)[1].lower()
    if hasattr(stream, "seek"):
        stream.seek(0)
    content = stream.read()
    if isinstance(content, str):
        content = content.encode("utf-8")
    if len(content) > MAX_UPLOAD_BYTES:
        return _failure(file_name, [f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."])
    if ext in (".pdf", ".docx"):
        return _parse_pdf_or_docx_bytes(content, file_name)
    if ext == ".json":
        return _parse_json_bytes(content, file_name)
    if ext in (".txt", ".text"):
        return extract_esg_from_text(content.decode("utf-8", errors="replace"), file_name)
    return _failure(file_name, [f"Unsupported file format '{ext}'. Expected .json, .pdf, .docx, or .txt"])


def _parse_path(path: str) -> dict[str, Any]:
    """Parses a document from a filesystem path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ESG document not found at: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".json", ".pdf", ".docx", ".txt", ".text"):
        return _failure(path, [f"Unsupported file extension '{ext}'. Supported: .json, .pdf, .docx, .txt"])
    if os.path.getsize(path) > MAX_UPLOAD_BYTES:
        return _failure(path, [f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."])
    with open(path, "rb") as handle:
        content = handle.read()
    if ext == ".json":
        return _parse_json_bytes(content, path)
    if ext in (".txt", ".text"):
        return extract_esg_from_text(content.decode("utf-8", errors="replace"), os.path.basename(path))
    return _parse_pdf_or_docx_bytes(content, os.path.basename(path))


# ---------------------------------------------------------------------------
# PDF / DOCX text extraction
# ---------------------------------------------------------------------------
def _extract_text_from_pdf_stream(stream: io.BytesIO) -> str:
    """Extracts text from a PDF using pypdf/PyPDF2. Returns '' on any failure."""
    if not PDF_PARSER_AVAILABLE or PdfReader is None:
        return ""
    try:
        reader = PdfReader(stream)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        return "\n".join(pages_text)
    except Exception as exc:
        logger.warning("PDF text extraction failed: %s", exc)
        return ""


def _extract_text_from_docx_stream(stream: io.BytesIO) -> str:
    """Extracts paragraph text from a DOCX using only the standard library.

    Rejects XML containing DTD/entity declarations (entity-expansion attacks)
    and oversized parts (zip bombs).
    """
    try:
        with zipfile.ZipFile(stream) as docx:
            if "word/document.xml" not in docx.namelist():
                return ""
            if docx.getinfo("word/document.xml").file_size > MAX_DOCX_XML_BYTES:
                logger.warning("DOCX document.xml exceeds size limit")
                return ""
            xml_content = docx.read("word/document.xml")
        if b"<!DOCTYPE" in xml_content or b"<!ENTITY" in xml_content:
            logger.warning("DOCX contains a DTD/entity declaration; refusing to parse")
            return ""
        root = ET.fromstring(xml_content)
    except (zipfile.BadZipFile, ET.ParseError, KeyError) as exc:
        logger.warning("DOCX read failed: %s", exc)
        return ""
    paragraphs = []
    for paragraph in root.iter(f"{{{WORD_NS}}}p"):
        texts = [node.text for node in paragraph.iter(f"{{{WORD_NS}}}t") if node.text]
        if texts:
            paragraphs.append("".join(texts))
    return "\n".join(paragraphs)


def _convert_with_docling(file_bytes: bytes, ext: str) -> Optional[str]:
    """Returns markdown from Docling, or None if Docling is unavailable or fails."""
    if not DOCLING_AVAILABLE or DocumentConverter is None:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            tmp.write(file_bytes)
        result = DocumentConverter().convert(tmp_path)
        return str(result.document.export_to_markdown())
    except Exception as exc:
        logger.warning("Docling conversion failed: %s", exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Could not delete temp file %s", tmp_path)


def _parse_pdf_or_docx_bytes(file_bytes: bytes, file_name: str) -> dict[str, Any]:
    """Parses PDF/DOCX bytes: Docling first, then pypdf / zipfile fallback."""
    ext = os.path.splitext(file_name)[1].lower()
    markdown = _convert_with_docling(file_bytes, ext)
    if markdown is not None:
        return extract_esg_from_text(markdown, file_name)

    stream = io.BytesIO(file_bytes)
    if ext == ".pdf":
        text = _extract_text_from_pdf_stream(stream)
        failure_message = (
            "PDF extraction failed: Could not read textual content from the PDF. "
            "Scanned or complex-table PDFs require IBM Docling ('pip install docling') "
            "or pypdf ('pip install pypdf')."
        )
    elif ext == ".docx":
        text = _extract_text_from_docx_stream(stream)
        failure_message = (
            "DOCX extraction failed: Could not read 'word/document.xml'. "
            "File may be corrupted, encrypted, or not a valid Word document."
        )
    else:
        return _failure(file_name, [f"Unsupported file format '{ext}'. Expected .pdf or .docx."])

    if not text.strip():
        return _failure(file_name, [failure_message])
    return extract_esg_from_text(text, file_name)


# ---------------------------------------------------------------------------
# Text -> structured extraction
# ---------------------------------------------------------------------------
_NUMBER = r"(\d[\d,]*(?:\.\d+)?)"


def _find_number(pattern: str, text: str) -> Optional[float]:
    """Returns the first finite number captured by ``pattern`` or None."""
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        val = float(match.group(1).replace(",", ""))
        return val if math.isfinite(val) else None
    except ValueError:
        return None


def extract_esg_from_text(text: str, source_name: str) -> dict[str, Any]:
    """Scans document text for ESG disclosures using flexible synonym matching.

    Refuses to invent numbers: if required disclosures are missing, returns
    an extraction_failed result listing what could not be found.
    """
    clean_text = text.replace("&amp;", "&")
    errors: list[str] = []
    emissions: dict[str, Any] = {}
    energy: dict[str, Any] = {}
    water_waste: dict[str, Any] = {}
    governance: dict[str, Any] = {}

    company = re.search(
        r"(?:Company(?:\s*Name)?|Entity|Corporation|Organization)\s*[:\-]\s*([^\n\r]+)", clean_text, re.IGNORECASE
    )
    year = re.search(r"(?:FY|fiscal\s*year|reporting\s*year|reporting\s*period)\s*[:\-]?\s*(?:FY)?(20\d{2})", clean_text, re.IGNORECASE)

    # Scope 1 (Direct Operational Emissions)
    scope_1 = _find_number(
        r"(?:Scope\s*1(?:\s*direct)?(?:\s*GHG)?\s*emissions?|Direct\s*Operational\s*Emissions(?:\s*\(Scope\s*1\))?|Direct\s*(?:GHG)?\s*emissions(?:\s*\(Scope\s*1\))?)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    if scope_1 is None:
        scope_1 = _find_number(r"(?:Scope\s*1|Direct\s*Emissions)\s*[:\-]?\s*" + _NUMBER, clean_text)

    # Scope 2 (Indirect / Purchased Electricity)
    scope_2 = _find_number(
        r"(?:Scope\s*2(?:\s*indirect)?(?:\s*GHG)?\s*emissions?|Purchased\s*Electricity\s*Emissions|Purchased\s*Energy\s*Emissions|Indirect\s*(?:GHG)?\s*emissions(?:\s*\(Scope\s*2\))?)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    if scope_2 is None:
        scope_2 = _find_number(r"(?:Scope\s*2|Purchased\s*Electricity)\s*[:\-]?\s*" + _NUMBER, clean_text)

    # Scope 3 (Value Chain / Supply Chain)
    scope_3 = _find_number(
        r"(?:Scope\s*3(?:\s*value\s*chain)?\s*emissions?|Value\s*Chain\s*Emissions|Supply\s*Chain\s*Emissions)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    if scope_3 is None:
        scope_3 = _find_number(r"Scope\s*3\s*[:\-]?\s*" + _NUMBER, clean_text)

    # Total GHG / Total Carbon Footprint
    total_ghg = _find_number(
        r"(?:Total\s*GHG\s*footprint|Total\s*Carbon\s*Footprint|Total\s*GHG\s*emissions|Total\s*emissions)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )

    # SBTi 2030 Target & YoY progress
    target_2030 = _find_number(
        r"(?:2030\s*Science[\s\-]Based\s*Target(?:\s*\(SBTi\))?|SBTi\s*2030\s*Target|2030\s*Target)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    yoy_reduction = _find_number(
        r"(?:YoY\s*progress\s*achieved|YoY\s*Carbon\s*Reduction|YoY\s*emissions?\s*reduction|Year[\s\-]over[\s\-]Year\s*reduction)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )

    # Energy: Total consumption, renewable sourced, and renewable share / clean power ratio
    total_mwh = _find_number(
        r"(?:Total\s*electricity\s*consumed|Total\s*Energy\s*Usage|Total\s*Energy\s*Consumption|Total\s*power\s*consumed)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    renewable_mwh = _find_number(
        r"(?:Renewable\s*electricity\s*sourced|Renewable\s*energy\s*sourced|Renewable\s*power\s*consumed|Clean\s*electricity\s*consumed)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    renewable_share = _find_number(
        r"(?:Renewable\s*energy\s*share|Clean\s*Power\s*Ratio|Renewable\s*electricity\s*share|Renewable\s*share|Green\s*power\s*ratio)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    re100_match = re.search(r"RE100\s*(?:Pledged|Committed|Member)?\s*[:\-]?\s*(Yes|True)", clean_text, re.IGNORECASE)

    # Water & Waste: Withdrawal, recycled / recovery, diversion from landfill
    water_withdrawal = _find_number(
        r"(?:Total\s*water\s*withdrawal|Water\s*withdrawal|Total\s*water\s*usage)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    water_recycled = _find_number(
        r"(?:Water\s*recycled\s*(?:percentage|share|ratio)?|Wastewater\s*Recovery\s*Rate|Water\s*recycling\s*rate)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    waste_diverted = _find_number(
        r"(?:Waste\s*diverted\s*from\s*landfill|Landfill\s*Diversion\s*Rate|Waste\s*diversion\s*rate)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )

    # Social & Governance: Female board rep, independent directors, supplier code
    female_board = _find_number(
        r"(?:Female\s*board\s*representation|Female\s*Representation\s*on\s*Board|Women\s*on\s*board|Board\s*gender\s*diversity)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    independent_directors = _find_number(
        r"(?:Independent\s*directors\s*(?:share|percentage|ratio)?|Board\s*Independence\s*Ratio|Independent\s*board\s*ratio)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )
    supplier_code = _find_number(
        r"(?:Supplier\s*Code\s*Sign[\s\-]off|Supplier\s*code\s*of\s*conduct\s*sign[\s\-]off|Vendor\s*code\s*sign[\s\-]off)\s*[:\-]?\s*" + _NUMBER,
        clean_text
    )

    # Validate essential requirements
    if scope_1 is None:
        errors.append("Scope 1 direct GHG emissions could not be identified in disclosure text")
    else:
        emissions["scope_1_direct"] = scope_1
    if scope_2 is None:
        errors.append("Scope 2 indirect GHG emissions could not be identified in disclosure text")
    else:
        emissions["scope_2_indirect_market"] = scope_2
    if scope_3 is not None:
        emissions["scope_3_value_chain"] = scope_3
    if total_ghg is not None:
        emissions["total_ghg"] = total_ghg
    if target_2030 is not None:
        emissions["target_reduction_2030_pct"] = target_2030
    if yoy_reduction is not None:
        emissions["achieved_reduction_yoy_pct"] = yoy_reduction

    # Renewable energy compilation
    if total_mwh is not None:
        energy["total_mwh_consumed"] = total_mwh
    if renewable_mwh is not None:
        energy["renewable_mwh"] = renewable_mwh
    if renewable_share is not None:
        energy["renewable_share_pct"] = renewable_share
    elif total_mwh and total_mwh > 0 and renewable_mwh is not None:
        energy["renewable_share_pct"] = round(renewable_mwh / total_mwh * 100.0, 2)
    if re100_match is not None:
        energy["re100_committed"] = True

    if renewable_share is None and energy.get("renewable_share_pct") is None:
        errors.append("Renewable energy share percentage could not be identified in disclosure text")

    # Water and waste compilation
    if water_withdrawal is not None:
        water_waste["total_water_withdrawal_m3"] = water_withdrawal
    if water_recycled is not None:
        water_waste["water_recycled_pct"] = water_recycled
    if waste_diverted is not None:
        water_waste["waste_diverted_from_landfill_pct"] = waste_diverted

    # Governance compilation
    if female_board is not None:
        governance["female_board_representation_pct"] = female_board
    if independent_directors is not None:
        governance["independent_directors_pct"] = independent_directors
    if supplier_code is not None:
        governance["supplier_code_of_conduct_signoff_pct"] = supplier_code

    if len(errors) >= 2 or not emissions:
        return {
            "status": "extraction_failed",
            "company_name": company.group(1).strip() if company else _clean_title(source_name),
            "source_file": source_name,
            "errors": errors,
        }

    extracted: dict[str, Any] = {
        "company_name": company.group(1).strip() if company else _clean_title(source_name),
        "reporting_year": int(year.group(1)) if year else None,
        "standards": [name for name in KNOWN_STANDARDS if name.lower() in clean_text.lower()],
        "emissions_metric_tons_co2e": emissions,
        "renewable_energy": energy,
        "water_and_waste": water_waste,
        "social_and_governance": governance,
    }
    return validate_and_normalize_esg(extracted)


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------
class _Collector:
    """Collects validation warnings and field-presence counts for one document."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.evaluated = 0
        self.present = 0

    def number(self, raw: Any, label: str, count: bool = False) -> Optional[float]:
        """Converts ``raw`` to a finite float or None, recording warnings.

        If ``count`` is True the field counts toward the completeness score,
        and counts as present only when it parsed to a usable number.
        """
        value = self._convert(raw, label)
        if count:
            self.evaluated += 1
            if value is not None:
                self.present += 1
        return value

    def _convert(self, raw: Any, label: str) -> Optional[float]:
        if raw is None:
            return None
        if isinstance(raw, bool):
            self.warnings.append(f"{label} contains non-numeric value '{raw}'.")
            return None
        try:
            value = float(raw)
        except (ValueError, TypeError):
            self.warnings.append(f"{label} contains non-numeric value '{raw}'.")
            return None
        if not math.isfinite(value):
            self.warnings.append(f"{label} is not a finite number ('{raw}').")
            return None
        if value < 0.0:
            self.warnings.append(f"{label} is negative ({value}); expected non-negative value.")
        return value

    def check_percent(self, value: Optional[float], label: str) -> None:
        """Warns if a percentage lies outside 0-100."""
        if value is not None and not 0.0 <= value <= 100.0:
            self.warnings.append(f"{label} ({value}%) is outside valid 0-100% boundary.")


def _normalize_emissions(emissions: dict[str, Any], col: _Collector) -> dict[str, Any]:
    """Validates Scope 1-3, reconciles against the reported total."""
    scope_1 = col.number(emissions.get("scope_1_direct"), "Scope 1", count=True)
    scope_2 = col.number(emissions.get("scope_2_indirect_market"), "Scope 2", count=True)
    scope_3 = col.number(emissions.get("scope_3_value_chain"), "Scope 3", count=True)
    reported = col.number(emissions.get("total_ghg"), "Reported Total GHG", count=True)
    calculated = (scope_1 or 0.0) + (scope_2 or 0.0) + (scope_3 or 0.0)

    status = "UNVERIFIED"
    variance = 0.0
    if reported is not None:
        variance = abs(reported - calculated)
        tolerance = max(RECONCILIATION_MIN_TOLERANCE_MT, RECONCILIATION_TOLERANCE_PCT * reported)
        status = "PASSED" if variance <= tolerance else "FAILED"
        if status == "FAILED":
            col.warnings.append(
                f"GHG Protocol Mismatch: Reported total ({reported:,.1f} MT) does not reconcile "
                f"with Scope 1+2+3 sum ({calculated:,.1f} MT). Variance: {variance:,.1f} MT."
            )
    else:
        col.warnings.append("Reported total GHG missing; Scope 1+2+3 reconciliation not possible.")

    effective_total = reported if status == "PASSED" else calculated
    if scope_3 is not None and effective_total is not None and 0 < effective_total < scope_3:
        col.warnings.append(
            f"Scope 3 emissions ({scope_3:,.1f} MT) exceed total GHG ({effective_total:,.1f} MT)."
        )
    target = col.number(emissions.get("target_reduction_2030_pct"), "2030 Target Reduction %")
    yoy = col.number(emissions.get("achieved_reduction_yoy_pct"), "YoY Reduction %", count=True)
    return {
        "section": {
            "scope_1_direct": scope_1,
            "scope_2_indirect_market": scope_2,
            "scope_3_value_chain": scope_3,
            "total_ghg": effective_total,
            "reported_total_ghg": reported,
            "calculated_total_ghg": calculated,
            "target_reduction_2030_pct": target,
            "achieved_reduction_yoy_pct": yoy,
        },
        "quality": {
            "reconciliation_status": status,
            "reported_total_ghg": reported,
            "calculated_total_ghg": calculated,
            "total_ghg_variance": variance,
        },
        "has_scope_data": any(v is not None for v in (scope_1, scope_2, scope_3)),
    }


def _normalize_energy(energy: dict[str, Any], col: _Collector) -> dict[str, Any]:
    """Validates renewable energy fields; derives the share if only MWh given."""
    total = col.number(energy.get("total_mwh_consumed"), "Total MWh Consumed")
    renewable = col.number(energy.get("renewable_mwh"), "Renewable MWh")
    share = col.number(energy.get("renewable_share_pct"), "Renewable Share %", count=True)
    if share is None and total and total > 0 and renewable is not None:
        share = round(renewable / total * 100.0, 2)
    col.check_percent(share, "Renewable share")
    if total is not None and renewable is not None and renewable > total:
        col.warnings.append(f"Renewable energy ({renewable} MWh) exceeds total consumed ({total} MWh).")
    return {
        "total_mwh_consumed": total,
        "renewable_mwh": renewable,
        "renewable_share_pct": share,
        "re100_committed": bool(energy.get("re100_committed", False)),
    }


def _normalize_water_waste(waste: dict[str, Any], col: _Collector) -> dict[str, Any]:
    """Validates water and waste fields."""
    withdrawn = col.number(waste.get("total_water_withdrawal_m3"), "Water Withdrawal")
    recycled = col.number(waste.get("water_recycled_pct"), "Water Recycled %")
    generated = col.number(waste.get("waste_generated_tons"), "Waste Generated")
    diverted = col.number(waste.get("waste_diverted_from_landfill_pct"), "Waste Diversion %", count=True)
    col.check_percent(diverted, "Waste diversion")
    col.check_percent(recycled, "Water recycling")
    zld = waste.get("zero_liquid_discharge")
    return {
        "total_water_withdrawal_m3": withdrawn,
        "water_recycled_pct": recycled,
        "waste_generated_tons": generated,
        "waste_diverted_from_landfill_pct": diverted,
        "zero_liquid_discharge": zld if isinstance(zld, bool) else None,
    }


def _normalize_governance(gov: dict[str, Any], col: _Collector) -> dict[str, Any]:
    """Validates social and governance fields. Missing values stay None."""
    board = col.number(gov.get("female_board_representation_pct"), "Female Board Diversity %", count=True)
    independent = col.number(gov.get("independent_directors_pct"), "Independent Directors %")
    pay_equity = col.number(gov.get("gender_pay_equity_ratio"), "Gender Pay Equity Ratio")
    supplier = col.number(gov.get("supplier_code_of_conduct_signoff_pct"), "Supplier Code Sign-off %")
    csr = col.number(gov.get("csr_spend_pct_net_profit"), "CSR Spend % of Net Profit")
    col.check_percent(board, "Board gender diversity")
    col.check_percent(independent, "Independent directors")
    col.check_percent(supplier, "Supplier code sign-off")
    return {
        "female_board_representation_pct": board,
        "gender_pay_equity_ratio": pay_equity,
        "independent_directors_pct": independent,
        "supplier_code_of_conduct_signoff_pct": supplier,
        "csr_spend_pct_net_profit": csr,
    }


def validate_and_normalize_esg(data: dict[str, Any]) -> dict[str, Any]:
    """Validates ESG metrics and reconciles GHG totals.

    Anomalies are kept and reported as warnings (never silently clamped).
    Missing values stay None. A payload with no Scope 1/2/3 data is refused.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid ESG payload: expected JSON object.")
    if data.get("status") == "extraction_failed":
        return data

    col = _Collector()
    emissions = _normalize_emissions(data.get("emissions_metric_tons_co2e") or {}, col)
    if not emissions["has_scope_data"]:
        return {
            "status": "extraction_failed",
            "company_name": data.get("company_name", "Unknown Entity"),
            "errors": ["No Scope 1, 2 or 3 emissions were disclosed; cannot audit."],
        }
    energy = _normalize_energy(data.get("renewable_energy") or {}, col)
    water_waste = _normalize_water_waste(data.get("water_and_waste") or {}, col)
    governance = _normalize_governance(data.get("social_and_governance") or {}, col)

    completeness = round(col.present / col.evaluated * 100.0, 1) if col.evaluated else 0.0
    quality = dict(emissions["quality"])
    quality["completeness_pct"] = completeness
    quality["validation_warnings"] = col.warnings
    return {
        "status": "success",
        "company_name": data.get("company_name", "Unknown Entity"),
        "reporting_year": data.get("reporting_year"),
        "country": data.get("country", ""),
        "jurisdiction": data.get("jurisdiction", "Not disclosed"),
        "standards": data.get("standards", []),
        "emissions_metric_tons_co2e": emissions["section"],
        "renewable_energy": energy,
        "water_and_waste": water_waste,
        "social_and_governance": governance,
        "data_quality": quality,
    }
