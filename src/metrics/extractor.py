"""
extractor.py — Deterministic rule-based extraction tailored to Docling's 
key-value sentence format ("Label = Value") without any LLM calls.
"""

from __future__ import annotations

import re
from typing import Optional
from src.metrics.schema import ESGMetric
from src.metrics.normalizer import normalize_metric_name, normalize_unit
from src.utils.logger import get_logger

log = get_logger(__name__)

# Matches Docling key-value serialization: "Label = Value"
_KV_PAIR_RE = re.compile(r"(?:\d+,\s*)?([^=.\n]+)=\s*([^.\n]+)")

# Extracts floating numbers, integers, percentages, or negative values from cell values
_NUMERIC_RE = re.compile(r"-?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|-?\b\d+(?:\.\d+)?\b")

_UNIT_IN_LABEL_RE = re.compile(
    r"\(([^)]*(?:tco2e|co2|kl|m3|mwh|gj|kwh|gwh|%|percent|mt\b|kg|tonnes|hours|number|inr|₹|lakh|crore|nos|nos\.)[^)]*)\)",
    re.IGNORECASE,
)

_CURRENT_YEAR_RE = re.compile(
    r"(FY\s*20?25-26|FY\s*20?26|2025-26|current\s*year|current\s*financial\s*year|reporting\s*year)",
    re.IGNORECASE,
)

_PREVIOUS_YEAR_RE = re.compile(
    r"(FY\s*20?24-25|FY\s*20?25|2024-25|previous\s*year|previous\s*financial\s*year|prior\s*year)",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Strips non-printable bullet characters, markdown tags, and extra spaces."""
    cleaned = re.sub(r"[\t\r\n\xa0]", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_numeric_value(val_str: str) -> tuple[Optional[float], str]:
    """Extracts floating point values and percentage indicators."""
    text = _clean_text(val_str)
    if not text or text.lower() in ("-", "n/a", "nil", "nil.", "none", "not applicable", "--", ""):
        return None, ""

    is_pct = "%" in text or "percent" in text.lower()
    
    match = _NUMERIC_RE.search(text)
    if not match:
        return None, ""

    clean_str = match.group(0).replace(",", "")
    try:
        val = float(clean_str)
        return val, ("%" if is_pct else "")
    except ValueError:
        return None, ""


def extract_metrics_from_table_chunk(chunk_text: str, chunk_meta: dict, chunk_id: str) -> list[ESGMetric]:
    metrics: list[ESGMetric] = []
    default_year = chunk_meta.get("reporting_year", "FY2025-26")

    # 1. Parse Key-Value Pairs from Docling text format ("Metric Name = Value")
    matches = _KV_PAIR_RE.findall(chunk_text)
    
    for raw_label, raw_val in matches:
        clean_label = _clean_text(raw_label)
        clean_val_str = _clean_text(raw_val)

        if not clean_label or len(clean_label) < 2 or clean_label.lower() in ("s.no", "particulars", "parameter", "segment", "description of main activity"):
            continue

        value, pct_unit = _parse_numeric_value(clean_val_str)
        if value is None:
            continue

        # Extract unit embedded in label (e.g. "% of Turnover")
        unit_match = _UNIT_IN_LABEL_RE.search(clean_label)
        row_unit = unit_match.group(1).strip() if unit_match else ""
        if not row_unit and ("%" in clean_label or "percent" in clean_label.lower()):
            row_unit = "%"

        final_label = _UNIT_IN_LABEL_RE.sub("", clean_label).strip(" :-|")
        norm_name = normalize_metric_name(final_label)

        # Detect period (FY26 vs FY25) from label text
        inferred_period = default_year
        if _PREVIOUS_YEAR_RE.search(final_label):
            inferred_period = "FY2024-25"

        final_unit = pct_unit or row_unit or ""

        raw_page = str(chunk_meta.get("page", "0")).split("-")[0]
        page_num = int(raw_page) if raw_page.isdigit() else 0

        metrics.append(ESGMetric(
            company=chunk_meta.get("company", ""),
            reporting_year=inferred_period,
            document=chunk_meta.get("filename", ""),
            document_type=chunk_meta.get("document_type", "BRSR Report"),
            principle=chunk_meta.get("principle", ""),
            section=chunk_meta.get("section", ""),
            metric_name=final_label,
            normalized_metric_name=norm_name,
            value=value,
            raw_value=clean_val_str,
            unit=normalize_unit(final_unit),
            period=inferred_period,
            table_title=chunk_meta.get("table_title", ""),
            page=page_num,
            chunk_id=chunk_id,
        ))

    return metrics


def build_metric_registry(vector_store, where: dict | None = None) -> list[ESGMetric]:
    table_chunks = vector_store.get_table_chunks(where=where)
    registry: list[ESGMetric] = []
    
    for tc in table_chunks:
        try:
            extracted = extract_metrics_from_table_chunk(tc["text"], tc["metadata"], tc["id"])
            registry.extend(extracted)
        except Exception as exc:
            log.warning("Metric extraction failed for chunk %s: %s", tc.get("id"), exc)
            
    log.info("Extracted %d ESG metrics deterministically from %d table chunks", len(registry), len(table_chunks))
    return registry
