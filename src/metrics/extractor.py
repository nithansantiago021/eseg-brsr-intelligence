"""
extractor.py — Turns indexed table chunks (markdown tables produced by the
preprocessor) into structured ESGMetric records.
"""

from __future__ import annotations

import re
from src.metrics.schema import ESGMetric
from src.metrics.normalizer import normalize_metric_name, normalize_unit
from src.utils.logger import get_logger

log = get_logger(__name__)

_NUMBER_RE = re.compile(r"^[\s₹$]*-?[\d,]+(?:\.\d+)?\s*%?\s*$")
_UNIT_IN_LABEL_RE = re.compile(
    r"\(([^)]*(?:tco2e|co2|kl|m3|mwh|gj|kwh|gwh|%|percent|mt\b|kg|tonnes|hours|number|inr|₹|lakh|crore)[^)]*)\)",
    re.IGNORECASE,
)
_PERIOD_HEADER_RE = re.compile(r"(FY\s*\d{2,4}(?:-\d{2,4})?|20\d{2}(?:-\d{2,4})?|current year|previous year)", re.IGNORECASE)


def _parse_markdown_table(md: str) -> tuple[list[str], list[list[str]]]:
    lines = [l for l in md.split("\n") if l.strip().startswith("|")]
    if len(lines) < 3:
        return [], []
    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(cells)
    return headers, rows


def _clean_number(cell: str) -> tuple[float, str]:
    is_pct = "%" in cell
    num = re.sub(r"[^\d.\-]", "", cell)
    return (float(num) if num not in ("", "-", ".") else None), ("%" if is_pct else "")


def extract_metrics_from_table_chunk(chunk_text: str, chunk_meta: dict, chunk_id: str) -> list[ESGMetric]:
    headers, rows = _parse_markdown_table(chunk_text)
    if not headers or not rows:
        return []

    metrics: list[ESGMetric] = []
    label_col = 0
    for row in rows:
        if len(row) <= label_col:
            continue
        raw_label = row[label_col]
        if not raw_label or len(raw_label) < 3:
            continue

        unit_match = _UNIT_IN_LABEL_RE.search(raw_label)
        row_unit = unit_match.group(1).strip() if unit_match else ""
        clean_label = _UNIT_IN_LABEL_RE.sub("", raw_label).strip(" :-")

        for col_idx in range(label_col + 1, len(row)):
            cell = row[col_idx]
            if not cell or not _NUMBER_RE.match(cell):
                continue
            value, pct_unit = _clean_number(cell)
            if value is None:
                continue

            header = headers[col_idx] if col_idx < len(headers) else ""
            period_match = _PERIOD_HEADER_RE.search(header)
            period = period_match.group(1) if period_match else ""
            unit = pct_unit or row_unit or ""

            norm_name = normalize_metric_name(clean_label)
            metrics.append(ESGMetric(
                company=chunk_meta.get("company", ""),
                reporting_year=chunk_meta.get("reporting_year", ""),
                document=chunk_meta.get("filename", ""),
                document_type=chunk_meta.get("document_type", ""),
                principle=chunk_meta.get("principle", ""),
                section=chunk_meta.get("section", ""),
                metric_name=clean_label,
                normalized_metric_name=norm_name,
                value=value,
                raw_value=cell,
                unit=normalize_unit(unit),
                period=period,
                table_title=chunk_meta.get("table_title", ""),
                page=int(chunk_meta.get("page", "0").split("-")[0] or 0) if str(chunk_meta.get("page", "")).split("-")[0].isdigit() else 0,
                chunk_id=chunk_id,
            ))
    return metrics


def build_metric_registry(vector_store, where: dict | None = None) -> list[ESGMetric]:
    table_chunks = vector_store.get_table_chunks(where=where)
    registry: list[ESGMetric] = []
    for tc in table_chunks:
        try:
            registry.extend(extract_metrics_from_table_chunk(tc["text"], tc["metadata"], tc["id"]))
        except Exception as exc:
            log.warning("Metric extraction failed for chunk %s: %s", tc.get("id"), exc)
    log.info("Extracted %d ESG metrics from %d table chunks", len(registry), len(table_chunks))
    return registry
