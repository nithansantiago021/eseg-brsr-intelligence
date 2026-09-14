"""
schema.py — Structured intermediate representation for extracted ESG metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ESGMetric:
    company: str
    reporting_year: str
    document: str
    document_type: str
    principle: str              # e.g. "P6", "" if not determinable
    section: str
    metric_name: str            # raw label as it appeared in the table
    normalized_metric_name: str # canonical name after synonym matching, or metric_name if none matched
    value: Optional[float]
    raw_value: str              # original cell text, e.g. "1,234.5"
    unit: str
    period: str                 # e.g. "FY2024-25", "" if unknown
    table_title: str
    page: int
    chunk_id: str
    is_calculated: bool = False # True if derived rather than directly reported
    derivation_note: str = ""   # required if is_calculated is True

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation_label(self) -> str:
        year = f" · {self.reporting_year}" if self.reporting_year else ""
        page = f" · p.{self.page}" if self.page else ""
        return f"{self.company}{year} · {self.document}{page}"

    @property
    def is_comparable_key(self) -> tuple:
        """Grouping key for cross-company / cross-year comparison charts."""
        return (self.normalized_metric_name, self.unit.lower().strip())
