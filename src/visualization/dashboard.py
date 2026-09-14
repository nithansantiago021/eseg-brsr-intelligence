"""
dashboard.py — Builds a principle-level (or company/year-scoped) dashboard
purely from the ESGMetric registry (src/metrics/extractor.py).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.metrics.schema import ESGMetric
from src.metrics.normalizer import units_compatible
from config.settings import cfg


@dataclass
class KPICard:
    label: str
    value: str
    unit: str
    company: str
    reporting_year: str
    citation: str


@dataclass
class DashboardData:
    kpi_cards: list
    grouped: dict          # normalized_metric_name -> list[ESGMetric]
    comparable_groups: dict  # normalized_metric_name -> list[ESGMetric] (>=2 pts, compatible units)
    table_rows: list       # flattened rows for a raw data table


def filter_registry(registry: list[ESGMetric], principle: str = "", companies: list = None,
                     years: list = None) -> list[ESGMetric]:
    out = registry
    if principle:
        out = [m for m in out if m.principle == principle]
    if companies:
        out = [m for m in out if m.company in companies]
    if years:
        out = [m for m in out if m.reporting_year in years]
    return out


def build_dashboard(registry: list[ESGMetric], principle: str = "", companies: list = None,
                     years: list = None) -> DashboardData:
    scoped = filter_registry(registry, principle, companies, years)

    grouped: dict[str, list[ESGMetric]] = defaultdict(list)
    for m in scoped:
        grouped[m.normalized_metric_name].append(m)

    kpi_cards = []
    comparable_groups = {}
    for name, items in grouped.items():
        # KPI cards: most recent/first value per (company) for a quick read
        seen_companies = set()
        for m in items:
            key = m.company or m.document
            if key in seen_companies:
                continue
            seen_companies.add(key)
            kpi_cards.append(KPICard(
                label=name, value=f"{m.value:,.2f}".rstrip("0").rstrip("."),
                unit=m.unit, company=m.company, reporting_year=m.reporting_year,
                citation=m.citation_label,
            ))

        if len(items) >= cfg.dashboard.min_metrics_for_chart:
            base_unit = items[0].unit
            if all(units_compatible(m.unit, base_unit) for m in items):
                comparable_groups[name] = items

    table_rows = [m.to_dict() for m in scoped]

    return DashboardData(
        kpi_cards=kpi_cards[:24],
        grouped=dict(grouped),
        comparable_groups=comparable_groups,
        table_rows=table_rows,
    )
