"""
chart_generator.py — Turns a natural-language question + the extracted
ESGMetric registry into a Plotly chart, when the question is comparable/
chart-shaped (e.g. "Compare Scope 1 and Scope 2 emissions for Company A
between FY2024 and FY2025"). 
"""

from __future__ import annotations

import re
from typing import Optional

import plotly.graph_objects as go

from src.metrics.schema import ESGMetric
from src.metrics.normalizer import units_compatible
from config.settings import cfg

_CHART_INTENT_RE = re.compile(
    r"\b(compare|comparison|chart|graph|visuali[sz]e|trend|plot|vs\.?|versus|"
    r"year[- ]on[- ]year|yoy|across (companies|years))\b", re.IGNORECASE,
)


def wants_chart(question: str) -> bool:
    return bool(_CHART_INTENT_RE.search(question))


def _matches_query(metric: ESGMetric, tokens: set[str]) -> bool:
    haystack = f"{metric.normalized_metric_name} {metric.metric_name}".lower()
    return any(t in haystack for t in tokens if len(t) > 2)


def find_relevant_metrics(question: str, registry: list[ESGMetric],
                           company: Optional[str] = None, reporting_year: Optional[str] = None) -> list[ESGMetric]:
    tokens = set(re.findall(r"[a-zA-Z0-9%]+", question.lower()))
    candidates = [m for m in registry if _matches_query(m, tokens)]
    if company:
        candidates = [m for m in candidates if m.company == company]
    if reporting_year:
        candidates = [m for m in candidates if m.reporting_year == reporting_year]
    return candidates


def build_chart(metrics: list[ESGMetric], chart_type: str = "bar") -> tuple[Optional[go.Figure], list[str], str]:
    """Returns (figure_or_None, citation_labels, refusal_reason)."""
    if len(metrics) < cfg.dashboard.min_metrics_for_chart:
        return None, [], "Not enough matching data points found in indexed tables to build a chart."

    # group by comparable key (normalized name + unit); refuse to mix incompatible units
    base_unit = metrics[0].unit
    incompatible = [m for m in metrics if not units_compatible(m.unit, base_unit)]
    if incompatible:
        return None, [], (
            f"Matched metrics use incompatible units ({base_unit!r} vs "
            f"{incompatible[0].unit!r}) — refusing to chart them together."
        )

    # x-axis: prefer company+year combo if varied, else just metric name
    companies = {m.company for m in metrics}
    years = {m.reporting_year for m in metrics}
    names = {m.normalized_metric_name for m in metrics}

    def _label(m: ESGMetric) -> str:
        parts = []
        if len(companies) > 1:
            parts.append(m.company)
        if len(years) > 1:
            parts.append(m.reporting_year or m.period)
        if len(names) > 1 and not parts:
            parts.append(m.normalized_metric_name)
        return " · ".join(p for p in parts if p) or m.normalized_metric_name

    x_labels = [_label(m) for m in metrics]
    y_values = [m.value for m in metrics]
    unit = metrics[0].unit

    fig = go.Figure()
    if chart_type == "line":
        fig.add_trace(go.Scatter(x=x_labels, y=y_values, mode="lines+markers", name=metrics[0].normalized_metric_name))
    elif chart_type == "pie":
        fig.add_trace(go.Pie(labels=x_labels, values=y_values))
    else:
        fig.add_trace(go.Bar(x=x_labels, y=y_values, name=metrics[0].normalized_metric_name))

    title = metrics[0].normalized_metric_name + (f" ({unit})" if unit else "")
    fig.update_layout(
        title=title, template="plotly_dark",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf0"), margin=dict(l=40, r=20, t=50, b=40),
        height=380,
    )

    citations = sorted({m.citation_label for m in metrics})
    return fig, citations, ""
