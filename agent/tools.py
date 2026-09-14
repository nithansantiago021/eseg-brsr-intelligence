"""
tools.py — The Compliance Sentinel agent's tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from strands import tool

from config.settings import cfg
from src.embedding.embedder import Embedder, VectorStore
from src.embedding.indexer import index_file
from src.retrieval.retriever import Retriever
from src.evaluation.ragas_guard import RAGASGuard
from src.rag.rag_pipeline import RAGPipeline
from src.metrics.extractor import build_metric_registry
from src.metrics.normalizer import units_compatible
from src.metrics.schema import ESGMetric
from agent.history_store import FilingHistoryStore


@dataclass
class AgentRuntime:
    """Everything the tool functions need for one compliance-check run.
    One AgentRuntime = one filing under review; the vector store is scoped
    to just that filing, kept in memory only for the duration of the run."""
    embedder: Embedder
    vector_store: VectorStore
    retriever: Retriever
    pipeline: RAGPipeline
    history: FilingHistoryStore
    api_key: str
    company: str = ""
    reporting_year: str = ""
    document_type: str = ""
    filename: str = ""
    rag_ask_budget: int = 10
    rag_ask_calls_made: int = 0


_runtime: Optional[AgentRuntime] = None


def configure_runtime(runtime: AgentRuntime) -> None:
    global _runtime
    _runtime = runtime


def _require_runtime() -> AgentRuntime:
    if _runtime is None:
        raise RuntimeError("Agent runtime not configured — call configure_runtime() before running the agent.")
    return _runtime


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@tool
def index_filing(file_path: str, dual_page_layout: bool = False) -> dict:
    rt = _require_runtime()
    chunks = index_file(
        Path(file_path), rt.embedder, rt.vector_store,
        company=rt.company, reporting_year=rt.reporting_year, document_type=rt.document_type,
        dual_page_layout=dual_page_layout,
    )
    principles = sorted({c.principle for c in chunks if c.principle})
    n_tables = sum(1 for c in chunks if c.content_type == "table")
    return {
        "filename": rt.filename,
        "total_chunks": len(chunks),
        "table_chunks": n_tables,
        "principles_detected": principles,
    }


# ---------------------------------------------------------------------------
# Disclosure completeness
# ---------------------------------------------------------------------------

@tool
def check_disclosure_completeness() -> dict:
    """Check which of the 9 BRSR principles (P1-P9) have zero detected
    content in the indexed filing — a candidate signal for a missing or
    very thin disclosure. Call this after index_filing.

    Returns:
        A dict with 'principles_present' and 'principles_missing' (using
        the standard P1..P9 codes), plus their human-readable names.
    """
    rt = _require_runtime()
    present = set(rt.vector_store.list_principles())
    all_codes = set(cfg.esg.principles.keys())
    missing = sorted(all_codes - present)
    return {
        "principles_present": [f"{p} — {cfg.esg.principles[p]}" for p in sorted(present)],
        "principles_missing": [f"{p} — {cfg.esg.principles[p]}" for p in missing],
    }


# ---------------------------------------------------------------------------
# Metric snapshot + year-over-year comparison
# ---------------------------------------------------------------------------

@tool
def get_current_metric_snapshot() -> list[dict]:
    rt = _require_runtime()
    registry = build_metric_registry(rt.vector_store)
    rt.history.save_filing_snapshot(
        company=rt.company, reporting_year=rt.reporting_year, document_type=rt.document_type,
        filename=rt.filename, metrics=[m.to_dict() for m in registry],
        principles_present=rt.vector_store.list_principles(),
    )
    return [m.to_dict() for m in registry]


@tool
def compare_year_over_year() -> dict:
    rt = _require_runtime()
    prior = rt.history.get_prior_filing(rt.company, exclude_year=rt.reporting_year)
    if prior is None:
        return {"prior_filing": None, "comparisons": [], "not_comparable": [],
                "note": "No prior filing on record for this company — nothing to compare yet."}

    current_registry = build_metric_registry(rt.vector_store)
    current_by_name: dict[str, list[ESGMetric]] = {}
    for m in current_registry:
        current_by_name.setdefault(m.normalized_metric_name, []).append(m)

    comparisons, not_comparable = [], []
    for prior_m in prior["metrics"]:
        name = prior_m["normalized_metric_name"]
        matches = current_by_name.get(name, [])
        if not matches:
            continue
        cur = matches[0]
        if not units_compatible(prior_m["unit"], cur.unit):
            not_comparable.append({
                "metric": name, "prior_unit": prior_m["unit"], "current_unit": cur.unit,
            })
            continue
        prior_val, cur_val = prior_m["value"], cur.value
        pct_change = None
        if prior_val not in (None, 0) and cur_val is not None:
            pct_change = round(((cur_val - prior_val) / abs(prior_val)) * 100, 1)
        comparisons.append({
            "metric": name, "unit": cur.unit,
            "prior_year": prior["reporting_year"], "prior_value": prior_val,
            "current_year": rt.reporting_year, "current_value": cur_val,
            "pct_change": pct_change, "citation": cur.citation_label,
        })

    return {
        "prior_filing": {"reporting_year": prior["reporting_year"], "filename": prior["filename"]},
        "comparisons": comparisons,
        "not_comparable": not_comparable,
    }


# ---------------------------------------------------------------------------
# Grounded Q&A (for verifying anything ambiguous before flagging it)
# ---------------------------------------------------------------------------

@tool
def rag_ask(question: str) -> dict:
    rt = _require_runtime()
    if rt.rag_ask_calls_made >= rt.rag_ask_budget:
        return {
            "answer": "",
            "is_answerable": False,
            "faithfulness": 0.0,
            "ragas_passed": False,
            "citations": [],
            "calls_remaining": 0,
            "budget_exhausted": True,
            "note": (
                f"rag_ask budget ({rt.rag_ask_budget} calls) is used up for this run. "
                "Do not call rag_ask again — write the memo using the metric values "
                "already extracted, and note in the memo which items could not be "
                "individually verified due to the verification budget."
            ),
        }

    rt.rag_ask_calls_made += 1
    resp = rt.pipeline.run(question=question, api_key=rt.api_key, company=rt.company,
                            reporting_year=rt.reporting_year)
    return {
        "answer": resp.answer,
        "is_answerable": resp.is_answerable,
        "faithfulness": resp.faithfulness_score,
        "ragas_passed": resp.ragas_passed,
        "citations": [f"{c.filename} p.{c.page}" for c in resp.sources[:5]],
        "calls_remaining": rt.rag_ask_budget - rt.rag_ask_calls_made,
    }


# ---------------------------------------------------------------------------
# Findings, memo, and human escalation
# ---------------------------------------------------------------------------

@tool
def flag_finding(topic: str, severity: str, description: str, citation: str = "") -> str:
    rt = _require_runtime()
    if severity not in ("info", "review_suggested", "action_required"):
        severity = "review_suggested"
    rt.history.add_finding(rt.company, rt.reporting_year, topic, severity, description, citation)
    return f"Recorded [{severity}] finding on '{topic}'."


@tool
def save_review_memo(memo_markdown: str, status: str) -> str:
    rt = _require_runtime()
    if status not in ("auto_closed", "pending_human_review"):
        status = "pending_human_review"
    return rt.history.save_memo(rt.company, rt.reporting_year, memo_markdown, status)


@tool
def notify_human(severity: str, summary: str) -> str:
    rt = _require_runtime()
    rt.history.add_notification(rt.company, rt.reporting_year, severity, summary)
    return f"Notified human reviewer ({severity}): {summary}"
