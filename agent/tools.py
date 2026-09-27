"""
tools.py — The Compliance Sentinel agent's tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    """Everything the tool functions need for one compliance-check run."""
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
        Path(file_path), rt.embedder, rt.vector_store, retriever=rt.retriever,
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
# Disclosure Completeness Verification
# ---------------------------------------------------------------------------

_PRINCIPLE_SEARCH_TERMS = {
    "P1": ["PRINCIPLE 1", "Ethics", "TCoC", "Whistleblower", "Anti Bribery", "PoSH"],
    "P2": ["PRINCIPLE 2", "Sustainable", "Safe Goods", "LCA", "Resource Efficiency"],
    "P3": ["PRINCIPLE 3", "Employee Well-being", "Safety", "Hazard", "Training"],
    "P4": ["PRINCIPLE 4", "Stakeholder Engagement", "Consultation", "Community"],
    "P5": ["PRINCIPLE 5", "Human Rights", "Minimum Wages", "Child Labour"],
    "P6": ["PRINCIPLE 6", "Environment", "Emissions", "Scope 1", "Scope 2", "Water"],
    "P7": ["PRINCIPLE 7", "Policy Advocacy", "Trade Association"],
    "P8": ["PRINCIPLE 8", "Inclusive Growth", "CSR", "Community Development"],
    "P9": ["PRINCIPLE 9", "Customer Value", "Consumer Feedback", "Data Privacy"],
}

@tool
def check_disclosure_completeness() -> dict:
    """Check which of the 9 BRSR principles (P1-P9) have content in the indexed filing.

    Returns:
        A dict with 'principles_present' and 'principles_missing' (P1..P9),
        verified using both metadata tags and hybrid text search.
    """
    rt = _require_runtime()
    
    # 1. Start with metadata tags directly saved in ChromaDB
    tagged_present = set(rt.vector_store.list_principles())
    all_codes = set(cfg.esg.principles.keys())
    
    present = set(tagged_present)
    missing = []

    # 2. For principles not explicitly tagged in metadata, perform hybrid search verification
    for p_code in sorted(all_codes - tagged_present):
        keywords = _PRINCIPLE_SEARCH_TERMS.get(p_code, [f"PRINCIPLE {p_code[1:]}"])
        query = " ".join(keywords)
        
        hits = rt.retriever.retrieve(query=query, top_k=5)
        found = False
        
        for hit in hits:
            hit_text_upper = hit.text.upper()
            p_num = p_code[1:]
            
            # Match "PRINCIPLE 1", "PRINCIPLE - 1", "PRINCIPLE:1", or core keywords
            if f"PRINCIPLE {p_num}" in hit_text_upper or f"PRINCIPLE-{p_num}" in hit_text_upper or f"PRINCIPLE:{p_num}" in hit_text_upper:
                found = True
                break
            
            # Secondary check for key descriptor terms
            matched_terms = sum(1 for kw in keywords[1:] if kw.upper() in hit_text_upper)
            if matched_terms >= 2:
                found = True
                break

        if found:
            present.add(p_code)
        else:
            missing.append(p_code)

    return {
        "principles_present": [f"{p} — {cfg.esg.principles[p]}" for p in sorted(present)],
        "principles_missing": [f"{p} — {cfg.esg.principles[p]}" for p in sorted(missing)],
    }


# ---------------------------------------------------------------------------
# Metric Snapshot + Year-over-Year Comparison
# ---------------------------------------------------------------------------

@tool
def get_current_metric_snapshot() -> dict:
    """Extracts structured ESG metrics, saves full snapshot to disk, and returns a summary."""
    rt = _require_runtime()
    
    # Clear stale findings from previous runs
    rt.history.clear_findings(rt.company, rt.reporting_year)
    
    registry = build_metric_registry(rt.vector_store)
    
    # Save full snapshot to history store
    rt.history.save_filing_snapshot(
        company=rt.company, 
        reporting_year=rt.reporting_year, 
        document_type=rt.document_type,
        filename=rt.filename, 
        metrics=[m.to_dict() for m in registry],
        principles_present=rt.vector_store.list_principles(),
    )
    
    # Return sample payload (first 15 metrics) to keep LLM context within TPM limits
    sample = [m.to_dict() for m in registry[:15]]
    return {
        "total_metrics_extracted": len(registry),
        "snapshot_saved": True,
        "sample_metrics": sample,
        "note": f"Saved all {len(registry)} metrics to snapshot. Returning top 15 in context."
    }


@tool
def compare_year_over_year() -> dict:
    """Compares current filing metrics against prior filing on record.
    Returns the top material changes sorted by magnitude to keep context lightweight."""
    rt = _require_runtime()
    prior = rt.history.get_prior_filing(rt.company, exclude_year=rt.reporting_year)
    if prior is None:
        return {
            "prior_filing": None, 
            "comparisons": [], 
            "not_comparable": [],
            "note": "No prior filing on record for this company — nothing to compare yet."
        }

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
            "metric": name, 
            "unit": cur.unit,
            "prior_year": prior["reporting_year"], 
            "prior_value": prior_val,
            "current_year": rt.reporting_year, 
            "current_value": cur_val,
            "pct_change": pct_change, 
            "abs_pct_change": abs(pct_change) if pct_change is not None else 0,
            "citation": cur.citation_label,
        })

    # Sort by absolute percentage change magnitude (largest swings first)
    comparisons.sort(key=lambda x: x["abs_pct_change"], reverse=True)
    
    # Strip temporary sorting field and limit payload size
    top_swings = []
    for c in comparisons[:15]:
        c_copy = dict(c)
        c_copy.pop("abs_pct_change", None)
        top_swings.append(c_copy)

    return {
        "prior_filing": {"reporting_year": prior["reporting_year"], "filename": prior["filename"]},
        "total_compared_metrics": len(comparisons),
        "top_material_swings": top_swings,
        "not_comparable_count": len(not_comparable),
        "note": f"Compared {len(comparisons)} metrics. Showing top 15 material swings to fit token budget."
    }


# ---------------------------------------------------------------------------
# Grounded Q&A
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
                "already extracted."
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
# Findings, Memo, and Escalate
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
