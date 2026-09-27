"""
compliance_sentinel.py — The ESG/BRSR Compliance Sentinel agent.

Usage
-----
CLI (one-off review):
    python -m agent.compliance_sentinel \\
        --file /path/to/report.pdf --company "Tata Steel" \\
        --year FY2024-25 --doctype "BRSR Report"

Programmatic (e.g. from app.py's Dashboard "Run Compliance Sentinel" button):
    from agent.compliance_sentinel import run_compliance_check
    result = run_compliance_check(file_path, company, reporting_year, document_type)
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*generator didn't stop after athrow.*")
warnings.filterwarnings("ignore", message=".*async_generator.*")

import argparse
import sys
from pathlib import Path

from strands import Agent

from config.settings import GROQ_API_KEY
from agent.model_provider import build_model
from agent.history_store import FilingHistoryStore
from agent.tools import (
    AgentRuntime, configure_runtime,
    index_filing, check_disclosure_completeness, get_current_metric_snapshot,
    compare_year_over_year, rag_ask, flag_finding, save_review_memo, notify_human,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = """
You are the ESG/BRSR Compliance Sentinel — an autonomous agent that reviews
a newly submitted ESG/BRSR/sustainability filing the way a careful junior
compliance analyst would, so a human reviewer only needs to look at filings
that actually need their judgment.

For every filing you are given, follow this process:

1. Call index_filing to load and structure the document.
2. Call check_disclosure_completeness to see which BRSR principles have no
   detected content — a missing/thin disclosure is worth flagging.
3. Call get_current_metric_snapshot to extract every structured metric from
   the filing's tables (this also saves the snapshot for future comparisons).
4. Call compare_year_over_year. If this is the company's first reviewed
   filing, there will be nothing to compare — that's fine, note it and move on.
5. Look at the comparisons from compare_year_over_year and rank them by the
   size of the change. Starting with the largest swings, use rag_ask to
   verify the exact figures and wording before concluding anything — but
   rag_ask is budget-limited per run (each call costs two LLM requests), so
   prioritize the handful of most material changes rather than trying to
   verify every metric that crossed 20%. Once rag_ask tells you the budget
   is exhausted, stop calling it and finish the memo using the metric values
   you already extracted, noting which items couldn't be individually
   verified. Never state a number or a conclusion you haven't either gotten
   directly from a tool result or verified via rag_ask — you have no other
   source of truth about this filing's content.
6. Call flag_finding for every genuine issue: missing disclosures, large
   swings, apparent target misses, or a rag_ask answer with low faithfulness
   that you can't fully trust. Use "info" for routine notes, "review_suggested"
   for things worth a human glance, "action_required" for material issues.
   Do not flag ordinary, expected, small variations — a clean filing with no
   flags is a normal and good outcome, not a failure to find something.
7. Write a structured markdown memo (Executive Summary, Year-over-Year
   Comparison, Findings, Missing Disclosures, Recommendation) with citations
   for every figure, and save it with save_review_memo. Set status to
   "auto_closed" if there are no review_suggested/action_required findings,
   otherwise "pending_human_review".
8. Only call notify_human if the memo status is "pending_human_review" — and
   keep the message to two sentences on what needs attention and why. Never
   call notify_human for a clean, auto_closed filing; silently filing the
   memo is the correct outcome for most filings, and interrupting a human
   for nothing wastes the whole point of automating this review.

Be precise and honest about uncertainty. If you cannot verify something,
say so in the memo rather than stating it with unearned confidence.
"""


def build_agent() -> Agent:
    return Agent(
        model=build_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=[
            index_filing, check_disclosure_completeness, get_current_metric_snapshot,
            compare_year_over_year, rag_ask, flag_finding, save_review_memo, notify_human,
        ],
    )


def run_compliance_check(
    file_path: str, company: str, reporting_year: str,
    document_type: str = "BRSR Report", dual_page_layout: bool = False,
    api_key: str = "",
) -> dict:
    from src.embedding.embedder import Embedder, VectorStore
    from src.retrieval.retriever import Retriever
    from src.evaluation.ragas_guard import RAGASGuard
    from src.rag.rag_pipeline import RAGPipeline

    key = api_key or GROQ_API_KEY
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(embedder, vector_store)
    pipeline = RAGPipeline(retriever)
    pipeline.set_ragas_guard(RAGASGuard(embedder))
    history = FilingHistoryStore()

    runtime = AgentRuntime(
        embedder=embedder, vector_store=vector_store, retriever=retriever, pipeline=pipeline,
        history=history, api_key=key, company=company, reporting_year=reporting_year,
        document_type=document_type, filename=Path(file_path).name,
    )
    configure_runtime(runtime)

    agent = build_agent()
    task = (
        f"Review the filing at {file_path!r} for {company}, reporting year {reporting_year}, "
        f"document type {document_type!r}"
        + (", noting that each PDF page holds two merged logical pages side-by-side (pass "
           "dual_page_layout=True to index_filing)" if dual_page_layout else "")
        + ". Follow your review process fully."
    )
    result = agent(task)

    memo = history.get_memo(company, reporting_year)
    findings = history.get_findings(company, reporting_year)
    return {
        "agent_response": str(result),
        "memo": memo,
        "findings": findings,
        "company": company,
        "reporting_year": reporting_year,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the ESG/BRSR Compliance Sentinel on a filing.")
    parser.add_argument("--file", required=True, help="Path to the report file")
    parser.add_argument("--company", required=True)
    parser.add_argument("--year", required=True, dest="reporting_year", help="Reporting year, e.g. FY2024-25")
    parser.add_argument("--doctype", default="BRSR Report", dest="document_type")
    parser.add_argument("--dual-page-layout", action="store_true",
                         help="Set if each PDF page merges two logical pages side-by-side")
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    result = run_compliance_check(
        args.file, args.company, args.reporting_year, args.document_type, args.dual_page_layout,
    )

    print("\n" + "=" * 70)
    print(f"Compliance Sentinel review — {result['company']} · {result['reporting_year']}")
    print("=" * 70)
    print(result["agent_response"])
    if result["memo"]:
        print(f"\nMemo saved to: {result['memo']['path']} (status: {result['memo']['status']})")
    if result["findings"]:
        print(f"\n{len(result['findings'])} finding(s) recorded:")
        for f in result["findings"]:
            print(f"  [{f['severity']}] {f['topic']}: {f['description']}")


if __name__ == "__main__":
    _cli()
