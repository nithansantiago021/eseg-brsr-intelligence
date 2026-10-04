from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from config.settings import cfg
from src.retrieval.retriever import Retriever, RetrievedChunk
from src.metrics.schema import ESGMetric
from src.visualization.chart_generator import wants_chart, find_relevant_metrics, build_chart
from src.utils.logger import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = """
You are a precise ESG / BRSR Report Analyst.

RULES:
1. Use ONLY the provided sources.
2. Every factual statement must include standard citations in the format [SOURCE N] (e.g., [SOURCE 1]).
3. NEVER use unicode or full-width brackets like 【SOURCE 1】 or (SOURCE 1) — use ASCII square brackets [SOURCE N] strictly.
4. Do NOT use outside knowledge or invent ESG figures.
5. Preserve exact reported units and reporting periods — do not convert or round silently.
6. If a value is not directly stated but calculated from source data, say so explicitly
   and show which sources it was derived from.
7. If insufficient context exists, respond exactly:
   UNANSWERABLE: insufficient information in sources
"""

CONTEXT_TEMPLATE = "[SOURCE {n}] (Company: {company} | FY: {year} | Page: {page} | Principle: {principle}) {text}"

USER_PROMPT_TEMPLATE = """
SOURCES:
{context}

QUESTION:
{question}

Answer clearly with citations [SOURCE N]. If the question asks to compare years,
companies, or principles, state each figure with its own citation.
"""


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list
    cited_source_indices: list
    is_answerable: bool
    faithfulness_score: float
    answer_relevancy_score: float
    context_relevancy_score: float
    ragas_overall: float
    ragas_passed: bool
    latency_ms: float
    model_used: str
    gate_failed: str = ""
    chart_metrics: list = field(default_factory=list)
    chart_refusal: str = ""


class RAGPipeline:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self._ragas_guard = None
        self._metric_registry: list[ESGMetric] = []

    def set_ragas_guard(self, guard):
        self._ragas_guard = guard

    def set_metric_registry(self, registry: list[ESGMetric]) -> None:
        """Called after (re-)indexing so chart generation has current data."""
        self._metric_registry = registry

    def _call_llm(self, messages, api_key):
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=cfg.llm.base_url,
            timeout=cfg.llm.timeout,
            max_retries=cfg.llm.max_retries,
        )
        resp = client.chat.completions.create(
            model=cfg.llm.model_name,
            messages=messages,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            extra_body={"reasoning_format": "hidden"},
        )
        return resp.choices[0].message.content.strip()

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            CONTEXT_TEMPLATE.format(
                n=i + 1, company=c.company or "N/A", year=c.reporting_year or "N/A",
                page=c.page or "N/A", principle=c.principle or "N/A", text=c.text.strip(),
            )
            for i, c in enumerate(chunks)
        )

    def _extract_citations(self, text: str) -> list[int]:
        """Extracts numerical source indices from both standard [SOURCE N] and unicode 【SOURCE N】 citations."""
        return sorted(set(int(x) for x in re.findall(r"[\[【]SOURCE\s*(\d+)[\]】]", text, flags=re.IGNORECASE)))

    def run(
        self, question: str, api_key: str,
        filename_filter: Optional[str] = None, company: Optional[str] = None,
        reporting_year: Optional[str] = None, principle: Optional[str] = None,
    ) -> RAGResponse:
        t0 = time.perf_counter()

        if not api_key:
            return self._fail(question, "Missing Groq API key (set GROQ_API_KEY)", "gate_0", t0)

        chunks = self._retriever.retrieve(
            question, filename_filter=filename_filter, company=company,
            reporting_year=reporting_year, principle=principle,
        )
        if not chunks:
            return self._fail(question, "No relevant chunks found", "gate_1", t0)

        context = self._build_context(chunks)
        prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)
        raw_answer = self._call_llm(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], api_key,
        )

        # Normalize any unicode brackets in the raw answer to standard ASCII brackets
        answer = re.sub(r"【\s*SOURCE\s*(\d+)\s*】", r"[SOURCE \1]", raw_answer, flags=re.IGNORECASE)
        citations = self._extract_citations(answer)

        ragas = self._ragas_guard.evaluate(question, answer, chunks, api_key) if self._ragas_guard else None
        is_unanswerable = answer.upper().startswith("UNANSWERABLE")
        if is_unanswerable:
            return self._fail(question, answer, "gate_llm_unanswerable", t0, chunks)

        if len(citations) == 0:
            log.warning("No citations found in generated answer text")

        chart_metrics, chart_refusal = [], ""
        if wants_chart(question) and self._metric_registry:
            matched = find_relevant_metrics(question, self._metric_registry, company, reporting_year)
            fig, cites, refusal = build_chart(matched)
            if fig is not None:
                chart_metrics = matched
            chart_refusal = refusal

        latency = (time.perf_counter() - t0) * 1000
        return RAGResponse(
            question=question, answer=answer, sources=chunks,
            cited_source_indices=citations, is_answerable=True,
            faithfulness_score=ragas.faithfulness if ragas else 0.0,
            answer_relevancy_score=ragas.answer_relevancy if ragas else 0.0,
            context_relevancy_score=ragas.context_relevancy if ragas else 0.0,
            ragas_overall=ragas.overall_score if ragas else 0.0,
            ragas_passed=ragas.passed if ragas else True,
            latency_ms=latency, model_used=cfg.llm.model_name,
            chart_metrics=chart_metrics, chart_refusal=chart_refusal,
        )

    def _fail(self, question, reason, gate, t0, chunks=None):
        return RAGResponse(
            question=question, answer=f"UNANSWERABLE: {reason}", sources=chunks or [],
            cited_source_indices=[], is_answerable=False,
            faithfulness_score=0.0, answer_relevancy_score=0.0, context_relevancy_score=0.0,
            ragas_overall=0.0, ragas_passed=False,
            latency_ms=(time.perf_counter() - t0) * 1000, model_used="none", gate_failed=gate,
        )
