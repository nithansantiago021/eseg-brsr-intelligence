"""
summarizer.py — Structured ESG/BRSR document summary generation.
"""

from __future__ import annotations

from config.settings import cfg
from src.retrieval.retriever import Retriever

SUMMARY_SECTIONS = [
    ("Executive Overview", "company overview, reporting scope, key highlights"),
    ("Environmental Performance", "emissions, energy, water, waste, environmental initiatives"),
    ("Social Performance", "employee well-being, diversity, training, community, human rights"),
    ("Governance", "board composition, ethics, transparency, policy advocacy"),
    ("Key ESG Metrics", "quantitative metrics and KPIs disclosed"),
    ("Targets and Commitments", "stated ESG targets, goals, commitments and timelines"),
    ("Material Issues and Risks", "material issues, risks, and notable disclosures"),
]

SUMMARY_SYSTEM_PROMPT = """
You are an ESG/BRSR Report Analyst producing a structured summary.

RULES:
1. Use ONLY the provided sources.
2. Every factual statement must include [SOURCE N].
3. Do NOT use outside knowledge or invent figures.
4. If a section has no supporting sources, write "Not disclosed in the provided document."
5. Clearly separate reported values from any inference; do not infer values.
"""

SUMMARY_USER_TEMPLATE = """
Produce a structured ESG summary for "{doc_label}" using these headings, in order:
{headings}

SOURCES:
{context}

For each heading, write 2-5 sentences grounded strictly in the sources, with [SOURCE N] citations.
"""


def _call_llm(messages, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=cfg.llm.base_url, timeout=cfg.llm.timeout,
                     max_retries=cfg.llm.max_retries)
    resp = client.chat.completions.create(
        model=cfg.llm.model_name, messages=messages,
        temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens,
        extra_body={"reasoning_format": "hidden"},
    )
    return resp.choices[0].message.content.strip()


def generate_summary(retriever: Retriever, api_key: str, doc_label: str,
                      filename_filter: str = None, company: str = None,
                      reporting_year: str = None) -> tuple[str, list]:
    """Returns (summary_markdown, sources_used)."""
    if not api_key:
        return "UNANSWERABLE: Missing Groq API key (set GROQ_API_KEY).", []

    all_chunks = []
    for _, query_hint in SUMMARY_SECTIONS:
        hits = retriever.retrieve(
            query=query_hint, top_k=6, filename_filter=filename_filter,
            company=company, reporting_year=reporting_year,
        )
        all_chunks.extend(hits)

    # de-dup by chunk_id, cap total context size
    seen, unique_chunks = set(), []
    for c in all_chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        unique_chunks.append(c)
    unique_chunks = unique_chunks[:24]

    if not unique_chunks:
        return "UNANSWERABLE: No relevant content found for this document.", []

    context = "\n\n".join(
        f"[SOURCE {i+1}] (p.{c.page}{' · ' + c.principle if c.principle else ''}) {c.text.strip()}"
        for i, c in enumerate(unique_chunks)
    )
    headings = "\n".join(f"- {h}" for h, _ in SUMMARY_SECTIONS)

    prompt = SUMMARY_USER_TEMPLATE.format(doc_label=doc_label, headings=headings, context=context)
    summary = _call_llm(
        [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        api_key,
    )
    return summary, unique_chunks
