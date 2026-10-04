from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from config.settings import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RAGASResult:
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
    passed: bool
    failure_reason: str = ""
    statements_checked: int = 0
    statements_supported: int = 0
    generated_questions: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        scores = [self.faithfulness, self.answer_relevancy, self.context_relevancy]
        if any(s <= 0 for s in scores):
            return 0.0
        return round(len(scores) / sum(1.0 / s for s in scores), 3)


# ---------------- LLM CALL ----------------
def _groq_json(prompt: str, api_key: str, system: str = "") -> Optional[dict | list]:
    """Issues LLM call and robustly extracts JSON array or object from response."""
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            max_retries=cfg.llm.max_retries,
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=cfg.llm.judge_model,
            messages=messages,
            temperature=0,
            max_tokens=512,
            extra_body={"reasoning_format": "hidden"},
        )

        text = resp.choices[0].message.content.strip()

        # Extract content between markdown code fences if present
        if "```" in text:
            match_fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
            if match_fence:
                text = match_fence.group(1).strip()

        # Extract the primary JSON array [] or object {} substring
        match_json = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
        if match_json:
            text = match_json.group(1).strip()

        return json.loads(text)

    except Exception as e:
        log.warning("Judge call failed: %s", e)
        return None


# ---------------- FAITHFULNESS ----------------
def _faithfulness(answer: str, context: list[str], api_key: str) -> tuple[float, int, int]:
    if answer.upper().startswith("UNANSWERABLE"):
        return 0.0, 0, 0

    # 1. sentence split
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0, 0, 0

    # 2. build context bag
    context_text = " ".join(context).lower()
    context_tokens = set(context_text.split())

    def is_supported(sentence: str) -> bool:
        tokens = set(sentence.lower().split())
        overlap = len(tokens & context_tokens)
        return overlap >= 3   # Grounding threshold

    supported = sum(is_supported(s) for s in sentences)
    score = supported / len(sentences)

    return round(score, 3), len(sentences), supported


# ---------------- ANSWER RELEVANCY ----------------
def _answer_relevancy(question: str, answer: str, embedder, api_key: str) -> tuple[float, list[str]]:
    if answer.upper().startswith("UNANSWERABLE"):
        return 0.0, []

    system_prompt = "You are an evaluation assistant. Output ONLY a valid JSON array of 3 strings."
    user_prompt = f"""Generate 3 distinct questions that this answer directly answers.

Answer:
{answer}

Respond ONLY with a JSON list of 3 strings, e.g.: ["Question 1?", "Question 2?", "Question 3?"]"""

    raw = _groq_json(user_prompt, api_key, system=system_prompt)

    if not isinstance(raw, list) or not raw:
        return 0.5, []

    gen_qs = [str(q) for q in raw[:3]]

    try:
        from src.utils.cosine import cosine_similarity

        q_vec = embedder.encode_query(question)
        gen_vecs = embedder.encode_documents(gen_qs)

        sims = [cosine_similarity(q_vec, v) for v in gen_vecs]
        return round(sum(sims) / len(sims), 3), gen_qs

    except Exception as exc:
        log.warning("Cosine similarity calculation failed in RAGAS: %s", exc)
        return 0.5, gen_qs


# ---------------- CONTEXT RELEVANCY ----------------
def _context_relevancy(scores: list[float]) -> float:
    if not scores:
        return 0.0

    # Sigmoid normalization for cross-encoder rerank scores if needed
    if max(scores) <= 1.0:
        vals = scores
    else:
        vals = [1 / (1 + math.exp(-s)) for s in scores]

    return round(sum(vals) / len(vals), 3)


# ---------------- MAIN ----------------
class RAGASGuard:
    def __init__(self, embedder):
        self._embedder = embedder

    def evaluate(self, question: str, answer: str, chunks: list, api_key: str) -> RAGASResult:
        ctx = [c.text for c in chunks]
        rerank = [getattr(c, "rerank_score", 0.0) for c in chunks]

        ctx_rel = _context_relevancy(rerank)
        faith, n, sup = _faithfulness(answer, ctx, api_key)
        ans_rel, gen_qs = _answer_relevancy(question, answer, self._embedder, api_key)

        failures = []

        if faith < cfg.ragas.min_faithfulness:
            failures.append("faithfulness low")

        if ans_rel < cfg.ragas.min_answer_relevancy:
            failures.append("answer relevancy low")

        if ctx_rel < cfg.ragas.min_context_relevancy:
            failures.append("context relevancy low")

        return RAGASResult(
            faithfulness=faith,
            answer_relevancy=ans_rel,
            context_relevancy=ctx_rel,
            passed=len(failures) == 0,
            failure_reason="; ".join(failures),
            statements_checked=n,
            statements_supported=sup,
            generated_questions=gen_qs,
        )