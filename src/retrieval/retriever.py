"""
retriever.py — Two-stage retrieval (bi-encoder ANN + cross-encoder re-rank)
for the ESG platform.
"""

from __future__ import annotations

import difflib
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config.settings import cfg
from src.embedding.embedder import Embedder, VectorStore
from src.utils.logger import get_logger

log = get_logger(__name__)

_METRIC_QUERY_RE = re.compile(
    r"\b(scope\s*[123]|emission|water|waste|energy|renewable|training|attrition|"
    r"csr|spend|intensity|percentage|percent|how much|how many|figure|kpi|target)\b",
    re.IGNORECASE,
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    filename: str
    section: str
    content_type: str
    principle: str
    page: str
    table_title: str
    company: str
    reporting_year: str
    document_type: str
    similarity: float
    rerank_score: float
    doc_id: str


class Retriever:
    """Two-stage semantic retriever with cross-encoder re-ranking and ESG metadata filtering."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = None

    def set_reranker(self, reranker) -> None:
        self._reranker = reranker

    def _load_reranker(self) -> None:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            log.info("Loading reranker: %s", cfg.retrieval.reranker_model)
            self._reranker = CrossEncoder(cfg.retrieval.reranker_model)

    def _deduplicate(self, chunks: list[RetrievedChunk], threshold: float = 0.85) -> list[RetrievedChunk]:
        unique = []
        for c in chunks:
            if any(difflib.SequenceMatcher(None, c.text, u.text, autojunk=False).ratio() >= threshold for u in unique):
                continue
            unique.append(c)
        return unique

    def _build_where(self, filename_filter, company, reporting_year, principle, section, document_type) -> Optional[dict]:
        clauses = []
        if filename_filter:
            clauses.append({"filename": filename_filter})
        if company:
            clauses.append({"company": company})
        if reporting_year:
            clauses.append({"reporting_year": reporting_year})
        if principle:
            clauses.append({"principle": principle})
        if document_type:
            clauses.append({"document_type": document_type})
        if section:
            clauses.append({"section": {"$eq": section}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filename_filter: Optional[str] = None,
        company: Optional[str] = None,
        reporting_year: Optional[str] = None,
        principle: Optional[str] = None,
        section: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> list[RetrievedChunk]:

        if self._vector_store.count == 0:
            log.warning("Empty vector store")
            return []

        query_vec = self._embedder.encode_query(query)
        where = self._build_where(filename_filter, company, reporting_year, principle, section, document_type)
        k = top_k or cfg.retrieval.top_k

        raw_hits = self._vector_store.similarity_search(query_embedding=query_vec, top_k=k, where=where)
        if not raw_hits:
            return []

        candidates = [
            RetrievedChunk(
                chunk_id=h["id"], text=h["text"],
                filename=h["metadata"].get("filename", "unknown"),
                section=h["metadata"].get("section", ""),
                content_type=h["metadata"].get("content_type", "text"),
                principle=h["metadata"].get("principle", ""),
                page=h["metadata"].get("page", ""),
                table_title=h["metadata"].get("table_title", ""),
                company=h["metadata"].get("company", ""),
                reporting_year=h["metadata"].get("reporting_year", ""),
                document_type=h["metadata"].get("document_type", ""),
                similarity=h["similarity"], rerank_score=0.0,
                doc_id=h["metadata"].get("doc_id", ""),
            )
            for h in raw_hits if h["similarity"] >= cfg.retrieval.similarity_threshold
        ]
        if not candidates:
            return []

        is_metric_query = bool(_METRIC_QUERY_RE.search(query))

        if cfg.retrieval.use_reranker and len(candidates) > 1:
            self._load_reranker()
            pairs = [(query, c.text) for c in candidates]
            scores = np.array(self._reranker.predict(pairs), dtype=float)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
            for c, s in zip(candidates, scores):
                boost = cfg.retrieval.table_query_boost if (is_metric_query and c.content_type == "table") else 0.0
                c.rerank_score = float(s) + boost
        else:
            for c in candidates:
                boost = cfg.retrieval.table_query_boost if (is_metric_query and c.content_type == "table") else 0.0
                c.rerank_score = c.similarity + boost

        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        candidates = candidates[: cfg.retrieval.rerank_top_n]
        candidates = self._deduplicate(candidates)

        log.info("Retrieved %d chunks | top score=%.3f | query='%s'",
                  len(candidates), candidates[0].rerank_score if candidates else 0.0, query[:50])
        return candidates
