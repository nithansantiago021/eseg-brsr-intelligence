"""
retriever.py — Hybrid Retrieval (BM25 Sparse + Bi-Encoder Dense Vector)
with Reciprocal Rank Fusion (RRF) and Cross-Encoder Re-ranking.
"""

from __future__ import annotations

import difflib
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any

from rank_bm25 import BM25Okapi

from config.settings import cfg
from src.embedding.embedder import Embedder, VectorStore
from src.utils.logger import get_logger

log = get_logger(__name__)

_METRIC_QUERY_RE = re.compile(
    r"\b(scope\s*[123]|emission|water|waste|energy|renewable|training|attrition|"
    r"csr|spend|intensity|percentage|percent|how much|how many|figure|kpi|target|"
    r"not applicable|na|nil|exempt|exempted|not reported)\b",
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
    """Two-stage Hybrid Retriever combining BM25, ChromaDB Vector Search, and Cross-Encoder Reranking."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker: Optional[Any] = None
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_chunk_map: list[dict] = []

    def set_reranker(self, reranker: Any) -> None:
        self._reranker = reranker

    def _get_or_load_reranker(self) -> Any:
        """Type-safe lazy-loader for CrossEncoder."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            log.info("Loading reranker: %s", cfg.retrieval.reranker_model)
            self._reranker = CrossEncoder(cfg.retrieval.reranker_model)
        return self._reranker

    def build_bm25_index(self, all_chunks: list[dict]) -> None:
        """Updates internal BM25 index with newly indexed document chunks."""
        if not all_chunks:
            return
        self._bm25_chunk_map.extend(all_chunks)
        corpus = [c["text"].lower().split() for c in self._bm25_chunk_map]
        self._bm25_index = BM25Okapi(corpus)
        log.info("BM25 index updated with %d total chunks", len(self._bm25_chunk_map))

    def _search_bm25(self, query: str, top_k: int) -> list[dict]:
        if self._bm25_index is None or not self._bm25_chunk_map:
            return []

        tokens = query.lower().split()
        scores = self._bm25_index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                chunk = self._bm25_chunk_map[idx]
                hits.append({
                    "id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "similarity": float(scores[idx]),
                })
        return hits

    def _reciprocal_rank_fusion(self, vector_hits: list[dict], bm25_hits: list[dict], rrf_k: int = 60) -> list[dict]:
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            h_id = hit["id"]
            scores[h_id] = scores.get(h_id, 0.0) + (1.0 / (rrf_k + rank))
            chunks_by_id[h_id] = hit

        for rank, hit in enumerate(bm25_hits, start=1):
            h_id = hit["id"]
            scores[h_id] = scores.get(h_id, 0.0) + (1.0 / (rrf_k + rank))
            if h_id not in chunks_by_id:
                chunks_by_id[h_id] = hit

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused = []
        for h_id in sorted_ids:
            item = chunks_by_id[h_id]
            item["rrf_score"] = scores[h_id]
            fused.append(item)
        return fused

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

        k = top_k or cfg.retrieval.top_k
        where = self._build_where(filename_filter, company, reporting_year, principle, section, document_type)

        query_vec = self._embedder.encode_query(query)
        vector_hits = self._vector_store.similarity_search(query_embedding=query_vec, top_k=k * 2, where=where)
        bm25_hits = self._search_bm25(query, top_k=k * 2)

        fused_hits = self._reciprocal_rank_fusion(vector_hits, bm25_hits)
        if not fused_hits:
            return []

        candidates = [
            RetrievedChunk(
                chunk_id=h["id"],
                text=h["text"],
                filename=h["metadata"].get("filename", "unknown"),
                section=h["metadata"].get("section", ""),
                content_type=h["metadata"].get("content_type", "text"),
                principle=h["metadata"].get("principle", ""),
                page=h["metadata"].get("page", ""),
                table_title=h["metadata"].get("table_title", ""),
                company=h["metadata"].get("company", ""),
                reporting_year=h["metadata"].get("reporting_year", ""),
                document_type=h["metadata"].get("document_type", ""),
                similarity=float(h.get("similarity", 0.0)),
                rerank_score=float(h.get("rrf_score", 0.0)),
                doc_id=h["metadata"].get("doc_id", ""),
            )
            for h in fused_hits
        ]

        is_metric_query = bool(_METRIC_QUERY_RE.search(query))

        if cfg.retrieval.use_reranker and len(candidates) > 1:
            reranker = self._get_or_load_reranker()
            pairs = [(query, c.text) for c in candidates]
            raw_scores = reranker.predict(pairs)
            scores = np.array(raw_scores, dtype=float)

            if scores.max() != scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

            for c, s in zip(candidates, scores):
                boost = cfg.retrieval.table_query_boost if (is_metric_query and c.content_type == "table") else 0.0
                c.rerank_score = float(s) + boost
        else:
            for c in candidates:
                boost = cfg.retrieval.table_query_boost if (is_metric_query and c.content_type == "table") else 0.0
                c.rerank_score += boost

        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        candidates = candidates[: getattr(cfg.retrieval, "rerank_top_n", 8)]
        candidates = self._deduplicate(candidates)

        log.info(
            "Retrieved %d hybrid chunks | top score=%.3f | query='%s'",
            len(candidates),
            candidates[0].rerank_score if candidates else 0.0,
            query[:50],
        )
        return candidates
