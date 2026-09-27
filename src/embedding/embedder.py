"""
embedder.py — Embedding + ephemeral vector store for the ESG platform.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import cfg
from src.preprocessing.preprocessor import DocumentChunk
from src.utils.logger import get_logger

log = get_logger(__name__)


class Embedder:
    def __init__(self) -> None:
        log.info("Loading embedding model: %s", cfg.embedding.model_name)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(cfg.embedding.model_name, device=cfg.embedding.device)
        
        # Explicitly set max_seq_length to prevent sequence length warnings for large chunks/tables
        self._model.max_seq_length = 512
        
        log.info("Model loaded in %.2fs", time.perf_counter() - t0)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=cfg.embedding.batch_size,
            normalize_embeddings=cfg.embedding.normalize,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def encode_query(self, query: str) -> list[float]:
        vec = self._model.encode([query], normalize_embeddings=cfg.embedding.normalize)
        return vec[0].tolist()


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection_name = f"{cfg.vectorstore.collection_name}_{uuid.uuid4().hex[:12]}"
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": cfg.vectorstore.distance_metric},
        )
        log.info("Ephemeral VectorStore created (isolated collection '%s', in-memory only)",
                  self._collection_name)

    @property
    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]], batch_size: int = 64) -> None:
        assert len(chunks) == len(embeddings)
        for i in range(0, len(chunks), batch_size):
            bc = chunks[i:i + batch_size]
            be = embeddings[i:i + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in bc],
                embeddings=be,
                documents=[c.text for c in bc],
                metadatas=[c.to_chroma_metadata() for c in bc],
            )
        log.info("Upserted %d chunks (total: %d)", len(chunks), self.count)

    def similarity_search(self, query_embedding: list[float], top_k: Optional[int] = None,
                           where: Optional[dict] = None) -> list[dict]:
        k = top_k or cfg.retrieval.top_k
        if self._collection.count() == 0:
            return []
        kw: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kw["where"] = where
        results = self._collection.query(**kw)
        hits = []
        for doc, meta, dist, cid in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0], results["ids"][0],
        ):
            similarity = 1.0 - dist
            if similarity >= cfg.retrieval.similarity_threshold:
                hits.append({"id": cid, "text": doc, "metadata": meta, "similarity": round(similarity, 4)})
        return hits

    def get_all_metadata(self) -> list[dict]:
        if self.count == 0:
            return []
        return self._collection.get(include=["metadatas"])["metadatas"]

    def get_table_chunks(self, where: Optional[dict] = None) -> list[dict]:
        """All table-content chunks (with text) — used by the metric extractor."""
        if self.count == 0:
            return []
        kw = {"include": ["documents", "metadatas"]}
        base_where = {"content_type": "table"}
        kw["where"] = {"$and": [base_where, where]} if where else base_where
        res = self._collection.get(**kw)
        return [{"id": cid, "text": d, "metadata": m}
                for cid, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def list_documents(self) -> list[dict]:
        if self.count == 0:
            return []
        seen, docs = set(), []
        for m in self.get_all_metadata():
            if m["doc_id"] not in seen:
                seen.add(m["doc_id"])
                docs.append({
                    "doc_id": m["doc_id"], "filename": m["filename"],
                    "company": m.get("company", ""), "reporting_year": m.get("reporting_year", ""),
                    "document_type": m.get("document_type", ""),
                })
        return docs

    def list_companies(self) -> list[str]:
        return sorted({m.get("company", "") for m in self.get_all_metadata() if m.get("company")})

    def list_years(self) -> list[str]:
        return sorted({m.get("reporting_year", "") for m in self.get_all_metadata() if m.get("reporting_year")})

    def list_principles(self) -> list[str]:
        return sorted({m.get("principle", "") for m in self.get_all_metadata() if m.get("principle")})

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})
        log.info("Deleted chunks for doc_id='%s'", doc_id)

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": cfg.vectorstore.distance_metric},
        )
        log.info("VectorStore reset: all chunks deleted")