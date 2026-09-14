"""
indexer.py — End-to-end document indexing pipeline for the ESG platform.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from src.preprocessing.preprocessor import DocumentChunk, load_document, chunk_document
from src.embedding.embedder import Embedder, VectorStore
from src.utils.logger import get_logger

log = get_logger(__name__)


def index_file(
    file_path: Path,
    embedder: Embedder,
    vector_store: VectorStore,
    company: str = "",
    reporting_year: str = "",
    document_type: str = "",
    dual_page_layout: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[DocumentChunk]:
    def _progress(msg: str) -> None:
        log.info(msg)
        if progress_callback:
            progress_callback(msg)

    _progress(f"Loading: {file_path.name}")
    doc = load_document(file_path, dual_page_layout=dual_page_layout)

    _progress(f"Chunking (markdown/table-aware): {file_path.name}")
    chunks = chunk_document(doc)

    if not chunks:
        log.warning("No chunks produced from '%s' — skipping", file_path.name)
        return []

    for c in chunks:
        c.metadata["company"] = company
        c.metadata["reporting_year"] = reporting_year
        c.metadata["document_type"] = document_type

    _progress(f"Embedding {len(chunks)} chunks...")
    t0 = time.perf_counter()
    embeddings = embedder.encode_documents([c.text for c in chunks])
    elapsed = time.perf_counter() - t0
    _progress(f"Embedded {len(chunks)} chunks in {elapsed:.1f}s")

    _progress("Storing in vector database...")
    vector_store.upsert_chunks(chunks, embeddings)
    n_tables = sum(1 for c in chunks if c.content_type == "table")
    _progress(f"Indexed '{file_path.name}': {len(chunks)} chunks ({n_tables} table chunks)")

    return chunks


def index_directory(
    directory: Path,
    embedder: Embedder,
    vector_store: VectorStore,
    company: str = "",
    reporting_year: str = "",
    document_type: str = "",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> int:
    supported_exts = {".pdf", ".txt", ".text", ".json", ".csv", ".xml"}
    files = [f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in supported_exts]
    if not files:
        log.warning("No supported files found in '%s'", directory)
        return 0

    log.info("Indexing %d files from '%s'", len(files), directory)
    total_chunks = 0
    for file_path in files:
        try:
            chunks = index_file(file_path, embedder, vector_store, company, reporting_year,
                                 document_type, progress_callback)
            total_chunks += len(chunks)
        except Exception as exc:
            log.error("Failed to index '%s': %s", file_path.name, exc, exc_info=True)

    log.info("Indexing complete. Total chunks stored: %d", total_chunks)
    return total_chunks
