"""
preprocessor.py — Layout-aware document extraction + table-preserving chunking
for the ESG / Multi-Framework Report Intelligence Platform using Docling.
"""

from __future__ import annotations

import gc
import json
import csv
import sys
import re
import hashlib
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass, field

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from config.settings import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Principle Detection Regex
# ---------------------------------------------------------------------------

_PRINCIPLE_RE = re.compile(r"\bPRINCIPLE\s*[-:]?\s*([1-9])\b", re.IGNORECASE)

def _detect_principle_tag(text: str) -> str:
    """Detects BRSR Principle numbers (P1-P9) from text or section breadcrumbs."""
    match = _PRINCIPLE_RE.search(text)
    if match:
        return f"P{match.group(1)}"
    return ""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    section: str = ""
    content_type: str = "text"     # "text" | "table"
    principle: str = ""
    page: str = ""                 # e.g., "12" or "12-13"
    table_title: str = ""
    metadata: dict = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict:
        return {
            "doc_id":       self.doc_id,
            "filename":     self.filename,
            "chunk_index":  self.chunk_index,
            "char_start":   self.char_start,
            "char_end":     self.char_end,
            "section":      self.section,
            "content_type": self.content_type,
            "principle":    self.principle,
            "page":         self.page,
            "table_title":  self.table_title,
            **{k: str(v) for k, v in self.metadata.items()},
        }


@dataclass
class RawDocument:
    doc_id: str
    filename: str
    file_type: str
    raw_text: str = ""
    docling_doc: Optional[DoclingDocument] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Singleton Docling Converter Setup
# ---------------------------------------------------------------------------

def _get_docling_converter() -> DocumentConverter:
    """Configures Docling for fast digital parsing without forcing CPU OCR."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False         # Fast digital text extraction
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


_CONVERTER: Optional[DocumentConverter] = None

def get_converter() -> DocumentConverter:
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = _get_docling_converter()
    return _CONVERTER


# ---------------------------------------------------------------------------
# PDF Processing (Docling Layout Engine)
# ---------------------------------------------------------------------------

def _process_pdf_docling(path: Path, doc_id: str) -> tuple[DoclingDocument, str]:
    log.info("Parsing PDF with Docling layout engine: %s", path.name)
    converter = get_converter()
    result = converter.convert(str(path))
    
    docling_doc: DoclingDocument = result.document
    full_markdown = docling_doc.export_to_markdown()
    
    return docling_doc, full_markdown


def chunk_pdf_docling(doc: RawDocument) -> list[DocumentChunk]:
    if doc.docling_doc is None:
        log.warning("No Docling document found for %s; skipping markdown chunking.", doc.filename)
        return []

    chunker = HybridChunker(
        merge_peers=True,
        repeat_table_header=True
    )
    
    raw_chunks = list(chunker.chunk(doc.docling_doc))
    chunks: list[DocumentChunk] = []
    offset = 0

    for idx, d_chunk in enumerate(raw_chunks):
        text = chunker.serialize(d_chunk).strip()
        if not text:
            continue

        # 1. Native Docling table detection from chunk metadata
        is_table = False
        meta = getattr(d_chunk, "meta", None)
        if meta and hasattr(meta, "doc_items") and meta.doc_items:
            for item in meta.doc_items:
                label = getattr(item, "label", "").lower()
                class_name = type(item).__name__.lower()
                if label == "table" or "table" in class_name:
                    is_table = True
                    break

        # 2. Fallback text pattern check if metadata tag wasn't captured
        if not is_table:
            is_table = "|---" in text or "| ---" in text or text.count("|") >= 4 or text.startswith("|")

        # Extract headings and section breadcrumbs safely
        headings = []
        if meta and hasattr(meta, "headings") and meta.headings:
            headings = [str(h) for h in meta.headings]
        
        section_path = " > ".join(headings) if headings else "General"

        # Detect Principle (P1..P9) from text or section breadcrumbs
        detected_p = _detect_principle_tag(section_path) or _detect_principle_tag(text)

        # Detect page numbers safely from doc_items metadata
        page_numbers = set()
        if meta and hasattr(meta, "doc_items") and meta.doc_items:
            for item in meta.doc_items:
                prov_list = getattr(item, "prov", []) or []
                for p in prov_list:
                    page_no = getattr(p, "page_no", None)
                    if page_no is not None:
                        page_numbers.add(page_no)
        
        page_str = ""
        if page_numbers:
            min_p, max_p = min(page_numbers), max(page_numbers)
            page_str = f"{min_p}" if min_p == max_p else f"{min_p}-{max_p}"

        enriched_text = f"[Section: {section_path}]\n{text}" if not text.startswith("[Section:") else text
        chunk_id = hashlib.sha256(f"{doc.doc_id}:{idx}".encode()).hexdigest()[:16]
        
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            filename=doc.filename,
            chunk_index=idx,
            text=enriched_text,
            char_start=offset,
            char_end=offset + len(enriched_text),
            section=section_path,
            content_type="table" if is_table else "text",
            principle=detected_p,
            page=page_str,
            metadata={"doc_file_type": doc.file_type}
        ))
        
        offset += len(enriched_text) + 1

    gc.collect()
    log.info("Document '%s' → %d Docling layout-aware chunks", doc.filename, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Fallback / Non-PDF Document Processing
# ---------------------------------------------------------------------------

def _extract_text_from_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("UTF-8 decode failed for '%s'; falling back to latin-1", path.name)
        return path.read_text(encoding="latin-1")


def _extract_text_from_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return "\n\n---\n\n".join(item.get("text", "") for item in data)
    if isinstance(data, dict):
        return data.get("text", str(data))
    return str(data)


def _extract_text_from_xml(path: Path) -> str:
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError as e:
        log.warning("XML parse error for '%s': %s — falling back to raw text read", path.name, e)
        return path.read_text(encoding="utf-8", errors="replace")

    lines = []
    for elem in root.iter():
        text = (elem.text or "").strip()
        if text:
            tag = elem.tag.split("}")[-1]
            attrs = " ".join(f'{k}="{v}"' for k, v in elem.attrib.items())
            lines.append(f"{tag}{(' ' + attrs) if attrs else ''}: {text}")
    return "\n".join(lines)


def _extract_text_from_csv(path: Path) -> str:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
    rows = []
    with open(path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_lines = [f"{col}: {val}" for col, val in row.items() if val]
            rows.append("\n".join(text_lines))
    return "\n\n---\n\n".join(rows)


def chunk_legacy_formats(doc: RawDocument) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    raw_blocks = doc.raw_text.split("\n\n---\n\n")
    offset = 0

    for i, block in enumerate(raw_blocks):
        text = block.strip()
        if not text:
            continue
        chunk_id = hashlib.sha256(f"{doc.doc_id}:{i}".encode()).hexdigest()[:16]
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            filename=doc.filename,
            chunk_index=i,
            text=text,
            char_start=offset,
            char_end=offset + len(text),
            content_type="table" if doc.file_type == "csv" else "text",
            metadata={"doc_file_type": doc.file_type}
        ))
        offset += len(text) + 10

    log.info("Document '%s' → %d chunks (legacy format: %s)", doc.filename, len(chunks), doc.file_type)
    return chunks


# ---------------------------------------------------------------------------
# Core Entry Point Functions
# ---------------------------------------------------------------------------

def load_document(path: Path, dual_page_layout: bool = False) -> RawDocument:
    ext = path.suffix.lower()
    log.info("Loading document: %s", path.name)
    doc_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    meta = {"source": str(path), "file_size_bytes": path.stat().st_size}

    if ext == ".pdf":
        docling_doc, full_text = _process_pdf_docling(path, doc_id)
        return RawDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type="pdf",
            raw_text=full_text,
            docling_doc=docling_doc,
            metadata=meta
        )

    extractors = {
        ".txt": _extract_text_from_txt,
        ".text": _extract_text_from_txt,
        ".json": _extract_text_from_json,
        ".csv": _extract_text_from_csv,
        ".xml": _extract_text_from_xml,
    }
    
    if ext not in extractors:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, {list(extractors.keys())}")

    raw_text = extractors[ext](path)
    return RawDocument(
        doc_id=doc_id,
        filename=path.name,
        file_type=ext.lstrip("."),
        raw_text=raw_text,
        docling_doc=None,
        metadata=meta
    )


def chunk_document(doc: RawDocument) -> list[DocumentChunk]:
    if doc.file_type == "pdf" and doc.docling_doc:
        return chunk_pdf_docling(doc)
    return chunk_legacy_formats(doc)


def process_directory(directory: Path) -> Iterator[DocumentChunk]:
    supported_exts = {".pdf", ".txt", ".text", ".json", ".csv", ".xml"}
    files = [f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in supported_exts]
    log.info("Found %d supported files in '%s'", len(files), directory)
    for file_path in files:
        try:
            doc = load_document(file_path)
            yield from chunk_document(doc)
        except Exception as exc:
            log.error("Failed to process '%s': %s", file_path.name, exc, exc_info=True)