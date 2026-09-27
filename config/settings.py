"""
settings.py — Configuration for the ESG / BRSR Report Intelligence Platform
(Hugging Face Spaces + local .env compatible).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # no-op if no .env present (e.g. HF Spaces, which injects env vars directly)


# ---------------------------------------------------------------------------
# Credentials — environment variables only. Never hardcoded, never in git.
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


@dataclass
class VectorStoreConfig:
    persist_directory: str = ""        # empty = EphemeralClient (no disk)
    collection_name: str = "esg_docs"
    distance_metric: str = "cosine"


@dataclass
class ChunkingConfig:
    # Prose chunking
    text_chunk_size: int = 700
    text_chunk_overlap: int = 100
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])

    # Table chunking — larger, because splitting rows apart destroys meaning
    table_chunk_max_chars: int = 3000      # soft cap before a table is split into sub-chunks
    table_chunk_min_rows: int = 4          # don't bother sub-splitting below this many rows
    table_row_overlap: int = 1             # repeat last N data rows across a table split for context

    use_markdown_aware_chunking: bool = True
    max_heading_levels: int = 4

    # pdfplumber table-detection tuning. None values mean "use pdfplumber's
    # own default" for that setting. These are the knobs worth adjusting
    # when tables are being missed, merged together, or split apart
    # incorrectly — see the "Table Extraction Tuning" notebook section for
    # how to find good values for a specific report's table style, and the
    # cheat-sheet at the bottom of that notebook for what each one does.
    # Passed straight through to pdfplumber's `page.find_tables(table_settings=...)`.
    table_settings: dict = field(default_factory=lambda: {
        "vertical_strategy": "text",      # "lines" | "lines_strict" | "text" | "explicit"
        "horizontal_strategy": "lines_strict",    # same options
        "snap_x_tolerance": 5,               # px: merge nearly-aligned edges within this distance
        "join_x_tolerance": 5,               # px: bridge small gaps in a ruled line
        "intersection_tolerance": 5,       # px: how exactly lines must cross to count as a cell corner
        "min_words_vertical": 8,           # only used by "text" strategy
        "min_words_horizontal": 1,         # only used by "text" strategy
    })

    # Memory controls for large PDFs (150+ pages is common for BRSR/annual
    # reports). See preprocessor._extract_elements_pdf — pdfplumber caches
    # every page's parsed objects until the file is closed, so we sample
    # for the heading-size threshold instead of scanning twice, and flush
    # each page's cache as soon as we're done with it.
    heading_detection_sample_pages: int = 15
    pdf_gc_interval_pages: int = 15

    dual_page_aspect_ratio_threshold: float = 1.25


@dataclass
class RetrievalConfig:
    # Stage 1 — bi-encoder ANN (fetch more than needed; re-ranker prunes)
    top_k: int = 12
    similarity_threshold: float = 0.25

    # Stage 2 — cross-encoder re-ranking
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 6
    rerank_score_threshold: float = -3.0
    min_top_rerank_score: float = -2.5

    # Table-aware retrieval — when a query looks metric/table-shaped, bias
    # results toward content_type == "table" chunks via a soft re-rank boost.
    table_query_boost: float = 0.08


@dataclass
class LLMConfig:
    provider: str = "groq"
    model_name: str = "openai/gpt-oss-120b"
    judge_model: str = "openai/gpt-oss-20b"
    temperature: float = 0.1
    max_tokens: int = 1400
    timeout: int = 60
    base_url: str = "https://api.groq.com/openai/v1"
    max_retries: int = 6


@dataclass
class RAGASConfig:
    enabled: bool = True
    min_faithfulness: float = 0.5
    min_answer_relevancy: float = 0.40
    min_context_relevancy: float = 0.30
    max_statements_to_check: int = 8


@dataclass
class ESGConfig:
    """BRSR principle map and document-type vocabulary used across the UI,
    metadata filters, and dashboard generation."""

    principles: dict = field(default_factory=lambda: {
        "P1": "Ethics, Transparency & Accountability",
        "P2": "Sustainable & Safe Goods/Services",
        "P3": "Employee Well-being",
        "P4": "Stakeholder Engagement",
        "P5": "Human Rights",
        "P6": "Environment Protection & Restoration",
        "P7": "Public Policy Advocacy",
        "P8": "Inclusive Growth & Equitable Development",
        "P9": "Consumer Value & Responsibility",
    })

    document_types: list = field(default_factory=lambda: [
        "BRSR Report", "ESG Report", "Sustainability Report",
        "Annual Report", "Other",
    ])

    # Section labels commonly found in ESG/BRSR reports — used to bucket
    # chunks for the Summarise feature even when principle tagging misses.
    section_labels: list = field(default_factory=lambda: [
        "Environmental Performance", "Social Performance", "Governance",
        "Key ESG Metrics", "Material Issues", "Targets and Commitments",
        "Risks", "Executive Overview",
    ])


@dataclass
class DashboardConfig:
    min_metrics_for_chart: int = 2
    max_companies_per_chart: int = 6
    max_years_per_chart: int = 6
    # units considered compatible enough to auto-align/combine
    compatible_unit_groups: list = field(default_factory=lambda: [
        {"tco2e", "tco2", "tonnes co2e", "t co2e", "mtco2e"},
        {"kl", "kilolitres", "kilolitre", "m3", "cubic metres"},
        {"mwh", "gj", "gwh", "kwh"},
        {"%", "percent", "percentage"},
        {"mt", "tonnes", "metric tonnes", "kg"},
    ])


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    log_file: str = "/tmp/esg_platform.log"


@dataclass
class EvaluationConfig:
    test_dataset_path: str = ""
    results_path: str = "/tmp/eval_results.json"


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ragas: RAGASConfig = field(default_factory=RAGASConfig)
    esg: ESGConfig = field(default_factory=ESGConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


cfg = Config()
