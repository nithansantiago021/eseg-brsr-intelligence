"""
preprocessor.py — Structured document extraction + markdown-aware chunking
for the ESG / BRSR Report Intelligence Platform.
"""

from __future__ import annotations

import sys
import re
import json
import csv
import gc
import hashlib
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass, field
from collections import Counter

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DocumentElement:
    """One structural unit extracted from a document, in reading order."""
    element_type: str          # "heading" | "paragraph" | "table"
    text: str                  # markdown representation
    page: int
    level: int = 0             # heading level (1 = top), 0 for non-headings
    section: str = ""          # breadcrumb, e.g. "Principle 6 > Environment"
    principle: str = ""        # "P1".."P9" or ""
    table_title: str = ""      # best-effort caption preceding the table
    col_headers: list = field(default_factory=list)


@dataclass
class RawDocument:
    doc_id: str
    filename: str
    file_type: str
    raw_text: str = ""                       # used by non-PDF formats
    elements: Optional[list] = None          # list[DocumentElement], PDFs only
    metadata: dict = field(default_factory=dict)


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
    page: str = ""                 # "12" or "12-13" for multi-page table splits
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


# ---------------------------------------------------------------------------
# BRSR principle / section detection
# ---------------------------------------------------------------------------

_PRINCIPLE_RE = re.compile(r"\bPRINCIPLE\s*[-:]?\s*([1-9])\b", re.IGNORECASE)

_SECTION_HEADING_HINTS = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+[A-Z]"
    r"|SECTION\s+[A-Z0-9]"
    r"|ESSENTIAL\s+INDICATORS"
    r"|LEADERSHIP\s+INDICATORS"
    r"|(?:[A-Z][A-Z &,/]{4,})\s*$"
    r")",
    re.MULTILINE,
)


def _detect_principle(text: str) -> str:
    m = _PRINCIPLE_RE.search(text)
    return f"P{m.group(1)}" if m else ""


# ---------------------------------------------------------------------------
# PDF structured extraction (pdfplumber)
# ---------------------------------------------------------------------------

def _merge_wrapped_rows(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows
    merged = [rows[0]]
    for row in rows[1:]:
        label_cell = (row[0] or "").strip()
        other_cells = [(c or "").strip() for c in row[1:]]
        has_other_content = any(other_cells)

        label_only_continuation = bool(label_cell) and not has_other_content
        value_only_continuation = (not label_cell) and has_other_content
        is_continuation = merged and (label_only_continuation or value_only_continuation)

        if is_continuation:
            prev = merged[-1]
            new_prev = list(prev)
            for i in range(len(new_prev)):
                addition = row[i].strip() if i < len(row) and row[i] else ""
                if addition:
                    new_prev[i] = f"{new_prev[i]} {addition}".strip() if new_prev[i] else addition
            merged[-1] = new_prev
        else:
            merged.append(row)
    return merged


def _table_to_markdown(rows: list[list]) -> tuple[str, list[str]]:
    rows = [[("" if c is None else str(c).strip().replace("\n", " ")) for c in r] for r in rows]
    rows = [r for r in rows if any(cell for cell in r)]
    if not rows:
        return "", []
    rows = _merge_wrapped_rows(rows)

    headers = rows[0]
    ncols = len(headers)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * ncols) + " |"]
    for r in rows[1:]:
        r = (r + [""] * ncols)[:ncols]
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines), headers


def _flush_page_cache(page) -> None:
    try:
        page.flush_cache()
    except AttributeError:
        for attr in ("_objects", "_layout", "chars", "lines", "rects", "curves", "images"):
            if attr in getattr(page, "__dict__", {}):
                page.__dict__[attr] = None


def _sample_heading_threshold(pdf, sample_pages: int) -> float:
    all_sizes = []
    n = min(sample_pages, len(pdf.pages))
    for page in pdf.pages[:n]:
        try:
            for w in page.extract_words(extra_attrs=["size"]):
                all_sizes.append(w.get("size", 10))
        finally:
            _flush_page_cache(page)
    median_size = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 10.0
    return median_size * 1.15


def _detect_page_gutter(page, require_aspect_gate: bool = True) -> float | None:
    if require_aspect_gate:
        try:
            aspect = page.width / page.height if page.height else 0
        except Exception:
            return None
        if aspect < cfg.chunking.dual_page_aspect_ratio_threshold:
            return None

    try:
        words = page.extract_words()
    except Exception:
        return None
    if len(words) < 4:
        return None

    center = page.width / 2.0
    search_lo, search_hi = center - page.width * 0.15, center + page.width * 0.15
    xs = sorted((w["x0"] + w["x1"]) / 2.0 for w in words)

    best_gap, best_mid = 0.0, None
    for a, b in zip(xs, xs[1:]):
        mid = (a + b) / 2.0
        if search_lo <= mid <= search_hi:
            gap = b - a
            if gap > best_gap:
                best_gap, best_mid = gap, mid

    # require a meaningfully empty gutter (not just normal word spacing)
    # before trusting it as a real page boundary
    if best_mid is not None and best_gap > page.width * 0.025:
        return best_mid
    return None


def _get_page_regions(page, force_split: bool = False) -> list:
    if not force_split:
        return [page]

    gutter_x = _detect_page_gutter(page, require_aspect_gate=False)
    split_x = gutter_x if gutter_x is not None else (page.width / 2.0)
    try:
        left = page.crop((0, 0, split_x, page.height))
        right = page.crop((split_x, 0, page.width, page.height))
        return [left, right]
    except Exception:
        return [page]


_NUMBERED_HEADING_BREAK_RE = re.compile(r"^\s*(?:\d{1,2}|[IVXLC]{1,4})[.)]\s+[A-Z]", re.MULTILINE)


def _get_ruled_rows(page, top_tol: float = 0.75) -> list[dict]:
    raw = [{"top": l["top"], "x0": l["x0"], "x1": l["x1"]} for l in page.lines if abs(l["top"] - l["bottom"]) < 0.5]
    raw += [{"top": r["top"], "x0": r["x0"], "x1": r["x1"]} for r in page.rects if r["height"] < 1.5]
    raw.sort(key=lambda l: l["top"])
    rows: list[dict] = []
    cur_top, cur_segs = None, []
    for l in raw:
        if cur_top is None or l["top"] - cur_top > top_tol:
            if cur_segs:
                rows.append({"top": cur_top, "segs": cur_segs})
            cur_top, cur_segs = l["top"], [l]
        else:
            cur_segs.append(l)
    if cur_segs:
        rows.append({"top": cur_top, "segs": cur_segs})
    return rows


def _cluster_ruled_rows(page, rows: list[dict], region_x0: float, region_x1: float,
                         max_gap: float, min_rows: int = 2) -> list[list[dict]]:
    if not rows:
        return []
    clusters: list[list[dict]] = []
    cur = [rows[0]]
    for prev, cur_r in zip(rows, rows[1:]):
        gap = cur_r["top"] - prev["top"]
        heading_hit = False
        if gap > 3:
            try:
                region = page.within_bbox((region_x0, prev["top"] + 1, region_x1, cur_r["top"] - 1), strict=False)
                gap_text = (region.extract_text() or "").strip()
                match = _NUMBERED_HEADING_BREAK_RE.search(gap_text) if gap_text else None
                if match and len(cur) >= 2:
                    established_x0 = min(s["x0"] for r in cur for s in r["segs"])
                    gap_words = sorted(region.extract_words(), key=lambda w: (w["top"], w["x0"]))
                    lead_word = gap_words[0] if gap_words else None
                    if lead_word and abs(lead_word["x0"] - established_x0) < 8:
                        match = None  # same column as the table's own rows -> not a real heading
                heading_hit = bool(match)
            except Exception:
                pass
        if gap > max_gap or heading_hit:
            clusters.append(cur)
            cur = []
        cur.append(cur_r)
    clusters.append(cur)
    return [c for c in clusters if len(c) >= min_rows]


def _column_boundaries_from_segments(cluster: list[dict], tol: float = 1.5) -> list[float]:
    n_rows = len(cluster)
    counts = Counter()
    for row in cluster:
        for seg in row["segs"]:
            counts[round(seg["x0"] / tol) * tol] += 1
            counts[round(seg["x1"] / tol) * tol] += 1
    keep = sorted(x for x, c in counts.items() if c >= max(2, n_rows * 0.5))
    merged: list[float] = []
    for x in keep:
        if merged and x - merged[-1] < 3:
            continue
        merged.append(x)
    return merged


def _cluster_words_into_lines(words: list[dict], y_tol: float = 3.0) -> list[list[dict]]:
    words = sorted(words, key=lambda w: w["top"])
    lines: list[list[dict]] = []
    cur_top, cur = None, []
    for w in words:
        if cur_top is None or w["top"] - cur_top > y_tol:
            if cur:
                lines.append(cur)
            cur_top, cur = w["top"], [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def _line_words_to_cells(line_words: list[dict], x_gap: float = 10.0) -> list[dict]:
    line_words = sorted(line_words, key=lambda w: w["x0"])
    cells: list[list[dict]] = []
    cur = [line_words[0]]
    for prev, w in zip(line_words, line_words[1:]):
        if w["x0"] - prev["x1"] > x_gap:
            cells.append(cur)
            cur = []
        cur.append(w)
    cells.append(cur)
    return [{"x0": min(w["x0"] for w in c), "x1": max(w["x1"] for w in c),
              "text": " ".join(w["text"] for w in c)} for c in cells]


def _detect_tables_by_word_geometry(page, x_tol: float, min_rows: int, min_cols: int) -> list[dict]:
    words = page.extract_words()
    if not words:
        return []
    line_cells = [_line_words_to_cells(l) for l in _cluster_words_into_lines(words)]
    tops = [min(min(w["top"] for w in l) for l in [line]) for line in _cluster_words_into_lines(words)]

    candidates: list[dict] = []
    i = 0
    while i < len(line_cells):
        if len(line_cells[i]) < min_cols:
            i += 1
            continue
        run = [i]
        col_starts = [c["x0"] for c in line_cells[i]]
        j = i + 1
        while j < len(line_cells):
            starts = [c["x0"] for c in line_cells[j]]
            if len(starts) < min_cols:
                break
            aligned = sum(1 for s in starts if any(abs(s - b) <= x_tol for b in col_starts))
            if aligned < max(1, min(len(starts), len(col_starts)) * 0.5):
                break
            run.append(j)
            for s in starts:
                if not any(abs(s - b) <= x_tol for b in col_starts):
                    col_starts.append(s)
            j += 1
        if len(run) >= min_rows:
            col_starts = sorted(col_starts)
            merged_cols: list[float] = []
            for s in col_starts:
                if merged_cols and s - merged_cols[-1] < x_tol:
                    continue
                merged_cols.append(s)
            rows_out = []
            for ridx in run:
                bins = [""] * len(merged_cols)
                for c in line_cells[ridx]:
                    nearest = min(range(len(merged_cols)), key=lambda k: abs(merged_cols[k] - c["x0"]))
                    bins[nearest] = f"{bins[nearest]} {c['text']}".strip() if bins[nearest] else c["text"]
                rows_out.append(bins)
            xall = [c["x0"] for ridx in run for c in line_cells[ridx]] + \
                   [c["x1"] for ridx in run for c in line_cells[ridx]]
            bbox = (min(xall) - 2, tops[run[0]] - 2, max(xall) + 2, tops[run[-1]] + 15)
            candidates.append({"strategy": "geometric", "bbox": bbox,
                                "rows": _clean_table_rows(rows_out), "col_xs": merged_cols})
            i = run[-1] + 1
        else:
            i += 1
    return candidates


def _clean_table_cell(cell) -> str:
    if cell is None:
        return ""
    cell = re.sub(r"\s+", " ", str(cell).replace("\n", " "))
    return cell.strip()


def _clean_table_rows(rows: list[list]) -> list[list[str]]:
    cleaned = []
    for row in rows:
        row = [_clean_table_cell(c) for c in row]
        if any(row):
            cleaned.append(row)
    return cleaned


def _score_table_candidate(rows: list[list[str]], bbox: tuple) -> float:
    if not rows:
        return -999.0
    row_count = len(rows)
    col_count = max((len(r) for r in rows), default=0)
    non_empty = sum(1 for r in rows for c in r if c)
    score = 0.0
    if row_count >= 3:
        score += 10
    if row_count >= 5:
        score += 10
    if row_count >= 10:
        score += 10
    if col_count >= 2:
        score += 10
    if col_count >= 3:
        score += 10
    score += min(non_empty, 60)
    x0, top, x1, bottom = bbox
    if x1 - x0 > 200:
        score += 5
    if bottom - top > 100:
        score += 5
    return score


def _bbox_iou(b1: tuple, b2: tuple) -> float:
    x0a, y0a, x1a, y1a = b1
    x0b, y0b, x1b, y1b = b2
    ix0, iy0 = max(x0a, x0b), max(y0a, y0b)
    ix1, iy1 = min(x1a, x1b), min(y1a, y1b)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (x1a - x0a) * (y1a - y0a)
    area_b = (x1b - x0b) * (y1b - y0b)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _dedupe_table_candidates(candidates: list[dict], iou_threshold: float) -> list[dict]:
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    selected: list[dict] = []
    for c in candidates:
        if not any(_bbox_iou(c["bbox"], s["bbox"]) > iou_threshold for s in selected):
            selected.append(c)
    return selected


def _detect_tables(page) -> list[dict]:
    candidates: list[dict] = []

    ruled_rows = _get_ruled_rows(page)
    if ruled_rows:
        region_x0, _, region_x1, _ = getattr(page, "bbox", (0, 0, page.width, page.height))
        clusters = _cluster_ruled_rows(page, ruled_rows, region_x0, region_x1,
                                        max_gap=cfg.chunking.table_region_max_gap)
        for cluster in clusters:
            tops = [r["top"] for r in cluster]
            xall = [s["x0"] for r in cluster for s in r["segs"]] + [s["x1"] for r in cluster for s in r["segs"]]
            # Pad the top generously: a header row's own text often sits
            # above the topmost rule with nothing separating it further up.
            # The bottom is extended dynamically to the actual extent of any
            # nearby text — a fixed pad (e.g. +20pt) is enough for a normal
            # single-line last row, but a wrapped multi-line last cell (a
            # real, visually-confirmed case: a "Reporting boundary..." cell
            # spanning 4 lines) needs more room than any fixed constant
            # reliably covers.
            bottom_search_limit = max(tops) + 120
            try:
                nearby_words = [w for w in page.extract_words()
                                 if max(tops) - 5 <= w["top"] <= bottom_search_limit
                                 and min(xall) - 5 <= w["x0"] <= max(xall) + 5]
                dynamic_bottom = max((w["bottom"] for w in nearby_words), default=max(tops) + 20) + 3
            except Exception:
                dynamic_bottom = max(tops) + 20
            bbox = (min(xall) - 2, min(tops) - 20, max(xall) + 2, dynamic_bottom)
            try:
                cropped = page.within_bbox(bbox, strict=False)
            except Exception:
                continue

            region_candidates: list[dict] = []
            try:
                for t in cropped.find_tables(table_settings=cfg.chunking.table_line_settings):
                    rows = _clean_table_rows(t.extract())
                    if rows:
                        region_candidates.append({"strategy": "ruled", "bbox": t.bbox, "rows": rows, "col_xs": None})
            except Exception:
                pass
            try:
                for t in cropped.find_tables(table_settings=cfg.chunking.table_borderless_settings):
                    rows = _clean_table_rows(t.extract())
                    if rows:
                        region_candidates.append({"strategy": "borderless", "bbox": t.bbox, "rows": rows, "col_xs": None})
            except Exception:
                pass
            xs = _column_boundaries_from_segments(cluster)
            if len(xs) >= 3:
                try:
                    for t in cropped.find_tables(table_settings={
                        "vertical_strategy": "explicit", "explicit_vertical_lines": xs,
                        "horizontal_strategy": "lines_strict",
                    }):
                        rows = _clean_table_rows(t.extract())
                        if rows:
                            region_candidates.append({"strategy": "semi_ruled", "bbox": t.bbox, "rows": rows, "col_xs": xs})
                except Exception:
                    pass

            if region_candidates:
                expected_width = max(xall) - min(xall)
                for c in region_candidates:
                    c["score"] = _score_table_candidate(c["rows"], c["bbox"])
                    cand_width = c["bbox"][2] - c["bbox"][0]
                    if expected_width > 0 and cand_width < expected_width * 0.85:
                        c["score"] *= 0.5
                candidates.append(max(region_candidates, key=lambda c: c["score"]))
    else:
        geom = _detect_tables_by_word_geometry(
            page, x_tol=cfg.chunking.table_geom_x_tolerance,
            min_rows=cfg.chunking.table_geom_min_rows, min_cols=cfg.chunking.table_geom_min_cols,
        )
        for c in geom:
            c["score"] = _score_table_candidate(c["rows"], c["bbox"])
        candidates.extend(geom)

    candidates = [c for c in candidates if c["score"] >= cfg.chunking.table_min_score]
    return _dedupe_table_candidates(candidates, cfg.chunking.table_dedup_iou_threshold)


def debug_visualize_tables(page, candidates: Optional[list[dict]] = None, resolution: int = 150):
    if candidates is None:
        candidates = _detect_tables(page)
    im = page.to_image(resolution=resolution)
    for c in candidates:
        im.draw_rect(c["bbox"], stroke="red", stroke_width=2, fill=None)
        col_xs = c.get("col_xs")
        if col_xs:
            x0, top, x1, bottom = c["bbox"]
            for x in col_xs:
                im.draw_line([(x, top), (x, bottom)], stroke="blue", stroke_width=1)
    return im


def _extract_elements_pdf(path: Path, dual_page_layout: bool = False) -> tuple[list[DocumentElement], str]:
    elements: list[DocumentElement] = []
    full_text_parts: list[str] = []
    current_principle = ""
    heading_stack: list[tuple[int, str]] = []   # (level, text)

    def _section_breadcrumb() -> str:
        return " > ".join(t for _, t in heading_stack)

    with pdfplumber.open(str(path)) as pdf:
        heading_size_threshold = _sample_heading_threshold(
            pdf, cfg.chunking.heading_detection_sample_pages
        )

        for page_num, page in enumerate(pdf.pages, start=1):
          regions = _get_page_regions(page, force_split=dual_page_layout)
          if len(regions) > 1:
              log.debug("Page %d detected as dual-page spread (aspect %.2f) — split left/right",
                        page_num, page.width / page.height if page.height else 0)

          for region in regions:
            # ---- Tables ----
            table_bboxes = []
            try:
                detected_tables = _detect_tables(region)
            except Exception:
                detected_tables = []

            for cand in detected_tables:
                md, headers = _table_to_markdown(cand["rows"])
                if not md:
                    continue
                table_bboxes.append(cand["bbox"])
                principle = _detect_principle(_section_breadcrumb()) or current_principle
                elements.append(DocumentElement(
                    element_type="table",
                    text=md,
                    page=page_num,
                    section=_section_breadcrumb(),
                    principle=principle,
                    table_title="",
                    col_headers=headers,
                ))
                full_text_parts.append(md)
            del detected_tables

            # ---- Prose (excluding table regions where possible) ----
            prose_page = region
            for bbox in table_bboxes:
                try:
                    prose_page = prose_page.outside_bbox(bbox)
                except Exception:
                    pass

            try:
                words = prose_page.extract_words(extra_attrs=["size", "top"])
            except Exception:
                words = []

            # group words into lines by rounded 'top' coordinate (safe now —
            # each region only contains words from its own logical page)
            lines: dict[int, list] = {}
            for w in words:
                key = round(w["top"] / 3)
                lines.setdefault(key, []).append(w)

            for key in sorted(lines.keys()):
                line_words = sorted(lines[key], key=lambda w: w["x0"])
                line_text = " ".join(w["text"] for w in line_words).strip()
                if not line_text:
                    continue
                max_size = max(w.get("size", 10) for w in line_words)
                is_heading = (
                    max_size >= heading_size_threshold
                    and len(line_text) < 100
                    and line_text.count(".") <= 1
                ) or bool(_PRINCIPLE_RE.search(line_text)) or bool(_SECTION_HEADING_HINTS.match(line_text))

                principle_hit = _detect_principle(line_text)
                if principle_hit:
                    current_principle = principle_hit

                if is_heading:
                    level = 1 if (_PRINCIPLE_RE.search(line_text) or line_text.isupper()) else 2
                    heading_stack = [h for h in heading_stack if h[0] < level]
                    heading_stack.append((level, line_text[:120]))
                    elements.append(DocumentElement(
                        element_type="heading",
                        text=line_text,
                        page=page_num,
                        level=level,
                        section=_section_breadcrumb(),
                        principle=current_principle,
                    ))
                else:
                    elements.append(DocumentElement(
                        element_type="paragraph",
                        text=line_text,
                        page=page_num,
                        section=_section_breadcrumb(),
                        principle=current_principle,
                    ))
                full_text_parts.append(line_text)

            del words, lines

          _flush_page_cache(page)
          if page_num % cfg.chunking.pdf_gc_interval_pages == 0:
              gc.collect()

    log.info("PDF '%s' → %d structural elements", path.name, len(elements))
    return elements, "\n".join(full_text_parts)


# ---------------------------------------------------------------------------
# Other format extraction (unchanged from legacy pipeline)
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
            tag = elem.tag.split("}")[-1]  # strip XML namespace prefix
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


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_document(path: Path, dual_page_layout: bool = False) -> RawDocument:
    ext = path.suffix.lower()
    log.info("Loading document: %s", path.name)
    doc_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    meta = {"source": str(path), "file_size_bytes": path.stat().st_size}

    if ext == ".pdf":
        elements, raw_text = _extract_elements_pdf(path, dual_page_layout=dual_page_layout)
        return RawDocument(doc_id=doc_id, filename=path.name, file_type="pdf",
                            raw_text=raw_text, elements=elements, metadata=meta)

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
    return RawDocument(doc_id=doc_id, filename=path.name, file_type=ext.lstrip("."),
                        raw_text=raw_text, elements=None, metadata=meta)


# ---------------------------------------------------------------------------
# Markdown-aware chunking (PDFs with structured elements)
# ---------------------------------------------------------------------------

def _split_table_rows(md_table: str, max_chars: int, min_rows: int, overlap_rows: int) -> list[str]:
    lines = md_table.split("\n")
    if len(lines) < 2:
        return [md_table]
    header, sep = lines[0], lines[1]
    data_rows = lines[2:]
    n = len(data_rows)

    if len(md_table) <= max_chars or n < min_rows:
        return [md_table]

    splits = []
    i = 0
    max_splits = max(n, 1) + 10  # can never legitimately need more splits than rows

    while i < n and len(splits) < max_splits:
        start = i
        chunk_rows = []
        chars = len(header) + len(sep)
        while i < n and (chars < max_chars or not chunk_rows):
            chunk_rows.append(data_rows[i])
            chars += len(data_rows[i])
            i += 1
        splits.append("\n".join([header, sep] + chunk_rows))

        advanced = i - start
        # Only back off for overlap if doing so still leaves forward progress.
        back = min(overlap_rows, advanced - 1) if advanced > 1 else 0
        i -= back
        if i <= start:       # guarantee: always move forward by at least 1 row
            i = start + 1

    if len(splits) >= max_splits:
        log.warning("Table split hit safety cap (%d rows) — table may be truncated", n)

    return splits if splits else [md_table]


def chunk_document_markdown(doc: RawDocument) -> list[DocumentChunk]:
    cc = cfg.chunking
    chunks: list[DocumentChunk] = []
    buffer_text: list[str] = []
    buffer_section = ""
    buffer_principle = ""
    buffer_pages: list[int] = []
    offset = 0

    def _flush_prose():
        nonlocal buffer_text, buffer_pages, offset
        if not buffer_text:
            return
        text = "\n".join(buffer_text).strip()
        if not text:
            buffer_text, buffer_pages = [], []
            return
        page_str = f"{min(buffer_pages)}" if len(set(buffer_pages)) <= 1 else f"{min(buffer_pages)}-{max(buffer_pages)}"
        chunk_id = hashlib.sha256(f"{doc.doc_id}:{len(chunks)}".encode()).hexdigest()[:16]
        chunks.append(DocumentChunk(
            chunk_id=chunk_id, doc_id=doc.doc_id, filename=doc.filename,
            chunk_index=len(chunks), text=text,
            char_start=offset, char_end=offset + len(text),
            section=buffer_section, content_type="text",
            principle=buffer_principle, page=page_str,
            metadata={"doc_file_type": doc.file_type},
        ))
        offset += len(text) + 1
        buffer_text, buffer_pages = [], []

    for el in doc.elements or []:
        if el.element_type == "heading":
            _flush_prose()
            buffer_section, buffer_principle = el.section, el.principle
            continue

        if el.element_type == "table":
            _flush_prose()
            splits = _split_table_rows(
                el.text, cc.table_chunk_max_chars, cc.table_chunk_min_rows, cc.table_row_overlap
            )
            for s in splits:
                header_ctx = f"[Section: {el.section or 'N/A'}]"
                if el.principle:
                    header_ctx += f" [{el.principle}]"
                enriched = f"{header_ctx}\n{s}"
                chunk_id = hashlib.sha256(f"{doc.doc_id}:{len(chunks)}".encode()).hexdigest()[:16]
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id, doc_id=doc.doc_id, filename=doc.filename,
                    chunk_index=len(chunks), text=enriched,
                    char_start=offset, char_end=offset + len(enriched),
                    section=el.section, content_type="table",
                    principle=el.principle, page=str(el.page),
                    table_title=el.table_title,
                    metadata={"doc_file_type": doc.file_type, "col_headers": ",".join(el.col_headers[:12])},
                ))
                offset += len(enriched) + 1
            continue

        # paragraph
        buffer_section, buffer_principle = el.section or buffer_section, el.principle or buffer_principle
        buffer_pages.append(el.page)
        buffer_text.append(el.text)
        current_len = sum(len(t) for t in buffer_text)
        if current_len >= cc.text_chunk_size:
            _flush_prose()

    _flush_prose()
    log.info("Document '%s' → %d markdown-aware chunks", doc.filename, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Legacy chunking (txt / json / csv — no structural elements available)
# ---------------------------------------------------------------------------

_NOISE_PATTERNS = [
    (re.compile(r"\f"), " "),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
    (re.compile(r"[-–—]{3,}"), "---"),
    (re.compile(r"\x00"), ""),
    (re.compile(r"[\u200b-\u200d\ufeff]"), ""),
]


def clean_text(raw_text: str) -> str:
    text = raw_text
    for pattern, replacement in _NOISE_PATTERNS:
        text = pattern.sub(replacement, text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def chunk_document_legacy(doc: RawDocument) -> list[DocumentChunk]:
    if doc.file_type == "csv":
        chunks: list[DocumentChunk] = []
        raw_chunks = doc.raw_text.split("\n\n---\n\n")
        offset = 0
        for i, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            chunk_id = hashlib.sha256(f"{doc.doc_id}:{i}".encode()).hexdigest()[:16]
            chunks.append(DocumentChunk(
                chunk_id=chunk_id, doc_id=doc.doc_id, filename=doc.filename,
                chunk_index=i, text=chunk_text, char_start=offset,
                char_end=offset + len(chunk_text), content_type="table",
                metadata={"doc_file_type": doc.file_type},
            ))
            offset += len(chunk_text) + 9
        log.info("Document '%s' → %d chunks (legacy/csv)", doc.filename, len(chunks))
        return chunks

    clean = clean_text(doc.raw_text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunking.text_chunk_size,
        chunk_overlap=cfg.chunking.text_chunk_overlap,
        separators=cfg.chunking.separators,
        length_function=len,
        add_start_index=True,
    )
    lc_chunks = splitter.create_documents(texts=[clean], metadatas=[{"filename": doc.filename}])

    chunks = []
    for i, lc_chunk in enumerate(lc_chunks):
        char_start = lc_chunk.metadata.get("start_index", 0)
        text = lc_chunk.page_content
        chunk_id = hashlib.sha256(f"{doc.doc_id}:{i}".encode()).hexdigest()[:16]
        chunks.append(DocumentChunk(
            chunk_id=chunk_id, doc_id=doc.doc_id, filename=doc.filename,
            chunk_index=i, text=text, char_start=char_start,
            char_end=char_start + len(text), content_type="text",
            metadata={"doc_file_type": doc.file_type},
        ))
    log.info("Document '%s' → %d chunks (legacy)", doc.filename, len(chunks))
    return chunks


def chunk_document(doc: RawDocument) -> list[DocumentChunk]:
    if doc.file_type == "pdf" and doc.elements and cfg.chunking.use_markdown_aware_chunking:
        return chunk_document_markdown(doc)
    return chunk_document_legacy(doc)


def process_directory(directory: Path) -> Iterator[DocumentChunk]:
    supported_exts = {".pdf", ".txt", ".text", ".json", ".csv", ".xml"}
    files = [f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in supported_exts]
    log.info("Found %d files in '%s'", len(files), directory)
    for file_path in files:
        try:
            doc = load_document(file_path)
            yield from chunk_document(doc)
        except Exception as exc:
            log.error("Failed to process '%s': %s", file_path.name, exc, exc_info=True)