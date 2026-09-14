"""
history_store.py — Persistent memory for the Compliance Sentinel agent.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_LOCK = threading.Lock()


class FilingHistoryStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or os.getenv("AGENT_HISTORY_PATH", "data/agent_history.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"filings": [], "findings": [], "memos": []})

    # ---- low-level read/write ----
    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"filings": [], "findings": [], "memos": []}

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ---- filing snapshots (metric registry at time of review) ----
    def save_filing_snapshot(self, company: str, reporting_year: str, document_type: str,
                              filename: str, metrics: list[dict], principles_present: list[str]) -> None:
        with _LOCK:
            data = self._read()
            data["filings"] = [
                f for f in data["filings"]
                if not (f["company"] == company and f["reporting_year"] == reporting_year)
            ]
            data["filings"].append({
                "company": company, "reporting_year": reporting_year,
                "document_type": document_type, "filename": filename,
                "metrics": metrics, "principles_present": principles_present,
                "reviewed_at": self._now(),
            })
            self._write(data)

    def get_prior_filing(self, company: str, exclude_year: str) -> Optional[dict]:
        """Most recently reviewed filing for this company, excluding the year
        currently being reviewed. Returns None for a company's first filing."""
        data = self._read()
        candidates = [
            f for f in data["filings"]
            if f["company"] == company and f["reporting_year"] != exclude_year
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda f: f["reviewed_at"], reverse=True)
        return candidates[0]

    def list_filings(self, company: Optional[str] = None) -> list[dict]:
        data = self._read()
        filings = data["filings"]
        if company:
            filings = [f for f in filings if f["company"] == company]
        return sorted(filings, key=lambda f: f["reviewed_at"], reverse=True)

    # ---- findings (issues the agent flags during review) ----
    def add_finding(self, company: str, reporting_year: str, topic: str, severity: str,
                     description: str, citation: str = "") -> None:
        with _LOCK:
            data = self._read()
            data["findings"].append({
                "company": company, "reporting_year": reporting_year, "topic": topic,
                "severity": severity, "description": description, "citation": citation,
                "flagged_at": self._now(),
            })
            self._write(data)

    def get_findings(self, company: str, reporting_year: str) -> list[dict]:
        data = self._read()
        return [
            f for f in data["findings"]
            if f["company"] == company and f["reporting_year"] == reporting_year
        ]

    # ---- memos ----
    def save_memo(self, company: str, reporting_year: str, memo_markdown: str, status: str) -> str:
        memo_dir = self.path.parent / "memos"
        memo_dir.mkdir(parents=True, exist_ok=True)
        safe_company = "".join(c if c.isalnum() else "_" for c in company)
        memo_path = memo_dir / f"{safe_company}_{reporting_year}_review.md"
        memo_path.write_text(memo_markdown, encoding="utf-8")

        with _LOCK:
            data = self._read()
            data["memos"] = [
                m for m in data["memos"]
                if not (m["company"] == company and m["reporting_year"] == reporting_year)
            ]
            data["memos"].append({
                "company": company, "reporting_year": reporting_year,
                "status": status, "path": str(memo_path), "saved_at": self._now(),
            })
            self._write(data)
        return str(memo_path)

    def get_memo(self, company: str, reporting_year: str) -> Optional[dict]:
        data = self._read()
        for m in data["memos"]:
            if m["company"] == company and m["reporting_year"] == reporting_year:
                return m
        return None

    # ---- notifications (what actually reached a human) ----
    def add_notification(self, company: str, reporting_year: str, severity: str, summary: str) -> None:
        notif_path = self.path.parent / "notifications.json"
        with _LOCK:
            try:
                notifs = json.loads(notif_path.read_text(encoding="utf-8")) if notif_path.exists() else []
            except json.JSONDecodeError:
                notifs = []
            notifs.append({
                "company": company, "reporting_year": reporting_year,
                "severity": severity, "summary": summary, "notified_at": self._now(),
            })
            notif_path.write_text(json.dumps(notifs, indent=2), encoding="utf-8")