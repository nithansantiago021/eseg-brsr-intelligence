"""
history_store.py — Session-isolated in-memory history store resilient to background threads.
"""

from __future__ import annotations
import threading
from datetime import datetime, timezone
from typing import Optional

_LOCK = threading.Lock()


class FilingHistoryStore:
    """Session-isolated in-memory history store with fallback for background threads."""
    
    def __init__(self, *args, **kwargs):
        self._fallback_data = {
            "filings": [],
            "findings": [],
            "memos": [],
            "notifications": []
        }

    def _read(self) -> dict:
        try:
            import streamlit as st
            if hasattr(st, "session_state") and "agent_history_data" in st.session_state:
                return st.session_state.agent_history_data
        except Exception:
            pass
        return self._fallback_data

    def _write(self, data: dict) -> None:
        try:
            import streamlit as st
            if hasattr(st, "session_state"):
                st.session_state.agent_history_data = data
                return
        except Exception:
            pass
        self._fallback_data = data

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_filing_snapshot(self, company: str, reporting_year: str, document_type: str,
                             filename: str, metrics: list[dict], principles_present: list[str]) -> None:
        with _LOCK:
            data = self._read()
            data["filings"] = [
                f for f in data["filings"]
                if not (f["company"].lower() == company.lower() and f["reporting_year"] == reporting_year)
            ]
            data["filings"].append({
                "company": company,
                "reporting_year": reporting_year,
                "document_type": document_type,
                "filename": filename,
                "metrics": metrics,
                "principles_present": principles_present,
                "saved_at": self._now()
            })
            self._write(data)

    def list_filings(self) -> list[dict]:
        return self._read().get("filings", [])

    def get_prior_filing(self, company: str, exclude_year: str) -> Optional[dict]:
        filings = self._read().get("filings", [])
        matching = [
            f for f in filings 
            if f["company"].lower() == company.lower() and f["reporting_year"] != exclude_year
        ]
        if not matching:
            return None
        return sorted(matching, key=lambda x: x.get("saved_at", ""), reverse=True)[0]

    def add_finding(self, company: str, reporting_year: str, topic: str, severity: str,
                     description: str, citation: str = "") -> None:
        with _LOCK:
            data = self._read()
            if "findings" not in data:
                data["findings"] = []
            data["findings"].append({
                "company": company, "reporting_year": reporting_year, "topic": topic,
                "severity": severity, "description": description, "citation": citation,
                "flagged_at": self._now(),
            })
            self._write(data)

    def clear_findings(self, company: str, reporting_year: str) -> None:
        with _LOCK:
            data = self._read()
            data["findings"] = [
                f for f in data.get("findings", [])
                if not (f["company"].lower() == company.lower() and f["reporting_year"] == reporting_year)
            ]
            self._write(data)

    def get_findings(self, company: str, reporting_year: str) -> list[dict]:
        data = self._read().get("findings", [])
        return [
            f for f in data
            if f["company"].lower() == company.lower() and f["reporting_year"] == reporting_year
        ]

    def save_memo(self, company: str, reporting_year: str, memo_markdown: str, status: str) -> str:
        safe_company = "".join(c if c.isalnum() else "_" for c in company)
        memo_path = f"memos/{safe_company}_{reporting_year}_review.md"

        with _LOCK:
            data = self._read()
            if "memos" not in data:
                data["memos"] = []
            data["memos"] = [
                m for m in data["memos"]
                if not (m["company"].lower() == company.lower() and m["reporting_year"] == reporting_year)
            ]
            data["memos"].append({
                "company": company, 
                "reporting_year": reporting_year,
                "status": status, 
                "path": memo_path, 
                "markdown_content": memo_markdown,
                "saved_at": self._now(),
            })
            self._write(data)
        return memo_path

    def get_memo(self, company: str, reporting_year: str) -> Optional[dict]:
        data = self._read().get("memos", [])
        for m in data:
            if m["company"].lower() == company.lower() and m["reporting_year"] == reporting_year:
                return m
        return None

    def add_notification(self, company: str, reporting_year: str, severity: str, summary: str) -> None:
        with _LOCK:
            data = self._read()
            if "notifications" not in data:
                data["notifications"] = []
            data["notifications"].append({
                "company": company, 
                "reporting_year": reporting_year,
                "severity": severity, 
                "summary": summary, 
                "notified_at": self._now(),
            })
            self._write(data)
