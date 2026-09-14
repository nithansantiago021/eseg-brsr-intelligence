from __future__ import annotations
import streamlit as st
from datetime import datetime
from typing import Optional, Any

class FilingHistoryStore:
    """Session-isolated in-memory history store for Hugging Face Spaces."""
    
    def __init__(self, *args, **kwargs):
        # Initialize session state structures if they don't exist
        if "agent_history_data" not in st.session_state:
            st.session_state.agent_history_data = {
                "filings": [],
                "memos": [],
                "notifications": []
            }

    def _read(self) -> dict:
        return st.session_state.agent_history_data

    def _write(self, data: dict) -> None:
        st.session_state.agent_history_data = data

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    def save_filing_snapshot(self, company: str, reporting_year: str, document_type: str,
                             filename: str, metrics: list[dict], principles_present: list[str]) -> None:
        data = self._read()
        # Remove duplicate filings for the same company/year
        data["filings"] = [
            f for f in data["filings"]
            if not (f["company"] == company and f["reporting_year"] == reporting_year)
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
        return self._read()["filings"]

    def get_prior_filing(self, company: str, exclude_year: str) -> Optional[dict]:
        filings = self._read()["filings"]
        matching = [
            f for f in filings 
            if f["company"].lower() == company.lower() and f["reporting_year"] != exclude_year
        ]
        if not matching:
            return None
        # Return the most recently saved matching filing
        return sorted(matching, key=lambda x: x["saved_at"], reverse=True)[0]

    def save_memo(self, company: str, reporting_year: str, memo_markdown: str, status: str) -> str:
        safe_company = "".join(c if c.isalnum() else "_" for c in company)
        memo_path = f"memos/{safe_company}_{reporting_year}_review.md"

        data = self._read()
        data["memos"] = [
            m for m in data["memos"]
            if not (m["company"] == company and m["reporting_year"] == reporting_year)
        ]
        data["memos"].append({
            "company": company, 
            "reporting_year": reporting_year,
            "status": status, 
            "path": memo_path, 
            "markdown_content": memo_markdown, # Store content directly in memory
            "saved_at": self._now(),
        })
        self._write(data)
        return memo_path

    def get_memo(self, company: str, reporting_year: str) -> Optional[dict]:
        data = self._read()
        for m in data["memos"]:
            if m["company"] == company and m["reporting_year"] == reporting_year:
                return m
        return None

    def add_notification(self, company: str, reporting_year: str, severity: str, summary: str) -> None:
        data = self._read()
        data["notifications"].append({
            "company": company, 
            "reporting_year": reporting_year,
            "severity": severity, 
            "summary": summary, 
            "notified_at": self._now(),
        })
        self._write(data)