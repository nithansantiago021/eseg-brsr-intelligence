"""
app.py — ESG / BRSR Report Intelligence Platform (Hugging Face Spaces & Local edition).
"""

from __future__ import annotations

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st

from config.settings import cfg, GROQ_API_KEY

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BRSR / ESG Intelligence Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e8eaf0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * { color: #c8ccd8 !important; }

.lia-header {
    display: flex; align-items: center; gap: 12px;
    padding: 1.2rem 0 0.6rem;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 1.4rem;
}
.lia-header h1 { font-size:1.5rem; font-weight:700; letter-spacing:-0.02em; color:#e8eaf0; margin:0; }
.lia-tagline { font-size:0.78rem; color:#6b7280; margin-top:2px; letter-spacing:0.03em; text-transform:uppercase; }

.chat-bubble { padding:0.85rem 1.1rem; border-radius:12px; margin-bottom:0.9rem; line-height:1.65; font-size:0.92rem; max-width:86%; }
.chat-user   { background:#1a2744; border:1px solid #233060; margin-left:auto; color:#c8d8ff; }
.chat-assistant { background:#141920; border:1px solid #1e2535; margin-right:auto; color:#dde2ec; }
.chat-unanswerable { background:#1f1418; border:1px solid #3d1f28; color:#f49090; }

.source-badge { display:inline-block; background:#1e2d4a; border:1px solid #2a3f6a; color:#7fa8e8; border-radius:6px; padding:2px 8px; font-size:0.72rem; margin:3px 3px 3px 0; font-family:monospace; }

.metric-row { display:flex; gap:10px; margin-top:6px; flex-wrap:wrap; }
.metric-pill { background:#0d1a2e; border:1px solid #1b2f4e; border-radius:20px; padding:3px 10px; font-size:0.72rem; color:#6b8ab8; }

.doc-chip { display:inline-flex; align-items:center; gap:5px; background:#131c2e; border:1px solid #1e2d45; border-radius:8px; padding:4px 10px; font-size:0.78rem; color:#8aa4cc; margin:3px 3px 3px 0; }

.warn-box    { background:#1f1a10; border:1px solid #4a3a10; border-radius:8px; padding:0.7rem 1rem; color:#d4a843; font-size:0.84rem; margin-bottom:1rem; }
.info-box    { background:#101c2e; border:1px solid #1e3a5a; border-radius:8px; padding:0.7rem 1rem; color:#6b9fcc; font-size:0.84rem; margin-bottom:1rem; }
.success-box { background:#0d1f18; border:1px solid #1a4030; border-radius:8px; padding:0.7rem 1rem; color:#5cc89a; font-size:0.84rem; margin-bottom:1rem; }
.error-box   { background:#1f0d0d; border:1px solid #4a1515; border-radius:8px; padding:0.7rem 1rem; color:#f49090; font-size:0.84rem; margin-bottom:1rem; }

.kpi-card { background:#131c2e; border:1px solid #1e2d45; border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
.kpi-value { font-size:1.5rem; font-weight:700; color:#c8d8ff; }
.kpi-label { font-size:0.78rem; color:#8a9ab8; margin-top:2px; }
.kpi-cite  { font-size:0.68rem; color:#4a5568; margin-top:6px; }

.nav-radio [role="radiogroup"] { gap: 4px; }

.step-card { background:#131c2e; border:1px solid #1e2d45; border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1rem; }
.step-number { display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:50%; background:#1e3a5a; color:#7fa8e8; font-size:0.85rem; font-weight:700; margin-right:10px; flex-shrink:0; }
.step-title { font-size:1rem; font-weight:600; color:#c8d8ff; }
.step-body  { font-size:0.84rem; color:#8a9ab8; margin-top:0.5rem; line-height:1.6; }
.feature-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:1rem; }
.feature-card { background:#0d1220; border:1px solid #1a2840; border-radius:10px; padding:1rem 1.1rem; }
.feature-icon { font-size:1.4rem; margin-bottom:6px; }
.feature-title { font-size:0.84rem; font-weight:600; color:#c8d8ff; }
.feature-desc  { font-size:0.76rem; color:#6b7a96; margin-top:3px; line-height:1.5; }
.privacy-table { width:100%; border-collapse:collapse; font-size:0.82rem; margin-top:0.5rem; }
.privacy-table th { color:#7fa8e8; text-align:left; padding:6px 10px; border-bottom:1px solid #1e2d45; font-weight:600; }
.privacy-table td { color:#8a9ab8; padding:6px 10px; border-bottom:1px solid #131c2e; }
.privacy-table tr:last-child td { border-bottom:none; }
.tick { color:#5cc89a; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Cached model loaders — process-level, shared across all sessions ──────────
@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embedder():
    from src.embedding.embedder import Embedder
    return Embedder()


@st.cache_resource(show_spinner="Loading re-ranker model…")
def _load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(cfg.retrieval.reranker_model)


def _make_ragas_guard():
    from src.evaluation.ragas_guard import RAGASGuard
    return RAGASGuard(_load_embedder())


# ── Session init ──────────────────────────────────────────────────────────────
def _init_session() -> None:
    from src.embedding.embedder import VectorStore
    from src.retrieval.retriever import Retriever
    from src.rag.rag_pipeline import RAGPipeline

    defaults = {
        "session_id":      str(uuid.uuid4())[:8],
        "chat_history":    [],
        "indexed_docs":    [],
        "chunks":          [],          # FIX 1: Persist chunks directly in session state
        "filename_filter": None,
        "scope_company":   None,
        "scope_year":      None,
        "scope_principle": None,
        "metric_registry": [],
        "active_section":  "Agent",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()

    if "retriever" not in st.session_state:
        retriever = Retriever(_load_embedder(), st.session_state.vector_store)
        retriever.set_reranker(_load_reranker())
        st.session_state.retriever = retriever

    if "pipeline" not in st.session_state:
        pipeline = RAGPipeline(st.session_state.retriever)
        pipeline.set_ragas_guard(_make_ragas_guard())
        st.session_state.pipeline = pipeline

    if "session_tmp_dir" not in st.session_state:
        st.session_state.session_tmp_dir = str(Path(tempfile.mkdtemp(prefix="esg_")))


def _clear_session() -> None:
    if "vector_store" in st.session_state:
        try:
            st.session_state.vector_store.reset()
        except Exception:
            pass
    tmp_dir = st.session_state.get("session_tmp_dir", "")
    if tmp_dir and Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


def _rebuild_metric_registry() -> None:
    from src.metrics.extractor import build_metric_registry
    registry = build_metric_registry(st.session_state.vector_store)
    st.session_state.metric_registry = registry
    st.session_state.pipeline.set_metric_registry(registry)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("##  BRSR/ESG Intelligence")
        st.markdown("---")

        st.markdown("### Navigate")
        st.session_state.active_section = st.radio(
            "Section", ["Agent", "Chatbot", "Dashboard", "How to Use"],
            index=["Agent", "Chatbot", "Dashboard", "How to Use"].index(st.session_state.active_section),
            label_visibility="collapsed",
        )
        st.markdown("---")

        if not GROQ_API_KEY:
            st.markdown(
                '<div class="warn-box">⚠ <code>GROQ_API_KEY</code> is not set in the environment. '
                'Chat and quality-guard features are disabled until it is configured '
                '(<code>.env</code> locally, Secrets on Hugging Face Spaces).</div>',
                unsafe_allow_html=True,
            )

        st.markdown("### 📄 Upload Reports")
        st.markdown(
            '<div style="font-size:0.78rem;color:#6b7280;margin-bottom:8px;">'
            'PDF, TXT, CSV, JSON, XML · Kept only in your private session folder until you clear the session.'
            '</div>', unsafe_allow_html=True,
        )
        with st.form("upload_form", clear_on_submit=True):
            up_company = st.text_input("Company", placeholder="e.g. ABC steel company")
            up_year = st.text_input("Reporting Year", placeholder="e.g. 2025-26")
            up_doctype = st.selectbox("Document Type", cfg.esg.document_types)
            up_dual_page = st.checkbox(
                "PDF has two pages merged side-by-side (landscape spread)",
                value=False,
                help="Enable this if each page image in the PDF actually shows two "
                     "logical report pages placed next to each other.",
            )
            uploaded_files = st.file_uploader(
                "Files", type=["pdf"],
                accept_multiple_files=True, label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Index Documents", use_container_width=True)

        if submitted and uploaded_files:
            if not up_company.strip():
                st.markdown('<div class="error-box">Company name is required.</div>', unsafe_allow_html=True)
            else:
                _handle_uploads(uploaded_files, up_company.strip(), up_year.strip(), up_doctype, up_dual_page)

        indexed = st.session_state.get("indexed_docs", [])
        total_chunks = len(st.session_state.get("chunks", []))
        if indexed:
            st.markdown(f"**Indexed documents ({total_chunks} total chunks):**")
            for d in indexed:
                st.markdown(
                    f'<div class="doc-chip">📎 {d["filename"]} · {d["company"]}'
                    f'{" · " + d["reporting_year"] if d["reporting_year"] else ""}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### 🔎 Chatbot Scope")
            vs = st.session_state.vector_store
            companies = ["All companies"] + vs.list_companies()
            years = ["All years"] + vs.list_years()
            principles = ["All principles"] + [f"{p} — {cfg.esg.principles[p]}" for p in vs.list_principles()]

            chosen_company = st.selectbox("Company", companies, key="scope_company_sel")
            chosen_year = st.selectbox("Reporting Year", years, key="scope_year_sel")
            chosen_principle = st.selectbox("Principle", principles, key="scope_principle_sel")

            st.session_state.scope_company = None if chosen_company == "All companies" else chosen_company
            st.session_state.scope_year = None if chosen_year == "All years" else chosen_year
            st.session_state.scope_principle = (
                None if chosen_principle == "All principles" else chosen_principle.split(" — ")[0]
            )

        st.markdown("---")
        st.markdown("### 🗑️ Session Control")
        if st.button("🗑️ Clear Session Data", use_container_width=True, type="secondary"):
            _clear_session()
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.72rem;color:#3d4455;text-align:center;">Session-isolated · Zero persistence</div>',
            unsafe_allow_html=True,
        )


def _handle_uploads(uploaded_files, company: str, reporting_year: str, doc_type: str,
                     dual_page_layout: bool = False) -> None:
    from src.embedding.indexer import index_file

    embedder = _load_embedder()
    vector_store = st.session_state.vector_store
    retriever = st.session_state.retriever  # FIX 2: Retrieve active retriever instance
    tmp_dir = Path(st.session_state.session_tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    already = {d["filename"] for d in st.session_state.indexed_docs}
    new_files = [f for f in uploaded_files if f.name not in already]
    if not new_files:
        return

    bar = st.sidebar.progress(0, text="Indexing…")
    slot = st.sidebar.empty()

    for i, uf in enumerate(new_files):
        tmp_path = tmp_dir / uf.name
        try:
            tmp_path.write_bytes(uf.read())
            slot.markdown(f'<div class="info-box">⏳ Indexing: {uf.name}…</div>', unsafe_allow_html=True)

            def _cb(msg: str, _slot=slot):
                _slot.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)

            # FIX 3: Pass retriever instance so BM25 index builds properly
            chunks = index_file(
                file_path=tmp_path,
                embedder=embedder,
                vector_store=vector_store,
                retriever=retriever,
                company=company,
                reporting_year=reporting_year,
                document_type=doc_type,
                dual_page_layout=dual_page_layout,
                progress_callback=_cb,
            )

            # FIX 4: Store chunk instances into session state
            if "chunks" not in st.session_state:
                st.session_state.chunks = []
            st.session_state.chunks.extend(chunks)

            st.session_state.indexed_docs.append({
                "filename": uf.name, "company": company,
                "reporting_year": reporting_year, "document_type": doc_type,
            })
            n_tables = sum(1 for c in chunks if c.content_type == "table")
            slot.markdown(
                f'<div class="success-box">✓ {uf.name} — {len(chunks)} chunks ({n_tables} tables)</div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            slot.markdown(f'<div class="error-box">✗ {uf.name}: {exc}</div>', unsafe_allow_html=True)
            if tmp_path.exists():
                tmp_path.unlink()
        bar.progress((i + 1) / len(new_files))
        time.sleep(0.2)

    bar.empty()
    time.sleep(0.6)
    slot.empty()
    _rebuild_metric_registry()
    st.rerun()  # FIX 5: Force Streamlit UI refresh to unlock buttons immediately


# ── Citation formatter ────────────────────────────────────────────────────────
def _format_citations(text: str) -> str:
    def _rep(m):
        return f'<span class="source-badge">[S{m.group(1)}]</span>'
    return re.sub(r"\[SOURCE\s*(\d+)\]", _rep, text, flags=re.IGNORECASE)


# ── Chatbot section ─────────────────────────────────────────────────────────
def _render_chatbot() -> None:
    no_docs = not st.session_state.get("indexed_docs")
    no_key = not GROQ_API_KEY

    if no_docs or no_key:
        messages = []
        if no_docs:
            messages.append("📄 <strong>Step 1:</strong> Upload an ESG/BRSR report in the sidebar.")
        if no_key:
            messages.append("🔑 <strong>Step 2:</strong> Configure your GROQ_API_KEY in environment or secrets.")
        
        if len(messages) == 2:
            cols = st.columns(2)
            for col, msg in zip(cols, messages):
                with col:
                    st.markdown(f'<div class="warn-box">{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warn-box">{messages[0]}</div>', unsafe_allow_html=True)

    history = st.session_state.get("chat_history", [])
    if not history:
        st.markdown(
            '<div class="info-box" style="text-align:center;padding:2rem 1rem;">'
            '🌱 Upload an ESG/BRSR report and ask about emissions, energy, water, waste, '
            'training, targets — or ask to compare companies and years.</div>',
            unsafe_allow_html=True,
        )
    else:
        for msg in history:
            role, content = msg["role"], msg["content"]
            if role == "user":
                st.markdown(f'<div class="chat-bubble chat-user">🧑‍💼 {content}</div>', unsafe_allow_html=True)
            else:
                extra = "chat-unanswerable" if content.startswith("UNANSWERABLE") else ""
                st.markdown(
                    f'<div class="chat-bubble chat-assistant {extra}">🌱 {_format_citations(content)}</div>',
                    unsafe_allow_html=True,
                )
                if msg.get("chart_fig") is not None:
                    st.plotly_chart(msg["chart_fig"], use_container_width=True, key=f"chart_{id(msg)}")
                    if msg.get("chart_citations"):
                        st.markdown(
                            '<div class="metric-row">' +
                            "".join(f'<span class="metric-pill">📎 {c}</span>' for c in msg["chart_citations"]) +
                            '</div>', unsafe_allow_html=True,
                        )
                elif msg.get("chart_refusal"):
                    st.markdown(f'<div class="warn-box">📊 {msg["chart_refusal"]}</div>', unsafe_allow_html=True)

                if "meta" in msg:
                    m = msg["meta"]
                    def _c(v):
                        return "#5cc89a" if v >= 0.7 else ("#d4a843" if v >= 0.45 else "#f49090")
                    ragas_badge = (
                        '<span class="metric-pill" style="color:#5cc89a;">✓ RAGAS passed</span>'
                        if m.get("ragas_passed") and not m.get("gate_failed") else
                        '<span class="metric-pill" style="color:#f49090;">✗ RAGAS failed</span>'
                    )
                    st.markdown(
                        f'<div class="metric-row">'
                        f'<span class="metric-pill">⏱ {m.get("latency_ms",0):.0f} ms</span>'
                        f'<span class="metric-pill" style="color:{_c(m.get("faithfulness",0))}">🎯 faith {m.get("faithfulness",0):.2f}</span>'
                        f'<span class="metric-pill" style="color:{_c(m.get("answer_relevancy",0))}">💬 ans_rel {m.get("answer_relevancy",0):.2f}</span>'
                        f'<span class="metric-pill" style="color:{_c(m.get("context_relevancy",0))}">🔍 ctx_rel {m.get("context_relevancy",0):.2f}</span>'
                        f'<span class="metric-pill">📚 {m.get("sources",0)} src</span>{ragas_badge}</div>',
                        unsafe_allow_html=True,
                    )
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} retrieved sources", expanded=False):
                        for j, src in enumerate(msg["sources"], 1):
                            tag = f"`{src['filename']}`"
                            if src.get("company"):
                                tag += f" · **{src['company']}**"
                            if src.get("reporting_year"):
                                tag += f" · {src['reporting_year']}"
                            if src.get("page"):
                                tag += f" · p.{src['page']}"
                            if src.get("content_type") == "table":
                                tag += " · 📊 table"
                            st.markdown(f"**[SOURCE {j}]** {tag}")
                            st.markdown(
                                f'<div style="background:#0d1220;border:1px solid #1a2840;border-radius:6px;'
                                f'padding:8px 12px;font-size:0.82rem;color:#9aadcc;margin-bottom:8px;">'
                                f'{src["text_preview"]}</div>', unsafe_allow_html=True,
                            )

    question = st.chat_input("Ask about emissions, water, waste, training, targets…", disabled=no_docs or no_key)
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Retrieving and generating answer…"):
            resp = st.session_state.pipeline.run(
                question=question, api_key=GROQ_API_KEY,
                filename_filter=st.session_state.get("filename_filter"),
                company=st.session_state.get("scope_company"),
                reporting_year=st.session_state.get("scope_year"),
                principle=st.session_state.get("scope_principle"),
            )
        sources_display = [
            {"filename": c.filename, "company": c.company, "reporting_year": c.reporting_year,
             "page": c.page, "content_type": c.content_type,
             "text_preview": c.text[:350] + "…" if len(c.text) > 350 else c.text}
            for c in resp.sources
        ]

        chart_fig, chart_citations, chart_refusal = None, [], resp.chart_refusal
        if resp.chart_metrics:
            from src.visualization.chart_generator import build_chart
            chart_fig, chart_citations, chart_refusal = build_chart(resp.chart_metrics)

        st.session_state.chat_history.append({
            "role": "assistant", "content": resp.answer, "sources": sources_display,
            "chart_fig": chart_fig, "chart_citations": chart_citations, "chart_refusal": chart_refusal,
            "meta": {
                "latency_ms": resp.latency_ms, "faithfulness": resp.faithfulness_score,
                "answer_relevancy": resp.answer_relevancy_score, "context_relevancy": resp.context_relevancy_score,
                "ragas_overall": resp.ragas_overall, "ragas_passed": resp.ragas_passed,
                "gate_failed": resp.gate_failed, "sources": len(resp.sources), "model": resp.model_used,
            },
        })
        st.rerun()


# ── Agent section (Compliance Sentinel) ──────────────────────────────────────
def _render_agent() -> None:
    st.markdown("#### 🛡️ Compliance Sentinel")
    st.markdown(
        '<div class="info-box">An autonomous agent that reviews a filing: indexes it, checks for missing BRSR '
        'disclosures, compares against this company\'s last reviewed filing, verifies anything ambiguous '
        'before concluding, and writes a cited memo. <strong>It only notifies a human when something '
        'actually needs attention</strong> — a clean filing is reviewed and filed silently.</div>',
        unsafe_allow_html=True,
    )

    provider = os.getenv("AGENT_MODEL_PROVIDER", "groq")
    provider_ok = provider != "groq" or bool(GROQ_API_KEY)
    provider_color = "#5cc89a" if provider_ok else "#f49090"
    st.markdown(
        f'<span class="metric-pill" style="color:{provider_color};">model provider: {provider}'
        f'{"" if provider_ok else " (GROQ_API_KEY missing)"}</span>',
        unsafe_allow_html=True,
    )

    indexed = st.session_state.get("indexed_docs", [])
    if not indexed:
        st.markdown('<div class="info-box">Upload a filing first (sidebar).</div>', unsafe_allow_html=True)
        return

    options = {f"{d['filename']} ({d['company']}, {d['reporting_year']})": d for d in indexed}
    choice = st.selectbox("Filing to review", list(options.keys()), key="sentinel_doc")
    target = options[choice]

    # In app.py inside _render_agent():
    if st.button("Run Compliance Sentinel", type="primary"):
        if not provider_ok:
            st.markdown('<div class="error-box">GROQ_API_KEY is required for AGENT_MODEL_PROVIDER=groq.</div>',
                        unsafe_allow_html=True)
            return
        with st.spinner("Sentinel reviewing filing — indexing, comparing, verifying…"):
            try:
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                import threading
                
                # Ensure the active thread holds Streamlit ScriptRunContext
                current_thread = threading.current_thread()
                add_script_run_ctx(current_thread)

                from agent.compliance_sentinel import run_compliance_check
                tmp_dir = Path(st.session_state.session_tmp_dir)
                source_path = tmp_dir / target["filename"]
                if not source_path.exists():
                    st.markdown(
                        '<div class="error-box">Original file no longer available in this session '
                        '(re-upload it to run the Sentinel).</div>', unsafe_allow_html=True,
                    )
                    return
                result = run_compliance_check(
                    str(source_path), target["company"], target["reporting_year"], target["document_type"],
                )
            except Exception as exc:
                st.markdown(f'<div class="error-box">Sentinel run failed: {exc}</div>', unsafe_allow_html=True)
                return

        memo = result.get("memo")
        findings = result.get("findings", [])
        if memo:
            status_color = "#5cc89a" if memo.get("status") == "auto_closed" else "#d4a843"
            status_label = "✓ auto_closed — no human needed" if memo.get("status") == "auto_closed" \
                else "⚠ pending_human_review"
            st.markdown(
                f'<div class="metric-pill" style="color:{status_color};">{status_label}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Review memo", expanded=True):
                memo_text = memo.get("text") or memo.get("content") or str(memo)
                st.markdown(memo_text)
        if findings:
            st.markdown("**Findings:**")
            for f in findings:
                st.markdown(f"- `[{f.get('severity', 'INFO')}]` **{f.get('topic', 'Notice')}** — {f.get('description', '')}")
        else:
            st.markdown('<div class="success-box">No findings — nothing flagged this review.</div>',
                        unsafe_allow_html=True)

# ── Dashboard section ─────────────────────────────────────────────────────────
def _render_dashboard() -> None:
    st.markdown("#### 📊 ESG Metric Analytics & Visualizations")
    
    registry = st.session_state.get("metric_registry", [])
    if not registry:
        st.markdown(
            '<div class="info-box">No structured metrics available. Upload an ESG/BRSR report '
            'in the sidebar to extract quantitative disclosures and view charts.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="success-box">✓ Currently tracking <strong>{len(registry)}</strong> '
        f'extracted ESG metrics across indexed filings.</div>',
        unsafe_allow_html=True,
    )

    # 1. High-Level KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    vs = st.session_state.vector_store
    companies = vs.list_companies()
    years = vs.list_years()
    principles = vs.list_principles()

    with col1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{len(companies)}</div>'
            f'<div class="kpi-label">Companies Tracked</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{len(years)}</div>'
            f'<div class="kpi-label">Reporting Years</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{len(principles)} / 9</div>'
            f'<div class="kpi-label">BRSR Principles Covered</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{len(registry)}</div>'
            f'<div class="kpi-label">Total Quantitative Metrics</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. Interactive Plotly Visualizations via chart_generator / dashboard
    st.markdown("### 📈 Metric Comparison Charts")
    
    # Filter metrics by selected company scope if set
    filtered_registry = registry
    if st.session_state.get("scope_company"):
        filtered_registry = [
            m for m in filtered_registry 
            if m.company.lower() == st.session_state.scope_company.lower()
        ]

    # Try rendering via src/visualization/dashboard.py or chart_generator.py
    try:
        from src.visualization.chart_generator import build_chart
        
        # Group metrics by canonical metric name
        metric_groups: dict[str, list] = {}
        for m in filtered_registry:
            metric_groups.setdefault(m.normalized_metric_name, []).append(m)

        # Filter metric names that have at least 2 comparable values
        comparable_names = [
            name for name, metrics in metric_groups.items() if len(metrics) >= 2
        ]

        if comparable_names:
            selected_metric_name = st.selectbox(
                "Select ESG Metric to Compare", comparable_names
            )
            selected_metrics = metric_groups[selected_metric_name]
            
            fig, citations, refusal = build_chart(selected_metrics)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                if citations:
                    st.markdown(
                        '<div class="metric-row">' +
                        "".join(f'<span class="metric-pill">📎 {c}</span>' for c in citations) +
                        '</div>', unsafe_allow_html=True,
                    )
            elif refusal:
                st.markdown(f'<div class="warn-box">📊 {refusal}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="info-box">Upload multiple filings or multi-year reports to unlock '
                'automated year-over-year comparison charts.</div>',
                unsafe_allow_html=True,
            )
    except ImportError:
        st.markdown(
            '<div class="warn-box">Chart generation module not found. Displaying metric data table below.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 3. Extracted Metrics Data Table
    st.markdown("### 📋 Extracted ESG Metrics Registry")
    table_data = [
        {
            "Company": m.company,
            "Period": m.reporting_year,
            "Principle": m.principle or "General",
            "Metric Name": m.metric_name,
            "Canonical Name": m.normalized_metric_name,
            "Value": m.value,
            "Raw Value": m.raw_value,
            "Unit": m.unit,
            "Page": m.page,
        }
        for m in filtered_registry
    ]
    st.dataframe(table_data, use_container_width=True)


# ── How to Use section ───────────────────────────────────────────────────────
def _render_how_to_use() -> None:
    st.markdown("#### 🚀 Quick Start")
    steps = [
        ("Upload a report", "In the sidebar, enter the <strong>Company</strong>, <strong>Reporting Year</strong>, and <strong>Document Type</strong>, then upload the PDF file. Tables are extracted and chunked separately from prose so structure is preserved."),
        ("Run the Compliance Sentinel", "The <strong>Agent</strong> section runs an autonomous agent that reviews the filing end to end — checks for missing disclosures, compares against this company's last reviewed filing, verifies anything ambiguous, and writes a cited memo."),
        ("Ask questions", "Use the <strong>Chatbot</strong> section for citation-backed Q&A. Narrow scope by company/year/principle in the sidebar. Comparison-style questions (\"compare X vs Y\") auto-generate a chart from extracted metrics when enough data points exist."),
        ("Clear your session", "Click <strong>🗑️ Clear Session Data</strong> any time to wipe all embeddings, extracted metrics, and chat history from memory."),
    ]
    for i, (title, body) in enumerate(steps, 1):
        st.markdown(
            f'<div class="step-card"><div style="display:flex;align-items:flex-start;gap:10px;">'
            f'<span class="step-number">{i}</span><div><div class="step-title">{title}</div>'
            f'<div class="step-body">{body}</div></div></div></div>', unsafe_allow_html=True,
        )

    st.markdown("#### ✨ Features")
    st.markdown(
        '<div class="feature-grid">'
        '<div class="feature-card"><div class="feature-icon">📊</div><div class="feature-title">Table-Aware Extraction</div>'
        '<div class="feature-desc">Tables are parsed structurally, chunked separately from prose, and never split mid-row.</div></div>'
        '<div class="feature-card"><div class="feature-icon">📎</div><div class="feature-title">Citation-Grounded Answers</div>'
        '<div class="feature-desc">Every claim cites [SOURCE N] with company, year, page, and principle.</div></div>'
        '<div class="feature-card"><div class="feature-icon">🏷️</div><div class="feature-title">Metadata Filtering</div>'
        '<div class="feature-desc">Scope retrieval and dashboards by company, reporting year, or BRSR principle.</div></div>'
        '<div class="feature-card"><div class="feature-icon">📈</div><div class="feature-title">Auto Chart Generation</div>'
        '<div class="feature-desc">Comparable questions generate charts directly from extracted table data — never invented.</div></div>'
        '<div class="feature-card"><div class="feature-icon">🧩</div><div class="feature-title">Metric Normalization</div>'
        '<div class="feature-desc">Synonymous metric names are aligned; incompatible units are never silently combined.</div></div>'
        '<div class="feature-card"><div class="feature-icon">🧪</div><div class="feature-title">RAGAS Quality Guard</div>'
        '<div class="feature-desc">Faithfulness, answer relevancy, and context relevancy gate every chat answer.</div></div>'
        '<div class="feature-card"><div class="feature-icon">🛡️</div><div class="feature-title">Compliance Sentinel Agent</div>'
        '<div class="feature-desc">An autonomous agent reviews filings, compares year-over-year, and interrupts a human only when something needs attention.</div></div>'
        '</div>', unsafe_allow_html=True,
    )

    st.markdown("#### 🔒 Privacy Guarantees")
    st.markdown(
        '<table class="privacy-table"><thead><tr><th>Data</th><th>Storage</th><th>Deleted when</th></tr></thead><tbody>'
        '<tr><td>Uploaded documents</td><td>Your private session temp folder</td><td><span class="tick">✓</span> Session ends or Clear button</td></tr>'
        '<tr><td>Embeddings / vector store</td><td>In-memory only (ChromaDB EphemeralClient)</td><td><span class="tick">✓</span> Session ends or Clear button</td></tr>'
        '<tr><td>Extracted ESG metrics</td><td>Session RAM only</td><td><span class="tick">✓</span> Session ends or Clear button</td></tr>'
        '<tr><td>Chat history</td><td>Browser session state</td><td><span class="tick">✓</span> Session ends or Clear button</td></tr>'
        '<tr><td>Groq API key</td><td>Server-side environment variable</td><td>Not session data — configured by the deployer, never logged</td></tr>'
        '<tr><td>Cross-user data</td><td>Never shared</td><td><span class="tick">✓</span> Fully isolated per session</td></tr>'
        '</tbody></table>', unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _init_session()
    _render_sidebar()

    st.markdown(
        '<div class="lia-header"><span style="font-size:2rem;">🌱</span><div>'
        '<h1>ESG Intelligence Platform</h1>'
        '<div class="lia-tagline">BRSR · ESG · Sustainability Reports — Citation-Grounded Analysis</div>'
        '</div></div>', unsafe_allow_html=True,
    )

    section = st.session_state.active_section
    if section == "Agent":
        _render_agent()
    elif section == "Chatbot":
        _render_chatbot()
    elif section == "Dashboard":
        _render_dashboard()
    else:
        _render_how_to_use()


if __name__ == "__main__":
    main()
