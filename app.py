"""
app.py
------
Career Sensei — Streamlit entry point.

This file only handles:
  - Page configuration
  - Session state initialisation
  - Sidebar inputs
  - Orchestrating backend (src/rag/) and UI (src/ui/) modules
"""

import streamlit as st
import os
import traceback
import shutil

# ── Streamlit Cloud ChromaDB SQLite Patch ─────────────────────────────────────
# ChromaDB requires SQLite > 3.35.0. Streamlit Cloud's default environment
# often has an older version. We patch it here before any Chroma imports.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from dotenv import load_dotenv

from src.rag.utils import extract_resume_text, fetch_job_description
from src.rag.pipeline import build_chains, run_gap_analysis, seed_chat_memory, LLM_MODEL
from src.ui.styles import inject_styles
from src.ui.components import (
    landing_page,
    waiting_card,
    results_layout,
    learning_plan_tab,
    chatbot_section,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

# Streamlit Community Cloud often stores secrets in `st.secrets` rather than
# exporting them into `os.environ`. The Gemini/LangChain libs rely on the
# `GOOGLE_API_KEY` env var, so we mirror the value into `os.environ` if needed.
if not os.environ.get("GOOGLE_API_KEY"):
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        # If secrets aren't configured, we'll just fall back to env/default creds.
        pass

st.set_page_config(
    page_title="Career Sensei",
    page_icon="🥷",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

# ── Session state defaults ────────────────────────────────────────────────────
_defaults = {
    "analysis":     None,
    "gap_chain":    None,
    "chat_chain":   None,
    "chat_history": [],
    "chroma_persist_dir": None,
    "learning_plan_markdown": None,
    "learning_plan_weeks": None,
    "learning_plan_hours": None,
    "learning_plan_docx_bytes": None,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Use st.html to avoid Streamlit's built-in anchor-link hover on h-tags
    st.html("""
    <div style="text-align:center; padding: 8px 0 4px;">
      <div style="font-size:28px; margin-bottom:4px;">🥷</div>
      <div style="font-size:17px; font-weight:700; color:#1a1d2e;">Career Sensei</div>
      <div style="font-size:12px; color:#9ca3af; margin-top:3px;">AI-powered gap analysis</div>
      <div style="font-size:11px; color:#b0b5c0; margin-top:6px; line-height:1.5;">
        Upload your resume and either a job URL or pasted job description to see how well you fit and where to improve.
      </div>
    </div>
    """)
    st.divider()

    uploaded = st.file_uploader("Resume", type=["docx", "pdf"],
                                 help="Upload your resume (.docx or .pdf)")

    st.caption("Job input")
    input_mode = st.radio(
        "Choose how to provide the job posting",
        options=["URL", "Paste description"],
        index=0,
        horizontal=False,
    )

    job_url = ""
    pasted_job_desc = ""
    if input_mode == "URL":
        job_url = st.text_input(
            "Job Posting URL",
            placeholder="https://jobs.company.com/...",
            help="We will try to fetch and extract the job description from this page.",
        )
    else:
        pasted_job_desc = st.text_area(
            "Paste job description",
            placeholder="Paste the full job description text here (include responsibilities/requirements).",
            height=200,
            help="We will use your pasted text directly (no scraping).",
        )

    analyze  = st.button("Analyze", use_container_width=True)

    st.divider()
    st.caption("Powered by Google Gemini · LangChain · ChromaDB")
    st.html(f"""
    <p style="font-size:10px; color:#c0c5d0; margin-top:4px; text-align:center;">
      Model: {LLM_MODEL}
    </p>
    """)
    st.html("""
    <p style="font-size:11px; color:#b0b5c0; line-height:1.6; margin-top:8px; text-align:center;">
      🔒 Your resume and data stay on your device and are never stored.
      Everything is cleared the moment you close this tab.
    </p>
    """)

# ── Main ──────────────────────────────────────────────────────────────────────
if not uploaded:
    landing_page()

else:
    if analyze:
        if input_mode == "URL":
            if not job_url.strip():
                st.error("Please enter a job URL (or switch to 'Paste description').")
                st.stop()

        if input_mode == "Paste description":
            if not pasted_job_desc.strip():
                st.error("Please paste a job description (or switch to 'URL').")
                st.stop()

        with st.spinner("Reading resume..."):
            resume_text = extract_resume_text(uploaded)

        if not resume_text.strip():
            st.error("Could not extract text. Please try a DOCX or text-based PDF.")
        else:
            if input_mode == "URL":
                with st.spinner("Fetching job description..."):
                    job_desc = fetch_job_description(job_url)

                if not job_desc:
                    st.error("Could not load that URL. Try a different job board or switch to 'Paste description'.")
                    st.stop()
            else:
                with st.spinner("Using pasted job description..."):
                    job_desc = pasted_job_desc.strip()

            _ok = False
            try:
                with st.spinner("Running analysis — this takes about 30 seconds..."):
                    # Cleanup any previous Chroma temp directory so Chroma doesn't
                    # reuse corrupted on-disk state on Streamlit Community Cloud.
                    prev_dir = st.session_state.get("chroma_persist_dir")
                    if prev_dir:
                        try:
                            shutil.rmtree(prev_dir, ignore_errors=True)
                        except Exception:
                            # Cleanup should never block analysis.
                            pass
                    st.session_state.chroma_persist_dir = None

                    gap_chain, chat_chain, memory, vectorstore = build_chains(resume_text, job_desc)
                    analysis = run_gap_analysis(gap_chain, job_desc)
                    seed_chat_memory(memory, analysis, vectorstore, job_desc)

                st.session_state.analysis = analysis
                st.session_state.gap_chain = gap_chain
                st.session_state.chat_chain = chat_chain
                st.session_state.chat_history = []
                # Reset learning plan so it matches the newly uploaded resume/job.
                st.session_state.learning_plan_markdown = None
                st.session_state.learning_plan_weeks = None
                st.session_state.learning_plan_hours = None
                st.session_state.learning_plan_docx_bytes = None
                # Keep track of the Chroma persist directory so we can clean it up
                # on the next analysis run (Streamlit Community Cloud).
                st.session_state.chroma_persist_dir = getattr(vectorstore, "_persist_directory", None)
                _ok = True
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "resourceexhausted" in err.replace(" ", ""):
                    st.error(
                        "⚠️ We've hit our usage limit for the moment. "
                        "Please wait a minute and try again — this usually resolves quickly."
                    )
                else:
                    st.error(
                        "Something went wrong while running the analysis. "
                        "Please try again in a moment."
                    )
                    with st.expander("Debug details (copy/paste this)"):
                        st.write(f"{type(e).__name__}: {e}")
                        # Include traceback so we can pinpoint the exact failing step
                        # (embeddings, LLM call, vectorstore init, etc.).
                        st.code(traceback.format_exc())

            # If parsing failed to produce core sections, show the raw model output
            # so the UI never appears "broken" during demos.
            if _ok:
                missing_core = (
                    not str(st.session_state.analysis.strengths or "").strip()
                    or not str(st.session_state.analysis.gaps or "").strip()
                    or not str(st.session_state.analysis.suggestions or "").strip()
                )
                parse_fallback_markers = (
                    "could not parse strengths" in str(st.session_state.analysis.strengths or "").lower()
                    or "could not parse gaps" in str(st.session_state.analysis.gaps or "").lower()
                    or "could not parse suggestions" in str(st.session_state.analysis.suggestions or "").lower()
                )
                if missing_core or parse_fallback_markers:
                    st.warning(
                        "We generated a response, but section parsing was partial. "
                        "Please click Analyze once more; if this persists, use the raw output in Debug details."
                    )

            if _ok:
                st.rerun()

    if st.session_state.analysis:
        tab_analysis, tab_learning = st.tabs(["Analysis", "Learning Plan"])
        with tab_analysis:
            results_layout(st.session_state.analysis)
        with tab_learning:
            learning_plan_tab(st.session_state.chat_chain, st.session_state.analysis)

        chatbot_section(st.session_state.chat_chain)
    elif not analyze:
        waiting_card()
