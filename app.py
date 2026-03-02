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

# ── Streamlit Cloud ChromaDB SQLite Patch ─────────────────────────────────────
# ChromaDB requires SQLite > 3.35.0. Streamlit Cloud's default environment
# often has an older version. We patch it here before any Chroma imports.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv

from src.rag.utils import extract_resume_text, fetch_job_description
from src.rag.pipeline import build_chains, run_gap_analysis, seed_chat_memory, LLM_MODEL
from src.ui.styles import inject_styles
from src.ui.components import (
    landing_page,
    waiting_card,
    results_layout,
    chatbot_section,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

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
        Upload your resume and a job URL to see how well you fit and where to improve.
      </div>
    </div>
    """)
    st.divider()

    uploaded = st.file_uploader("Resume", type=["docx", "pdf"],
                                 help="Upload your resume (.docx or .pdf)")
    job_url  = st.text_input("Job Posting URL",
                              placeholder="https://jobs.company.com/...")
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
    if analyze and job_url:
        with st.spinner("Reading resume..."):
            resume_text = extract_resume_text(uploaded)

        if not resume_text.strip():
            st.error("Could not extract text. Please try a DOCX or text-based PDF.")
        else:
            with st.spinner("Fetching job description..."):
                job_desc = fetch_job_description(job_url)

            if not job_desc:
                st.error("Could not load that URL. Try a different job board or link.")
            else:
                _ok = False
                try:
                    with st.spinner("Running analysis — this takes about 30 seconds..."):
                        gap_chain, chat_chain, memory, vectorstore = build_chains(resume_text, job_desc)
                        analysis = run_gap_analysis(gap_chain, job_desc)
                        seed_chat_memory(memory, analysis, vectorstore, job_desc)

                    st.session_state.analysis     = analysis
                    st.session_state.gap_chain    = gap_chain
                    st.session_state.chat_chain   = chat_chain
                    st.session_state.chat_history = []
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

                if _ok:
                    st.rerun()

    if st.session_state.analysis:
        results_layout(st.session_state.analysis)
        chatbot_section(st.session_state.chat_chain)
    elif not analyze:
        waiting_card()
