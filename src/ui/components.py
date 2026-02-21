"""
src/ui/components.py
---------------------
Reusable Streamlit UI components for Career Sensei.
"""

import streamlit as st
from src.rag.pipeline import AnalysisResult


# ── Helpers ───────────────────────────────────────────────────────────────────
def _bullets_html(text: str) -> str:
    """Convert newline-separated bullet text to an HTML <ul> list."""
    lines = [l.lstrip("-*• ").strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return "<p class='body'>—</p>"
    return "<ul>" + "".join(f"<li>{l}</li>" for l in lines) + "</ul>"


def _score_classes(score: int):
    """Return (circle_css_class, label_css_class, fit_label) for a given score."""
    if score >= 70:
        return "score-high", "score-high-txt", "Strong fit"
    if score >= 45:
        return "score-mid",  "score-mid-txt",  "Partial fit"
    return "score-low", "score-low-txt", "Stretch role"


# ── Components ────────────────────────────────────────────────────────────────
def section_header(title: str) -> None:
    st.markdown(
        f"<div style='font-size:11px;font-weight:700;letter-spacing:1.5px;"
        f"text-transform:uppercase;color:#9ca3af;margin:28px 0 14px;'>{title}</div>",
        unsafe_allow_html=True,
    )


def info_card(icon: str, label: str, color: str, body_html: str) -> None:
    """Render a bordered card with an icon label and arbitrary HTML body."""
    st.markdown(f"""
    <div style="background:#fff;border:1px solid #e8eaf0;border-radius:12px;padding:22px 24px;height:100%;box-sizing:border-box;">
      <div style="font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:6px;color:{color};">{icon} {label}</div>
      {body_html}
    </div>""", unsafe_allow_html=True)


def score_card(score: int) -> None:
    """Render the circular match score badge card with inline gradient styles."""
    if score >= 70:
        grad = "linear-gradient(135deg,#10b981,#059669)"
        shadow = "0 4px 20px rgba(16,185,129,.25)"
        label_color = "#059669"
        fit_label = "Strong fit"
    elif score >= 45:
        grad = "linear-gradient(135deg,#f59e0b,#d97706)"
        shadow = "0 4px 20px rgba(245,158,11,.25)"
        label_color = "#d97706"
        fit_label = "Partial fit"
    else:
        grad = "linear-gradient(135deg,#ef4444,#dc2626)"
        shadow = "0 4px 20px rgba(239,68,68,.25)"
        label_color = "#dc2626"
        fit_label = "Stretch role"

    st.markdown(f"""
    <div style="background:#fff;border:1px solid #e8eaf0;border-radius:12px;text-align:center;padding:28px 16px;">
      <div style="text-align:center;">
        <div style="display:inline-flex;align-items:center;justify-content:center;width:110px;height:110px;border-radius:50%;font-size:36px;font-weight:700;color:#fff;margin-bottom:6px;background:{grad};box-shadow:{shadow};">{score}</div>
        <div style="font-size:12px;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;color:{label_color};">{fit_label}</div>
      </div>
    </div>""", unsafe_allow_html=True)


def results_layout(analysis: AnalysisResult) -> None:
    """
    Render the full structured analysis output:
      Row 1 — Score  |  Job Summary
      Row 2 — Strengths  |  Gaps
      Row 3 — Experience Suggestions (full-width)
    """
    score = analysis.score or 0

    # Row 1: Score + Job Summary
    section_header("Overview")
    col_score, col_summary = st.columns([1, 3], gap="medium")
    with col_score:
        score_card(score)
    with col_summary:
        info_card("📋", "What This Role Is About", "#6366f1",
                  f"<p class='body'>{analysis.job_summary}</p>")

    # Row 2: Strengths + Gaps
    section_header("Fit Analysis")
    col_str, col_gap = st.columns(2, gap="medium")
    with col_str:
        info_card("✅", "Key Strengths", "#059669", _bullets_html(analysis.strengths))
    with col_gap:
        info_card("🔍", "Key Gaps", "#dc2626", _bullets_html(analysis.gaps))

    # Row 3: Experience Suggestions
    section_header("Action Plan")
    info_card("💡", "How to Get the Missing Experience", "#d97706",
              _bullets_html(analysis.suggestions))


def chatbot_section(chat_chain) -> None:
    """
    Render the follow-up chatbot section.
    Uses st.chat_input() so Enter key submits naturally.
    Reads/writes st.session_state.chat_history.
    """
    st.divider()
    st.html("<h3 style='color:#1a1d2e;font-size:18px;font-weight:600;margin-bottom:0;'>💬 Ask a Follow-up Question</h3>")
    st.html(
        "<p style='font-size:13px;color:#9ca3af;margin-top:4px;'>"
        "Ask anything about this role, your resume, or next steps.</p>"
    )

    # Render history oldest-first
    for role, msg in st.session_state.chat_history:
        cls = "bubble-user" if role == "user" else "bubble-ai"
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    # st.chat_input natively submits on Enter and returns the text
    chat_input = st.chat_input("e.g. What skill should I prioritize learning first?")
    if chat_input:
        try:
            with st.spinner("Thinking..."):
                from src.rag.pipeline import run_chat
                answer = run_chat(chat_chain, chat_input)
            st.session_state.chat_history.append(("user", chat_input))
            st.session_state.chat_history.append(("ai", answer))
            st.rerun()
        except Exception as e:
            err = str(e).lower()
            st.session_state.chat_history.append(("user", chat_input))
            if "429" in err or "quota" in err or "resourceexhausted" in err.replace(" ", ""):
                st.session_state.chat_history.append(("ai", "We've hit our usage limit. Please wait a minute and try again."))
            else:
                st.session_state.chat_history.append(("ai", "Something went wrong. Please try again in a moment."))
            st.rerun()


def landing_page() -> None:
    st.html("""
    <div style="text-align:center; padding:90px 20px 0;">
      <div style="font-size:52px;margin-bottom:18px;">🥷</div>
      <h1 style="font-size:28px;font-weight:700;color:#1a1d2e;margin-bottom:10px;">Career Sensei</h1>
      <p style="color:#6b7280;font-size:15px;max-width:400px;margin:0 auto;line-height:1.8;">
        Upload your resume and paste a job URL to get an instant gap analysis,
        match score, and personalised action plan.
      </p>
    </div>""")


def waiting_card() -> None:
    st.markdown("""
    <div style="background:#fff;border:1px solid #e8eaf0;border-radius:12px;text-align:center;padding:48px;margin-top:20px;">
      <div style="font-size:32px;margin-bottom:12px;">📄</div>
      <p style="font-size:14px;color:#6b7280;">
        Resume uploaded. Paste a job URL in the sidebar and click <strong>Analyze</strong>.
      </p>
    </div>""", unsafe_allow_html=True)
