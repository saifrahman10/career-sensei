"""
src/ui/styles.py
----------------
CSS injection for the Career Sensei Streamlit app.
Call inject_styles() once at app startup.
"""

import streamlit as st

_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  *:not([class*="material"]):not([class*="icon"]):not([data-testid="stIconMaterial"]) {
      font-family: 'Inter', sans-serif !important;
  }
  /* Ensure Streamlit's icon fonts always load */
  .material-symbols-rounded,
  .material-symbols-outlined,
  .material-icons,
  [class*="MaterialIcon"],
  span[data-testid="stIconMaterial"] {
      font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
  }

  /* Base */
  .stApp { background: #f7f8fc; }
  section[data-testid="stSidebar"] {
      background: #ffffff;
      border-right: 1px solid #e8eaf0;
  }
  #MainMenu, footer { visibility: hidden; }

  /* Global text */
  h1, h2, h3, h4 { color: #1a1d2e !important; }
  p, label, .stMarkdown { color: #5a6070 !important; }

  /* Ensure button text stays white (Streamlit wraps labels in <p>) */
  .stButton > button p,
  .stButton > button span,
  .stFormSubmitButton > button p,
  .stFormSubmitButton > button span,
  button[kind="primary"] p,
  button[kind="primary"] span { color: #ffffff !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] label { color: #374151 !important; font-weight: 500; }
  section[data-testid="stSidebar"] p { color: #6b7280 !important; }

  /* Cards */
  .cs-card {
      background: #ffffff;
      border: 1px solid #e8eaf0;
      border-radius: 12px;
      padding: 22px 24px;
      height: 100%;
      box-sizing: border-box;
  }
  .cs-card-label {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 6px;
  }
  .cs-card ul { margin: 0; padding-left: 20px; }
  .cs-card li { color: #374151 !important; font-size: 14px; line-height: 1.75; }
  .cs-card p.body { color: #374151 !important; font-size: 14px; line-height: 1.75; margin: 0; }

  /* Score circle */
  .score-wrap { text-align: center; }
  .score-circle {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 110px; height: 110px;
      border-radius: 50%;
      font-size: 36px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 6px;
  }
  .score-high { background: linear-gradient(135deg,#10b981,#059669); box-shadow: 0 4px 20px rgba(16,185,129,.25); }
  .score-mid  { background: linear-gradient(135deg,#f59e0b,#d97706); box-shadow: 0 4px 20px rgba(245,158,11,.25); }
  .score-low  { background: linear-gradient(135deg,#ef4444,#dc2626); box-shadow: 0 4px 20px rgba(239,68,68,.25); }
  .score-label { font-size: 12px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; }
  .score-high-txt { color: #059669; }
  .score-mid-txt  { color: #d97706; }
  .score-low-txt  { color: #dc2626; }

  /* Section label */
  .section-header {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: #9ca3af !important;
      margin: 28px 0 14px;
  }

  /* Inputs */
  .stTextInput input, .stTextArea textarea {
      background: #f9fafb !important;
      color: #1a1d2e !important;
      border: 1px solid #d1d5db !important;
      border-radius: 8px !important;
      font-size: 14px !important;
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
      border-color: #6366f1 !important;
      box-shadow: 0 0 0 3px rgba(99,102,241,.12) !important;
  }

  /* ── Buttons (all variants) ── */
  .stButton > button,
  .stFormSubmitButton > button,
  .stDownloadButton > button,
  button[kind="primary"],
  button[kind="secondary"] {
      background-color: #111827 !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: 8px !important;
      font-weight: 600 !important;
      font-size: 14px !important;
      padding: 10px 20px !important;
      width: 100%;
      transition: opacity 0.2s;
  }
  .stButton > button:hover,
  .stFormSubmitButton > button:hover,
  .stDownloadButton > button:hover,
  button[kind="primary"]:hover,
  button[kind="secondary"]:hover {
      background-color: #1f2937 !important;
      color: #ffffff !important;
      opacity: 0.88;
  }
  .stButton > button:active,
  button[kind="primary"]:active {
      background-color: #374151 !important;
      color: #ffffff !important;
  }

  /* ── Kill Streamlit's heading anchor / "keyboard double" icon ── */
  .stHeadingWithActionElements [data-testid="stHeaderActionElements"] { display: none !important; }
  h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
  [data-testid="stHeaderActionElements"] { display: none !important; visibility: hidden !important; }

  /* Chat bubbles */
  .bubble-user {
      background: #111827;
      color: #fff !important;
      border-radius: 16px 16px 4px 16px;
      padding: 10px 16px;
      margin: 8px 0;
      max-width: 72%;
      margin-left: auto;
      font-size: 14px;
      line-height: 1.6;
  }
  .bubble-ai {
      background: #ffffff;
      border: 1px solid #e8eaf0;
      color: #374151 !important;
      border-radius: 16px 16px 16px 4px;
      padding: 10px 16px;
      margin: 8px 0;
      max-width: 78%;
      font-size: 14px;
      line-height: 1.6;
  }

  hr { border-color: #e8eaf0 !important; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .fade-up { animation: fadeUp 0.4s ease forwards; }
</style>
"""


def inject_styles() -> None:
    """Inject all custom CSS into the Streamlit app."""
    st.html(_CSS)
