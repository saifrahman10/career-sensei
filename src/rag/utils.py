"""
src/rag/utils.py
----------------
File ingestion and URL scraping utilities.
"""

import io
import re
import requests
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup


def extract_resume_text(uploaded_file) -> str:
    """Extract plain text from an uploaded DOCX or PDF file object."""
    name = uploaded_file.name.lower()
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(p.text for p in doc.paragraphs)
    elif name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return ""


def fetch_job_description(url: str, char_limit: int = 8000) -> str:
    """
    Fetch a job posting page and return cleaned plain text.

    Returns an empty string if the page cannot be retrieved.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:char_limit]
