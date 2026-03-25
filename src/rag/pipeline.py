"""
src/rag/pipeline.py
-------------------
LangChain RAG pipeline: vector store construction, chain building,
LLM prompts, and output parsing.
"""

import re
import os
from dataclasses import dataclass
from typing import Optional
import tempfile
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage


# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "models/gemini-embedding-001"
# Use a stable default model; preview or older variants can be revoked for new users.
# Allow override via env var so model changes do not require code edits.
# Gemini 3.1 Flash Lite is available under a preview model id in many accounts.
# Allow override via env var so model changes do not require code edits.
LLM_MODEL        = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
CHUNK_SIZE       = 600
CHUNK_OVERLAP    = 40
TOP_K            = 8
MAX_OUTPUT_TOKENS = 900

GAP_PROMPT_TEMPLATE = """You are a professional career advisor. Compare the candidate's resume against the job description.

Resume sections:
{context}

Job Description:
{question}

Output EXACTLY these five sections with these exact headers.
Be specific to the actual candidate and role, and make the bullets genuinely actionable (not generic).

## MATCH SCORE
[integer 0-100 only, nothing else]

## JOB SUMMARY
2-3 sentences: summarize what the role is really about, the core outcomes, and the ideal candidate profile.

## KEY STRENGTHS
3-4 bullets, ordered by impact.
Each bullet must include:
1) resume evidence (a specific skill/experience),
2) how it maps to the job requirement,
3) why it matters for success in this role.

## KEY GAPS
3-5 bullets, ordered by impact.
Each bullet must include:
1) the missing/weak capability (be concrete),
2) what the job expects instead,
3) the likely risk/impact if you don't address it.

## EXPERIENCE SUGGESTIONS
3-4 bullets, ordered to help you land interviews first.
Each bullet must include:
1) a specific action (project, certification, course, or practice),
2) a concrete deliverable you can add to your resume/portfolio,
3) a suggested timeframe (roughly: 1-2 weeks / 3-4 weeks / 6-8 weeks).

Each bullet: ONE line only (keep it compact but specific).
No intro text before the first header."""


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class AnalysisResult:
    score: Optional[int]
    job_summary: str
    strengths: str
    gaps: str
    suggestions: str


# ── Chain builder ─────────────────────────────────────────────────────────────
def build_chains(resume_text: str, job_description: str):
    """
    Embed the resume into an in-memory vector store and return a
    (gap_chain, chat_chain, memory, vectorstore) tuple.

    Only the resume is embedded initially so the gap-analysis retriever
    always surfaces resume content.  Call seed_chat_memory() afterwards
    to inject the job description into the store for the chatbot.

    Args:
        resume_text:     Plain text of the candidate's resume.
        job_description: Plain text of the job posting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # Only embed the resume so the retriever always returns resume content.
    # The job description is passed directly via the prompt's {question} variable,
    # so it does NOT need to be in the vector store.
    resume_chunks = splitter.split_text(resume_text)
    resume_docs   = [f"[RESUME] {c}" for c in resume_chunks]

    # Streamlit Community Cloud can have a corrupted/shared Chroma on-disk
    # state across runs. Use a fresh, unique persistent directory + collection
    # name per analysis to force Chroma to initialize its DB schema correctly.
    persist_dir = tempfile.mkdtemp(prefix="career_sensei_chroma_")
    collection_name = f"resume_session_{uuid.uuid4().hex}"

    # Embed + in-memory store
    embeddings  = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_texts(
        texts=resume_docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    # Gap analysis chain (built once; chat_chain is used for follow-ups)
    gap_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=GAP_PROMPT_TEMPLATE,
    )
    gap_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type="stuff",
        chain_type_kwargs={"prompt": gap_prompt},
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )

    return gap_chain, chat_chain, memory, vectorstore


def seed_chat_memory(
    memory: ConversationBufferMemory,
    analysis: AnalysisResult,
    vectorstore=None,
    job_description: str = "",
) -> None:
    """
    Prime the chatbot memory with the completed gap analysis so that follow-up
    questions have full context about the job and candidate fit.

    Also adds the analysis *and* job description as searchable text in the
    vector store so the retriever can find them when answering follow-ups.
    """
    # Build a rich text summary of the analysis
    analysis_text = (
        f"[GAP ANALYSIS RESULTS]\n"
        f"Match Score: {analysis.score}/100\n\n"
        f"Job Summary: {analysis.job_summary}\n\n"
        f"Key Strengths:\n{analysis.strengths}\n\n"
        f"Key Gaps:\n{analysis.gaps}\n\n"
        f"Experience Suggestions:\n{analysis.suggestions}"
    )

    # Add analysis + job description to vector store for chatbot follow-ups
    if vectorstore is not None:
        extra_texts = [analysis_text]
        if job_description:
            extra_texts.append(f"[JOB DESCRIPTION] {job_description}")
        vectorstore.add_texts(extra_texts)

    seed_human = (
        "I just completed a gap analysis comparing my resume to this job. "
        f"The role is: {analysis.job_summary} "
        "Please keep this context in mind for all follow-up questions."
    )
    seed_ai = (
        f"Understood! Here's a quick summary of what I found:\n\n"
        f"**Match Score:** {analysis.score}/100\n\n"
        f"**Key Strengths:**\n{analysis.strengths}\n\n"
        f"**Key Gaps:**\n{analysis.gaps}\n\n"
        f"**Suggestions:**\n{analysis.suggestions}\n\n"
        "Feel free to ask me anything about this role, your resume, or how to improve your application."
    )
    memory.chat_memory.add_message(HumanMessage(content=seed_human))
    memory.chat_memory.add_message(AIMessage(content=seed_ai))


# ── Convenience wrappers ──────────────────────────────────────────────────────
def run_gap_analysis(gap_chain, job_description: str) -> AnalysisResult:
    """Run the gap analysis chain and return a parsed AnalysisResult."""
    raw = gap_chain.invoke({"query": job_description})["result"]
    return parse_analysis(raw)


def run_chat(chat_chain, question: str) -> str:
    """Send a follow-up question to the conversational chain and return the answer."""
    response = chat_chain.invoke({"question": question})
    return response.get("answer", "")


# ── Output parser ─────────────────────────────────────────────────────────────
def parse_analysis(raw_output: str) -> AnalysisResult:
    """
    Parse the structured LLM response into an AnalysisResult dataclass.
    Falls back gracefully if a section is missing.
    """
    cleaned = raw_output.replace("**", "").strip()
    lines = [ln.rstrip() for ln in cleaned.splitlines()]

    section_aliases = {
        "match_score": ("match score", "score"),
        "job_summary": ("job summary", "role summary", "summary"),
        "strengths": ("key strengths", "strengths"),
        "gaps": ("key gaps", "gaps"),
        "suggestions": ("experience suggestions", "suggestions", "action plan", "recommendations"),
    }

    def _normalize_header_text(line: str) -> str:
        txt = line.strip().lower()
        txt = re.sub(r"^#+\s*", "", txt)   # markdown headings
        txt = re.sub(r"^[\-\*\u2022]\s*", "", txt)  # bullets
        txt = txt.rstrip(":").strip()
        return txt

    def _detect_section(line: str) -> Optional[str]:
        txt = _normalize_header_text(line)
        for key, aliases in section_aliases.items():
            for alias in aliases:
                if txt == alias:
                    return key
        return None

    score = None
    # Try strict structured score first
    score_match = re.search(r"(?:^|\n)\s*#+\s*MATCH SCORE\s*:?\s*\n+(\d{1,3})\b", cleaned, re.IGNORECASE)
    if not score_match:
        # Fallback: "Match score: 74/100" or similar
        score_match = re.search(r"match\s*score[^0-9]{0,20}(\d{1,3})\b", cleaned, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        score = max(0, min(100, score))

    # 1) Line-based parser (handles plain labels and markdown headings)
    buckets = {
        "match_score": [],
        "job_summary": [],
        "strengths": [],
        "gaps": [],
        "suggestions": [],
    }
    current = None
    for ln in lines:
        sec = _detect_section(ln)
        if sec:
            current = sec
            continue
        if current:
            stripped = ln.strip()
            if stripped:
                buckets[current].append(stripped)

    def _join_bucket(name: str) -> str:
        return "\n".join(buckets[name]).strip()

    job_summary = _join_bucket("job_summary")
    strengths = _join_bucket("strengths")
    gaps = _join_bucket("gaps")
    suggestions = _join_bucket("suggestions")

    # 2) Regex fallback for markdown heading blocks.
    def _regex_extract(aliases: tuple[str, ...]) -> str:
        escaped_aliases = [re.escape(a.upper()) for a in aliases]
        pattern = rf"(?:^|\n)\s*#+\s*(?:{'|'.join(escaped_aliases)})\s*:?\s*\n(.*?)(?=\n\s*#+\s*[A-Za-z][^\n]*\n|\Z)"
        m = re.search(pattern, cleaned.upper(), re.DOTALL)
        if not m:
            return ""
        # Use original cleaned text slice indices from a case-insensitive search.
        m2 = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        return m2.group(1).strip() if m2 else ""

    if not job_summary:
        job_summary = _regex_extract(section_aliases["job_summary"])
    if not strengths:
        strengths = _regex_extract(section_aliases["strengths"])
    if not gaps:
        gaps = _regex_extract(section_aliases["gaps"])
    if not suggestions:
        suggestions = _regex_extract(section_aliases["suggestions"])

    # Final graceful fallback so UI never renders fully blank cards.
    if not job_summary:
        lines = [ln.strip("-*• ").strip() for ln in cleaned.splitlines() if ln.strip()]
        if lines:
            job_summary = " ".join(lines[:2])

    # 3) Secondary heuristic: infer from bullet content by keywords.
    if not strengths or not gaps or not suggestions:
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        bullet_lines = [ln for ln in lines if re.match(r"^[-*•]\s+", ln)]

        if not strengths:
            strength_candidates = [
                ln for ln in bullet_lines
                if re.search(r"\b(strong|strength|experience|proficient|skilled|background)\b", ln, re.IGNORECASE)
            ]
            if strength_candidates:
                strengths = "\n".join(strength_candidates[:5])

        if not gaps:
            gap_candidates = [
                ln for ln in bullet_lines
                if re.search(r"\b(gap|missing|lack|limited|need|improve|required)\b", ln, re.IGNORECASE)
            ]
            if gap_candidates:
                gaps = "\n".join(gap_candidates[:5])

        if not suggestions:
            suggestion_candidates = [
                ln for ln in bullet_lines
                if re.search(r"\b(build|learn|take|complete|practice|create|certification|project)\b", ln, re.IGNORECASE)
            ]
            if suggestion_candidates:
                suggestions = "\n".join(suggestion_candidates[:5])

    # Final fallback text.
    if not strengths:
        strengths = "- Could not parse strengths from the model output."
    if not gaps:
        gaps = "- Could not parse gaps from the model output."
    if not suggestions:
        suggestions = "- Could not parse suggestions from the model output."

    return AnalysisResult(
        score=score,
        job_summary=job_summary,
        strengths=strengths,
        gaps=gaps,
        suggestions=suggestions,
    )
