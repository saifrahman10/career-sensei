"""
src/rag/pipeline.py
-------------------
LangChain RAG pipeline: vector store construction, chain building,
LLM prompts, and output parsing.
"""

import re
from dataclasses import dataclass
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage


# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "models/gemini-embedding-001"
LLM_MODEL        = "gemini-2.5-flash-lite-preview-09-2025"
CHUNK_SIZE       = 600
CHUNK_OVERLAP    = 40
TOP_K            = 8
MAX_OUTPUT_TOKENS = 900

GAP_PROMPT_TEMPLATE = """You are a professional career advisor. Compare the candidate's resume against the job description.

Resume sections:
{context}

Job Description:
{question}

Output EXACTLY these five sections with these exact headers. Be specific to the actual candidate and role:

## MATCH SCORE
[integer 0-100 only, nothing else]

## JOB SUMMARY
[2-3 sentences: what this role is about, the company context, and the ideal candidate profile]

## KEY STRENGTHS
[3-4 bullet points of the candidate's genuine strengths that apply to this role]

## KEY GAPS
[3-5 bullet points of concrete missing skills, experience, or qualifications]

## EXPERIENCE SUGGESTIONS
[3-4 very specific, actionable things the candidate can do — projects, certifications, or courses — to close the top gaps.]

Each bullet: ONE line only. No intro text before the first header."""


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
    Embed the resume AND job description into a shared in-memory vector store,
    then return a (gap_chain, chat_chain, memory) tuple.

    Call seed_chat_memory(memory, analysis) after you have run the gap analysis
    to give the chatbot full context from the start.

    Args:
        resume_text:     Plain text of the candidate's resume.
        job_description: Plain text of the job posting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # Tag each chunk with its source so the model can distinguish them
    resume_chunks = splitter.split_text(resume_text)
    resume_docs   = [f"[RESUME] {c}" for c in resume_chunks]

    job_chunks    = splitter.split_text(job_description)
    job_docs      = [f"[JOB DESCRIPTION] {c}" for c in job_chunks]

    all_chunks = resume_docs + job_docs

    # Embed + in-memory store
    embeddings  = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        collection_name="resume_session",
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


def seed_chat_memory(memory: ConversationBufferMemory, analysis: AnalysisResult, vectorstore=None) -> None:
    """
    Prime the chatbot memory with the completed gap analysis so that follow-up
    questions have full context about the job and candidate fit.

    Also adds the analysis as searchable text in the vector store so the
    retriever can find it when answering follow-up questions.
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

    # Add to vector store so the retriever can find it
    if vectorstore is not None:
        vectorstore.add_texts([analysis_text])

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
    def get_section(name: str) -> str:
        pattern = rf"##\s*{name}\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, raw_output, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    score = None
    score_match = re.search(r"##\s*MATCH SCORE\s*\n+(\d+)", raw_output, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))

    return AnalysisResult(
        score=score,
        job_summary=get_section("JOB SUMMARY"),
        strengths=get_section("KEY STRENGTHS"),
        gaps=get_section("KEY GAPS"),
        suggestions=get_section("EXPERIENCE SUGGESTIONS"),
    )
