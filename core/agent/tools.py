"""
tools.py — LangGraph-compatible tools for Health MCP RAG agent
Location: core/agent/tools.py

Handles:
- RAG retrieval from ChromaDB vector store
- PDF/file ingestion with error recovery
- Query routing decision
- Fallback response when no good answer found
"""

from __future__ import annotations

import os
import re
import traceback
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── paths (relative to project root) ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # health-mcp-rag/
CHROMA_DIR   = PROJECT_ROOT / "chroma_db"
DATA_DIR     = PROJECT_ROOT / "Data"

# ── shared singletons ────────────────────────────────────────────────────────
_embeddings   = None
_vectorstore  = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print("  [tools] Loading embeddings model…")
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("  [tools] ✅ Embeddings ready")
    return _embeddings


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=_get_embeddings(),
        )
    return _vectorstore


# ── Tool 1: RAG retrieval ────────────────────────────────────────────────────

@tool
def rag_retrieval(query: str, k: int = 6) -> str:
    """
    Search the ChromaDB vector store for chunks relevant to the query.
    Returns formatted context passages with source metadata.
    Use this for any question about skills, experience, health projects,
    aged care standards, or technical capabilities.
    """
    try:
        vs = _get_vectorstore()
        results = vs.similarity_search_with_score(query, k=k)

        if not results:
            return "NO_RESULTS"

        passages = []
        for i, (doc, score) in enumerate(results, 1):
            source   = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "general")
            passages.append(
                f"[{i}] SOURCE: {source} | CATEGORY: {category} | "
                f"SCORE: {score:.3f}\n{doc.page_content.strip()}"
            )

        return "\n\n---\n\n".join(passages)

    except Exception as exc:
        return f"RETRIEVAL_ERROR: {exc}\n{traceback.format_exc()}"


# ── Tool 2: PDF / file ingestion with error recovery ────────────────────────

@tool
def ingest_file(file_path: str) -> str:
    """
    Ingest a single file (PDF or TXT) into ChromaDB with full error recovery.
    Tries PDF extraction first; falls back to raw text read on failure.
    Returns a status message describing what was indexed.
    """
    path = Path(file_path)
    if not path.exists():
        return f"INGEST_ERROR: File not found — {file_path}"

    suffix = path.suffix.lower()
    text   = ""
    method = ""

    # ── attempt PDF extraction ───────────────────────────────────────────────
    if suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            text   = "\n".join(pages).strip()
            method = "pdfplumber"
        except Exception as pdf_err:
            # fallback: try pypdf
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                text   = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                ).strip()
                method = "pypdf (fallback)"
            except Exception as pypdf_err:
                return (
                    f"INGEST_ERROR: Both PDF extractors failed.\n"
                    f"  pdfplumber: {pdf_err}\n"
                    f"  pypdf:      {pypdf_err}"
                )

    # ── plain text ───────────────────────────────────────────────────────────
    elif suffix in (".txt", ".md", ".html"):
        try:
            text   = path.read_text(encoding="utf-8", errors="ignore")
            method = "text-read"
        except Exception as txt_err:
            return f"INGEST_ERROR: Cannot read text file — {txt_err}"
    else:
        return f"INGEST_ERROR: Unsupported file type '{suffix}'"

    if not text.strip():
        return f"INGEST_WARN: No text extracted from {path.name}"

    # ── chunk & upsert ───────────────────────────────────────────────────────
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        chunks = splitter.split_text(text)
        docs   = [
            Document(
                page_content=chunk,
                metadata={"source": path.name, "method": method},
            )
            for chunk in chunks
        ]
        vs = _get_vectorstore()
        vs.add_documents(docs)
        return (
            f"✅ Ingested '{path.name}' via {method}: "
            f"{len(chunks)} chunks added to ChromaDB."
        )
    except Exception as exc:
        return f"INGEST_ERROR: ChromaDB upsert failed — {exc}"


# ── Tool 3: Query router ─────────────────────────────────────────────────────

@tool
def route_query(query: str) -> str:
    """
    Analyse the user query and return which tool should handle it.
    Returns one of: 'rag_retrieval', 'ingest_file', 'fallback_response'.
    Use this as the first step in the agent graph to decide the next action.
    """
    q = query.lower().strip()

    ingest_signals = [
        "ingest", "upload", "add file", "index", "load document",
        "process pdf", "import",
    ]
    if any(sig in q for sig in ingest_signals):
        return "ingest_file"

    # short / ambiguous / greeting
    if len(q.split()) <= 2 or re.match(r"^(hi|hello|hey|thanks|ok|yes|no)$", q):
        return "fallback_response"

    # everything else → RAG
    return "rag_retrieval"


# ── Tool 4: Fallback response ────────────────────────────────────────────────

@tool
def fallback_response(query: str) -> str:
    """
    Generate a helpful fallback message when no good RAG result is available
    or the query is too vague to answer from the knowledge base.
    """
    suggestions = [
        "Try asking about a specific skill, technology, or project.",
        "Ask about aged care standards, health data systems, or AI architecture.",
        "Ask about the candidate's experience with a specific role or company.",
    ]
    return (
        f"I couldn't find a confident answer for: \"{query}\"\n\n"
        "Suggestions:\n" + "\n".join(f"  • {s}" for s in suggestions)
    )


# ── export list for agent ────────────────────────────────────────────────────
ALL_TOOLS = [rag_retrieval, ingest_file, route_query, fallback_response]
