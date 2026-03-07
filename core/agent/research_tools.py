"""
research_tools.py — Agentic AI research tools for employer demo
Location: core/agent/research_tools.py

Tools showcased:
- web_search_latest       : Deep search + LLM synthesis (matches DEEP mode)
- search_ai_bci_research  : Filtered search for AI/BCI/human augmentation
- search_edge_hardware    : Search for Mini Mac M4/M5, neuromorphic, edge AI
- search_human_ai_interface: Human-AI-Machine interface latest research
- synthesise_findings     : Synthesise multi-tool results into a report
- route_research_query    : Decide which research tool(s) to use
"""

from __future__ import annotations

import os
import re
import json
import time
import traceback
from typing import Any

from langchain_core.tools import tool

# ── Groq LLM for synthesis (reuses existing .env key) ───────────────────────
from groq import Groq as _Groq
_groq_client = None

def _get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = _Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    return _groq_client

def _llm(system: str, user: str, max_tokens: int = 1500) -> str:
    try:
        r = _get_groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM_ERROR: {e}"


# ── ChromaDB deep search helper (k=12, same as main site DEEP mode) ─────────

from pathlib import Path as _Path
from langchain_huggingface import HuggingFaceEmbeddings as _HFEmb
from langchain_chroma import Chroma as _Chroma

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
_CHROMA_DIR   = _PROJECT_ROOT / "chroma_db"
_vs = None
_emb = None

def _get_vs():
    global _vs, _emb
    if _emb is None:
        _emb = _HFEmb(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    if _vs is None:
        _vs = _Chroma(persist_directory=str(_CHROMA_DIR), embedding_function=_emb)
    return _vs

def _deep_search(query: str, k: int = 12) -> str:
    """Search ChromaDB with k=12 (DEEP mode) and return formatted passages."""
    try:
        vs = _get_vs()
        docs = vs.similarity_search(query, k=k)
        if not docs:
            return "No results found in knowledge base."
        parts = []
        for i, doc in enumerate(docs, 1):
            src  = doc.metadata.get("source", "unknown")
            name = _Path(src).name if src != "unknown" else "knowledge base"
            parts.append(f"[{i}] SOURCE: {name}\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Search error: {e}"


# ── Tool 1: General deep knowledge base search + LLM synthesis ──────────────

@tool
def web_search_latest(query: str) -> str:
    """
    Deep search the knowledge base (k=12) for the latest AI technology,
    research topics, technical concepts, and domain knowledge.
    Use for general queries about AI developments, tools, frameworks,
    MCP, RAG, health AI, or any technical topic.
    Retrieves top 12 passages then synthesises a clean answer via LLM.
    """
    raw = _deep_search(query, k=12)

    # ── FIX: Always synthesise raw chunks into a clean LLM answer ──────────
    summary = _llm(
        system=(
            "You are an expert AI portfolio assistant for a skilled AI engineer. "
            "You are talking to potential employers, recruiters, and technical interviewers. "
            "\n\n"
            "Your job: synthesise the search results below into a clear, professional, "
            "well-structured answer to the user's query. "
            "\n\n"
            "Rules:\n"
            "- Do NOT dump raw source chunks or '[1] SOURCE:' citations\n"
            "- Write in flowing, readable paragraphs with clear sections\n"
            "- Use ## headings to structure your response\n"
            "- Be technically precise and specific\n"
            "- Highlight the most important and relevant findings\n"
            "- Connect findings to real-world AI engineering applications\n"
            "- Keep the tone: confident, knowledgeable, professional\n"
            "- Length: 3-5 paragraphs or structured sections, not too long\n"
        ),
        user=f"Query: {query}\n\nKnowledge base search results:\n\n{raw}",
        max_tokens=1200,
    )
    return summary


# ── Tool 2: AI/BCI & human augmentation research ─────────────────────────────

@tool
def search_ai_bci_research(topic: str) -> str:
    """
    Search for the latest research on Brain-Computer Interfaces (BCI),
    human augmentation, carbon-to-silicon body transformation,
    neuralink-style implants, neural dust, and human-AI symbiosis.
    Use when the query involves BCI, neural interfaces, human augmentation,
    cyborg technology, or transforming biological humans with silicon/AI.
    """
    combined_query = f"{topic} brain computer interface neural implant human augmentation"
    raw = _deep_search(combined_query, k=12)

    summary = _llm(
        system=(
            "You are a cutting-edge AI/BCI research analyst. "
            "Synthesise the search results below focusing on: "
            "1) Latest BCI/neural interface breakthroughs, "
            "2) Human-silicon hybrid body research, "
            "3) AI-driven human augmentation technologies. "
            "Be specific about technologies, companies, and research groups. "
            "Format with clear ## sections and highlight the most significant findings. "
            "Do NOT include raw SOURCE citations — write in professional prose."
        ),
        user=f"Topic: {topic}\n\nSearch results:\n{raw}"
    )
    return summary


# ── Tool 3: Edge hardware & neuromorphic computing ───────────────────────────

@tool
def search_edge_hardware(query: str) -> str:
    """
    Search for latest edge AI hardware including Apple Silicon (M4/M5),
    neuromorphic chips, on-device AI, miniaturised compute for BCI,
    and compact AI-capable devices for human-machine interfaces.
    Use when query involves hardware, chips, Mac M4/M5, edge computing,
    neuromorphic, or physical AI devices.
    """
    combined_query = f"{query} edge AI hardware neuromorphic chip Apple Silicon"
    raw = _deep_search(combined_query, k=12)

    summary = _llm(
        system=(
            "You are an edge AI hardware expert specialising in neuromorphic computing "
            "and miniaturised AI devices. Summarise the search results focusing on: "
            "1) Apple M4/M5 capabilities for AI/BCI workloads, "
            "2) Neuromorphic chips (Intel Loihi, IBM NorthPole, etc.), "
            "3) Ultra-compact AI hardware for implantable/wearable use. "
            "Be specific about specs, use cases, and research breakthroughs. "
            "Do NOT include raw SOURCE citations — write in professional prose."
        ),
        user=f"Query: {query}\n\nSearch results:\n{raw}"
    )
    return summary


# ── Tool 4: Human-AI-Machine interface research ──────────────────────────────

@tool
def search_human_ai_interface(query: str) -> str:
    """
    Search for latest research on Human-AI-Machine interfaces including
    multimodal interaction, AR/VR/XR neural integration, digital twins,
    embodied AI, and next-generation HCI (Human Computer Interaction).
    Use when query involves human-machine interaction, AR/VR, digital twins,
    embodied AI, or multimodal AI interfaces.
    """
    combined_query = f"{query} human AI machine interface multimodal embodied AR VR"
    raw = _deep_search(combined_query, k=12)

    summary = _llm(
        system=(
            "You are a Human-AI interaction researcher. Summarise findings focusing on: "
            "1) Latest multimodal AI interfaces (voice/vision/gesture/neural), "
            "2) AR/VR/XR integration with AI agents, "
            "3) Embodied AI and physical world interaction, "
            "4) Digital twin technologies for human augmentation. "
            "Highlight research from top labs: DeepMind, OpenAI, Meta AI, MIT, Stanford. "
            "Do NOT include raw SOURCE citations — write in professional prose."
        ),
        user=f"Query: {query}\n\nSearch results:\n{raw}"
    )
    return summary


# ── Tool 5: Synthesise multi-source findings ─────────────────────────────────

@tool
def synthesise_findings(findings: str) -> str:
    """
    Takes multiple research findings from different tools and synthesises
    them into a single coherent research report with key insights,
    technology convergence points, and implications for AI engineering.
    Use this as the FINAL step after gathering results from other tools.
    """
    report = _llm(
        system=(
            "You are a senior AI research analyst writing for a technical audience "
            "of AI engineers, health-tech CTOs, and ML researchers. "
            "Synthesise the provided research findings into a structured report with:\n"
            "## Executive Summary\n"
            "## Key Technology Breakthroughs\n"
            "## Convergence Points (where BCI + edge AI + human interfaces meet)\n"
            "## Implications for Health & Aged Care AI\n"
            "## What This Means for AI Engineers\n\n"
            "Be technically precise, cite specific technologies/companies/papers where mentioned. "
            "Do NOT include raw SOURCE citations — write in professional prose. "
            "This report demonstrates the analyst's deep understanding of the AI frontier."
        ),
        user=f"Research findings to synthesise:\n\n{findings}",
        max_tokens=2000,
    )
    return report


# ── Tool 6: Query router for research agent ──────────────────────────────────

@tool
def route_research_query(query: str) -> str:
    """
    Analyse the research query and return a JSON list of tools to use in order.
    Available tools: web_search_latest, search_ai_bci_research,
    search_edge_hardware, search_human_ai_interface, synthesise_findings.
    Always end multi-tool sequences with synthesise_findings.
    Returns JSON: {"tools": ["tool1", "tool2", ...], "reason": "..."}
    """
    q = query.lower()

    bci_signals       = ["bci", "brain", "neural", "neuralink", "implant",
                         "carbon silicon", "augment", "cyborg", "symbiosis"]
    hardware_signals  = ["m4", "m5", "mac", "chip", "hardware", "neuromorphic",
                         "edge", "device", "compute", "loihi"]
    interface_signals = ["interface", "hci", "ar", "vr", "xr", "multimodal",
                         "embodied", "digital twin", "interaction"]

    tools_needed = []

    if any(s in q for s in bci_signals):
        tools_needed.append("search_ai_bci_research")
    if any(s in q for s in hardware_signals):
        tools_needed.append("search_edge_hardware")
    if any(s in q for s in interface_signals):
        tools_needed.append("search_human_ai_interface")

    # default: general deep search
    if not tools_needed:
        tools_needed.append("web_search_latest")

    # always synthesise if more than one tool
    if len(tools_needed) > 1:
        tools_needed.append("synthesise_findings")

    reason = (
        f"Query '{query[:60]}' matched: "
        + ", ".join(t for t in tools_needed if t != "synthesise_findings")
    )

    return json.dumps({"tools": tools_needed, "reason": reason})


# ── Export ───────────────────────────────────────────────────────────────────

RESEARCH_TOOLS = [
    web_search_latest,
    search_ai_bci_research,
    search_edge_hardware,
    search_human_ai_interface,
    synthesise_findings,
    route_research_query,
]
