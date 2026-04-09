"""
agent.py — LangGraph agent for Health MCP RAG
Location: core/agent/agent.py

Graph flow:
  START → route → [retrieve | ingest | fallback] → respond → END

Run standalone:
  python core/agent/agent.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

# make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from core.agent.tools import (
    rag_retrieval,
    ingest_file,
    route_query,
    fallback_response,
    ALL_TOOLS,
)

# ── LLM setup (google.genai — new SDK) ───────────────────────────────────
from google import genai
from google.genai import types

_genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))

def _call_llm(system: str, user: str) -> str:
    """Thin wrapper around Gemini via google.genai SDK."""
    try:
        # Valid model names: gemini-2.5-flash-preview-04-17, gemini-2.0-flash, gemini-1.5-flash
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        response = _genai_client.models.generate_content(
            model=model_name,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=2048,
                temperature=0.7,
            ),
        )
        # response.text can be None if the model returns no candidates
        return (response.text or "").strip()
    except Exception as exc:
        return f"LLM_ERROR: {exc}"


# ── Agent state ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    query:        str
    route:        str          # 'rag_retrieval' | 'ingest_file' | 'fallback_response'
    context:      str          # raw tool output
    final_answer: str


# ── Graph nodes ──────────────────────────────────────────────────────────────

def node_route(state: AgentState) -> AgentState:
    """Decide which tool to invoke."""
    decision = route_query.invoke({"query": state["query"]})
    return {**state, "route": decision}


def node_retrieve(state: AgentState) -> AgentState:
    """Run RAG retrieval."""
    context = rag_retrieval.invoke({"query": state["query"], "k": 6})
    return {**state, "context": context}


def node_ingest(state: AgentState) -> AgentState:
    """Ingest a file with error recovery."""
    import re
    match = re.search(r'[\w/\\:.\-]+(\.pdf|\.txt|\.md|\.html)', state["query"], re.I)
    if match:
        result = ingest_file.invoke({"file_path": match.group(0)})
    else:
        result = (
            "INGEST_INFO: No file path detected in query. "
            "Please provide a full path, e.g. 'ingest Data/my_cv.pdf'"
        )
    return {**state, "context": result}


def node_fallback(state: AgentState) -> AgentState:
    """Return a helpful fallback message."""
    context = fallback_response.invoke({"query": state["query"]})
    return {**state, "context": context}


def node_respond(state: AgentState) -> AgentState:
    """Synthesise context + LLM into a final answer."""
    context = state.get("context", "")
    query   = state["query"]
    route   = state.get("route", "rag_retrieval")

    if route == "ingest_file" or context.startswith("INGEST"):
        answer = context
    elif context in ("NO_RESULTS", "") or context.startswith("RETRIEVAL_ERROR"):
        answer = fallback_response.invoke({"query": query})
    else:
        system_prompt = (
            "You are an AI Portfolio assistant for a health-tech candidate named Zhi. "
            "Answer the user's question using ONLY the context passages below. "
            "Be concise, professional, and cite the source file where relevant. "
            "If the context does not contain enough information, say so honestly.\n\n"
            f"CONTEXT:\n{context}"
        )
        answer = _call_llm(system_prompt, query)

    updated_messages = state["messages"] + [AIMessage(content=answer)]
    return {**state, "final_answer": answer, "messages": updated_messages}


# ── Routing function ─────────────────────────────────────────────────────────

def _route_edge(state: AgentState) -> str:
    return state.get("route", "rag_retrieval")


# ── Build graph ──────────────────────────────────────────────────────────────

def build_graph() -> CompiledStateGraph:   # correct return type
    g = StateGraph(AgentState)

    g.add_node("route",    node_route)
    g.add_node("retrieve", node_retrieve)
    g.add_node("ingest",   node_ingest)
    g.add_node("fallback", node_fallback)
    g.add_node("respond",  node_respond)

    g.add_edge(START, "route")

    g.add_conditional_edges(
        "route",
        _route_edge,
        {
            "rag_retrieval":     "retrieve",
            "ingest_file":       "ingest",
            "fallback_response": "fallback",
        },
    )

    g.add_edge("retrieve", "respond")
    g.add_edge("ingest",   "respond")
    g.add_edge("fallback", "respond")
    g.add_edge("respond",  END)

    return g.compile()   # CompiledStateGraph — has .invoke()


# ── Public API ───────────────────────────────────────────────────────────────

_graph: CompiledStateGraph | None = None

def get_agent() -> CompiledStateGraph:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, history: list[BaseMessage] | None = None) -> str:
    """
    Main entry point. Call this from server.py or any other module.

    Example:
        from core.agent.agent import run_query
        answer = run_query("What are Zhi's Python skills?")
    """
    agent = get_agent()
    initial_state: AgentState = {
        "messages":     (history or []) + [HumanMessage(content=query)],
        "query":        query,
        "route":        "",
        "context":      "",
        "final_answer": "",
    }
    result = agent.invoke(initial_state)
    return result["final_answer"]


# ── CLI for standalone testing ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Health MCP RAG — LangGraph Agent (standalone)")
    print("  Type 'exit' to quit")
    print("=" * 60)

    history: list[BaseMessage] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if not user_input:
            continue

        answer = run_query(user_input, history)
        print(f"\nAgent: {answer}")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=answer))
        history = history[-20:]
