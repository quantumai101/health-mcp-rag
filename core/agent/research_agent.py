"""
research_agent.py — LangGraph agentic AI research demo
Location: core/agent/research_agent.py

Graph flow:
  START → route → execute → synthesise? → respond → END

Shows employers: multi-tool routing, live web research, LLM synthesis,
step-by-step reasoning visibility — all focused on AI/BCI/human augmentation.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Annotated, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from core.agent.research_tools import (
    web_search_latest,
    search_ai_bci_research,
    search_edge_hardware,
    search_human_ai_interface,
    synthesise_findings,
    route_research_query,
    _llm,          # reuse same Groq client for safety-net synthesis
)


# ── Agent state ──────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    messages:      Annotated[list[BaseMessage], add_messages]
    query:         str
    tools_plan:    list[str]
    tool_outputs:  dict[str, str]
    steps:         list[str]
    final_answer:  str


# ── Node: route ──────────────────────────────────────────────────────────────

def node_route(state: ResearchState) -> ResearchState:
    raw    = route_research_query.invoke({"query": state["query"]})
    plan   = json.loads(raw)
    tools  = plan["tools"]
    reason = plan["reason"]
    step   = f"🔀 **Router** → Selected tools: `{'` → `'.join(tools)}`  Reason: {reason}"
    return {
        **state,
        "tools_plan": tools,
        "steps": state.get("steps", []) + [step],
    }


# ── Node: execute tools in sequence ─────────────────────────────────────────

def node_execute_tools(state: ResearchState) -> ResearchState:
    outputs = dict(state.get("tool_outputs", {}))
    steps   = list(state.get("steps", []))
    plan    = state["tools_plan"]

    tool_map = {
        "web_search_latest":         web_search_latest,
        "search_ai_bci_research":    search_ai_bci_research,
        "search_edge_hardware":      search_edge_hardware,
        "search_human_ai_interface": search_human_ai_interface,
    }

    for tool_name in plan:
        if tool_name in ("synthesise_findings", "route_research_query"):
            continue

        tool_fn = tool_map.get(tool_name)
        if not tool_fn:
            continue

        steps.append(f"⚙️  **{tool_name}** → Searching knowledge base (k=12)…")

        try:
            # each tool accepts different param names
            if tool_name == "web_search_latest":
                result = tool_fn.invoke({"query": state["query"]})
            elif tool_name == "search_ai_bci_research":
                result = tool_fn.invoke({"topic": state["query"]})
            else:
                result = tool_fn.invoke({"query": state["query"]})

            outputs[tool_name] = result
            preview = result[:120].replace("\n", " ")
            steps.append(f"   ✅ `{tool_name}` complete → {preview}…")
        except Exception as e:
            outputs[tool_name] = f"ERROR: {e}"
            steps.append(f"   ❌ `{tool_name}` failed: {e}")

    return {**state, "tool_outputs": outputs, "steps": steps}


# ── Node: synthesise (multi-tool only) ───────────────────────────────────────

def node_synthesise(state: ResearchState) -> ResearchState:
    outputs = state.get("tool_outputs", {})
    steps   = list(state.get("steps", []))

    # only synthesise when multiple tools ran
    if "synthesise_findings" not in state["tools_plan"] or len(outputs) < 2:
        return state

    steps.append("🔬 **synthesise_findings** → Combining all research streams…")
    combined = "\n\n".join(
        f"=== {k} ===\n{v}" for k, v in outputs.items()
    )
    try:
        synthesis = synthesise_findings.invoke({"findings": combined})
        outputs["synthesise_findings"] = synthesis
        steps.append("   ✅ Synthesis complete")
    except Exception as e:
        outputs["synthesise_findings"] = f"SYNTHESIS_ERROR: {e}"
        steps.append(f"   ❌ Synthesis failed: {e}")

    return {**state, "tool_outputs": outputs, "steps": steps}


# ── Node: respond ────────────────────────────────────────────────────────────

def node_respond(state: ResearchState) -> ResearchState:
    outputs = state.get("tool_outputs", {})
    steps   = list(state.get("steps", []))
    query   = state["query"]

    # ── Pick the best available output ───────────────────────────────────
    if "synthesise_findings" in outputs:
        # multi-tool synthesis — already clean LLM output
        answer = outputs["synthesise_findings"]

    elif len(outputs) == 1:
        # single tool — all tools now return LLM-synthesised text, use directly
        answer = list(outputs.values())[0]

    else:
        # fallback: something went wrong, do a final LLM pass on whatever we have
        steps.append("🔬 **synthesise_findings** → Final LLM synthesis pass…")
        combined = "\n\n".join(outputs.values()) if outputs else "No results retrieved."
        answer = _llm(
            system=(
                "You are an expert AI portfolio assistant. "
                "Synthesise the following research findings into a clear, professional answer. "
                "Use ## headings, write in paragraphs, no raw source citations."
            ),
            user=f"Query: {query}\n\nFindings:\n{combined}",
            max_tokens=1200,
        )

    steps.append("💬 **respond** → Final answer ready")
    updated_msgs = state["messages"] + [AIMessage(content=answer)]

    return {
        **state,
        "final_answer": answer,
        "steps":        steps,
        "messages":     updated_msgs,
    }


# ── Build graph ──────────────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    g = StateGraph(ResearchState)

    g.add_node("route",      node_route)
    g.add_node("execute",    node_execute_tools)
    g.add_node("synthesise", node_synthesise)
    g.add_node("respond",    node_respond)

    g.add_edge(START,         "route")
    g.add_edge("route",       "execute")
    g.add_edge("execute",     "synthesise")
    g.add_edge("synthesise",  "respond")
    g.add_edge("respond",     END)

    return g.compile()


_research_graph = None

def get_research_agent():
    global _research_graph
    if _research_graph is None:
        _research_graph = build_research_graph()
    return _research_graph


def run_research(query: str) -> dict:
    """
    Main entry point.
    Returns: {
        "answer":     str,        ← clean LLM-synthesised response
        "steps":      list[str],  ← visible reasoning trace for UI
        "tools_used": list[str]
    }
    """
    agent = get_research_agent()
    initial: ResearchState = {
        "messages":     [HumanMessage(content=query)],
        "query":        query,
        "tools_plan":   [],
        "tool_outputs": {},
        "steps":        [],
        "final_answer": "",
    }
    result = agent.invoke(initial)
    return {
        "answer":     result["final_answer"],
        "steps":      result.get("steps", []),
        "tools_used": result.get("tools_plan", []),
    }


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Research Agent — BCI / AI / Human Augmentation")
    print("  Type 'exit' to quit")
    print("=" * 60)

    while True:
        try:
            q = input("\nResearch query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", ""):
            break

        result = run_research(q)
        print("\n--- REASONING STEPS ---")
        for step in result["steps"]:
            print(step)
        print("\n--- ANSWER ---")
        print(result["answer"])
