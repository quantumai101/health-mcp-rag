#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   HEALTH-MCP-RAG · AI Portfolio Server                      ║
║   Stack: Groq/LLaMA · ChromaDB · FastAPI · LangGraph        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import io
import sys
import time
import shutil
import threading
from pathlib import Path
from datetime import date, datetime
from collections import defaultdict
from dotenv import load_dotenv

# ── Ensure project root is on path (works when run from app/ folder) ───────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── LangGraph agent (replaces direct Gemini/LLM call) ─────────────────────
from core.agent.agent import run_query as agent_run_query
from langchain_core.messages import HumanMessage, AIMessage

# ── Research agent (BCI / human augmentation demo) ─────────────────────────
from core.agent.research_agent import run_research

try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

DATA_DIR   = Path(os.getenv("DATA_PATH",   "./data"))
CHROMA_DIR = Path(os.getenv("CHROMA_PATH", "./chroma_db"))
OWNER_NAME   = os.getenv("OWNER_NAME", "the candidate")

# ── Model label (display only — LLM is now handled by agent) ──────────────
LLM_LABEL   = "LLaMA 3.3 · Groq"
RETRIEVAL_K = 6
DEEP_K      = 12

# ── RATE LIMITING ─────────────────────────────────────────────────────────
# Max 10 requests per IP per minute (protects free API quota)
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW   = 60  # seconds
_request_counts: dict = defaultdict(list)

def check_rate_limit(ip: str):
    now = time.time()
    _request_counts[ip] = [t for t in _request_counts[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_request_counts[ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Max {RATE_LIMIT_REQUESTS} per minute. Please wait and try again."
        )
    _request_counts[ip].append(now)

# ── Persona prompt (kept for context passed into agent) ───────────────────
PERSONA_PROMPT = """You are an AI portfolio assistant representing {name}.
You are deployed as a live demonstration of {name}'s practical AI engineering skills —
specifically their ability to build RAG systems, MCP integrations, and AI-powered
applications for health and aged care contexts.

Your role when speaking to employers or interviewers:
- Answer questions about {name}'s experience, skills, and projects confidently
- Explain AI/ML concepts clearly showing deep technical understanding
- Reference specific documents, projects, or achievements from the knowledge base
- Demonstrate that {name} can build production-grade AI systems (because you ARE one)
- Connect technical AI skills to real-world health and aged care applications

Technical areas: RAG, MCP, ChromaDB, LangChain, FastAPI, Python,
Health informatics, aged care workflows, Dept of Health documentation,
AI ethics and responsible deployment in healthcare.

Tone: Confident, professional, technically precise. Warm when needed.
Never say "as an AI" — you are {name}'s living portfolio.

Today's date: {date}

━━━ RETRIEVED FROM KNOWLEDGE BASE ━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

app = FastAPI(title="Health MCP RAG · AI Portfolio", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

_embeddings  = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None and CHROMA_AVAILABLE:
        print("  Loading embeddings model…")
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("  ✅ Embeddings ready")
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None and CHROMA_AVAILABLE:
        if not CHROMA_DIR.exists():
            return None
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=get_embeddings()
        )
        print(f"  ✅ Vector store ready ({_vectorstore._collection.count():,} chunks)")
    return _vectorstore


class ChatRequest(BaseModel):
    text: str
    deep: bool = False
    conversation_history: list = []


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = []
    latency_ms: int = 0


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = Path(__file__).parent / "chat_ui.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>chat_ui.html not found</h1>")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    # ── Rate limit check ──────────────────────────────────────────────────
    client_ip = request.client.host
    check_rate_limit(client_ip)

    t0 = time.time()
    sources: list[str] = []

    # ── Get sources for UI display (agent handles actual retrieval) ───────
    vs = get_vectorstore()
    if vs:
        k = DEEP_K if req.deep else RETRIEVAL_K
        try:
            docs = vs.similarity_search(req.text, k=k)
            for doc in docs:
                src  = doc.metadata.get("source", "unknown")
                name = Path(src).name if src != "unknown" else "knowledge base"
                sources.append(name)
        except Exception as e:
            pass  # agent will still respond via fallback

    # ── Build conversation history for agent ──────────────────────────────
    history_msgs = []
    for msg in req.conversation_history[-6:]:
        if msg["role"] == "user":
            history_msgs.append(HumanMessage(content=msg["content"]))
        else:
            history_msgs.append(AIMessage(content=msg["content"]))

    # ── Run LangGraph agent (routing → retrieval → LLM synthesis) ─────────
    try:
        reply = agent_run_query(req.text, history=history_msgs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return ChatResponse(
        reply=reply,
        sources=list(dict.fromkeys(sources))[:5],
        latency_ms=int((time.time() - t0) * 1000)
    )


@app.get("/stats")
async def get_stats():
    stats = {
        "owner_name":   OWNER_NAME,
        "doc_count":    0,
        "chunk_count":  0,
        "storage_gb":   "—",
        "model":        f"Groq / {LLM_LABEL}",
        "chroma_ready": CHROMA_DIR.exists(),
    }
    vs = get_vectorstore()
    if vs:
        try:
            stats["chunk_count"] = vs._collection.count()
        except:
            pass
    if DATA_DIR.exists():
        docs = (
            list(DATA_DIR.rglob("*.txt")) +
            list(DATA_DIR.rglob("*.pdf")) +
            list(DATA_DIR.rglob("*.md"))
        )
        stats["doc_count"] = len(docs)
    try:
        total, used, _ = shutil.disk_usage(str(DATA_DIR) if DATA_DIR.exists() else ".")
        stats["storage_gb"] = f"{used/(1024**3):.1f} / {total/(1024**3):.0f} GB"
    except:
        pass
    return stats


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model":  LLM_LABEL,
        "chroma": CHROMA_DIR.exists(),
        "time":   datetime.now().isoformat(),
    }


@app.post("/ingest/trigger")
async def trigger_ingest(background_tasks: BackgroundTasks):
    def run():
        import subprocess
        subprocess.run(["python", "core/ingest.py"], capture_output=True, text=True)
    background_tasks.add_task(run)
    return {"status": "Ingestion started"}


@app.get("/agent-demo", response_class=HTMLResponse)
async def serve_research_demo():
    demo_path = Path(__file__).parent / "research_demo.html"
    if demo_path.exists():
        return HTMLResponse(content=demo_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>research_demo.html not found</h1>")


class ResearchRequest(BaseModel):
    query: str


@app.post("/research")
async def research(req: ResearchRequest, request: Request):
    client_ip = request.client.host
    check_rate_limit(client_ip)
    try:
        result = run_research(req.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research agent error: {str(e)}")


@app.on_event("startup")
async def startup():
    print(f"\n{'═'*58}")
    print(f"  🧬 Health MCP RAG · {OWNER_NAME}")
    print(f"  🤖 LLM:      {LLM_LABEL} (via LangGraph agent)")
    print(f"  🗄️  ChromaDB: {'✅ ready' if CHROMA_DIR.exists() else '❌ run ingest.py'}")
    print(f"  🛡️  Rate limit: {RATE_LIMIT_REQUESTS} req/min per IP")
    print(f"  🌐 Open:     http://localhost:8000")
    print(f"{'═'*58}\n")
    threading.Thread(target=get_vectorstore, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, log_level="warning")
