#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   HEALTH-MCP-RAG · AI Portfolio Server                      ║
║   Stack: Gemini (free) · ChromaDB · FastAPI                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import io
import time
import shutil
import threading
from pathlib import Path
from datetime import date, datetime
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

DATA_DIR   = Path(os.getenv("DATA_PATH",   "./data"))
CHROMA_DIR = Path(os.getenv("CHROMA_PATH", "./chroma_db"))
OWNER_NAME   = os.getenv("OWNER_NAME", "the candidate")
GEMINI_MODEL = "gemini-3-flash-preview"
# GEMINI_MODEL = "gemini-1.5-flash"
RETRIEVAL_K  = 6
DEEP_K       = 12

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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_embeddings  = None
_vectorstore = None
_gemini      = None

def get_embeddings():
    global _embeddings
    if _embeddings is None and CHROMA_AVAILABLE:
        print("  Loading embeddings model…")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("  ✅ Embeddings ready")
    return _embeddings

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None and CHROMA_AVAILABLE:
        if not CHROMA_DIR.exists():
            return None
        _vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())
        print(f"  ✅ Vector store ready ({_vectorstore._collection.count():,} chunks)")
    return _vectorstore

def get_gemini():
    global _gemini
    if _gemini is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env file")
        genai.configure(api_key=api_key)
        _gemini = genai.GenerativeModel(GEMINI_MODEL)
    return _gemini

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
async def chat(req: ChatRequest):
    t0 = time.time()
    context, sources = "", []

    vs = get_vectorstore()
    if vs:
        k = DEEP_K if req.deep else RETRIEVAL_K
        try:
            docs = vs.similarity_search(req.text, k=k)
            parts = []
            for doc in docs:
                src  = doc.metadata.get("source", "unknown")
                name = Path(src).name if src != "unknown" else "knowledge base"
                sources.append(name)
                parts.append(f"[Source: {name}]\n{doc.page_content}")
            context = "\n\n".join(parts)
        except Exception as e:
            context = f"[Retrieval error: {e}]"
    else:
        context = "[Knowledge base not yet built — run ingest.py to index your documents]"

    system = PERSONA_PROMPT.format(
        name=OWNER_NAME,
        date=date.today().strftime("%B %d, %Y"),
        context=context or "No specific documents retrieved."
    )

    history_text = ""
    for msg in req.conversation_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"\n{role}: {msg['content']}"

    full_prompt = f"{system}\n\nConversation:{history_text}\n\nUser: {req.text}\nAssistant:"

    try:
        model    = get_gemini()
        response = model.generate_content(full_prompt)
        reply    = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    return ChatResponse(
        reply=reply,
        sources=list(dict.fromkeys(sources))[:5],
        latency_ms=int((time.time() - t0) * 1000)
    )

@app.get("/stats")
async def get_stats():
    stats = {"owner_name": OWNER_NAME, "doc_count": 0, "chunk_count": 0,
             "storage_gb": "—", "model": f"Gemini / {GEMINI_MODEL}", "chroma_ready": CHROMA_DIR.exists()}
    vs = get_vectorstore()
    if vs:
        try: stats["chunk_count"] = vs._collection.count()
        except: pass
    if DATA_DIR.exists():
        docs = list(DATA_DIR.rglob("*.txt")) + list(DATA_DIR.rglob("*.pdf")) + list(DATA_DIR.rglob("*.md"))
        stats["doc_count"] = len(docs)
    try:
        total, used, _ = shutil.disk_usage(str(DATA_DIR) if DATA_DIR.exists() else ".")
        stats["storage_gb"] = f"{used/(1024**3):.1f} / {total/(1024**3):.0f} GB"
    except: pass
    return stats

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": GEMINI_MODEL, "chroma": CHROMA_DIR.exists(), "time": datetime.now().isoformat()}

@app.post("/ingest/trigger")
async def trigger_ingest(background_tasks: BackgroundTasks):
    def run():
        import subprocess
        subprocess.run(["python", "core/ingest.py"], capture_output=True, text=True)
    background_tasks.add_task(run)
    return {"status": "Ingestion started"}

@app.on_event("startup")
async def startup():
    print(f"\n{'═'*58}")
    print(f"  🧬 Health MCP RAG · {OWNER_NAME}")
    print(f"  🤖 LLM:      Gemini / {GEMINI_MODEL} (free)")
    print(f"  🗄️  ChromaDB: {'✅ ready' if CHROMA_DIR.exists() else '❌ run ingest.py'}")
    print(f"  🌐 Open:     http://localhost:8000")
    print(f"{'═'*58}\n")
    threading.Thread(target=get_vectorstore, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, log_level="warning")
