---
title: Health MCP RAG AI Portfolio
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
short_description: RAG AI portfolio chatbot · Gemini · ChromaDB · FastAPI
---

# 🧬 Health MCP RAG · AI Portfolio Showcase

RAG-Powered AI Portfolio · Gemini (Free) · ChromaDB · FastAPI · HuggingFace Spaces

> This application IS the demonstration. Employers can chat with an AI that answers questions about your skills, experience, and health/AI knowledge — powered by real RAG architecture you built yourself.

👉 [Chat with the AI Portfolio](https://huggingface.co/spaces/aiq00479/health-mcp-rag)

---

## What This Proves To Employers

| Capability | Evidence |
|---|---|
| RAG architecture | ChromaDB vector store + semantic retrieval |
| LLM integration | Gemini API (free tier) |
| API development | FastAPI with async endpoints + rate limiting |
| Health domain knowledge | Indexed health docs, aged care guidelines |
| Deployment | HuggingFace Spaces Docker, env var management |
| Python engineering | Clean, modular, production-grade code |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   HEALTH MCP RAG                    │
├─────────────────┬───────────────────────────────────┤
│   INGESTION     │           RETRIEVAL               │
│                 │                                   │
│  Your Docs      │  User Question                    │
│  (PDF/TXT/MD)   │       ↓                           │
│       ↓         │  Embed Query                      │
│  Chunk Text     │       ↓                           │
│       ↓         │  ChromaDB Similarity Search       │
│  Embed Chunks   │       ↓                           │
│       ↓         │  Top K Relevant Chunks            │
│  ChromaDB Store │       ↓                           │
│                 │  Gemini + System Prompt           │
│                 │       ↓                           │
│                 │  Grounded AI Response             │
└─────────────────┴───────────────────────────────────┘
```

---

## Project Structure

```
health-mcp-rag/
├── app/
│   ├── server.py          ← FastAPI server (Gemini + ChromaDB + rate limiting)
│   └── chat_ui.html       ← Dark tech chat interface with markdown rendering
├── core/
│   ├── ingest.py          ← Document indexing pipeline
│   └── ingest_official.py ← Official docs ingestion (appends to index)
├── chroma_db/             ← Vector index (baked into Docker image)
├── Dockerfile             ← HuggingFace Spaces deployment config
├── .env.example           ← Environment variable template
├── requirements.txt
└── README.md
```

---

## Quick Start (Local)

Clone and install:
```bash
git clone https://github.com/quantumai101/health-mcp-rag
cd health-mcp-rag
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

Set up environment:
```bash
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY and OWNER_NAME
```
Get your free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

Add your documents:
```
data/
├── resumes/       ← drop your CV PDFs here
├── health_docs/   ← health guidelines, aged care documents
└── notes/         ← any .txt or .md notes
```

Index and run:
```bash
python core/ingest.py
python app/server.py
# Open: http://localhost:8000
```

---

## Deploy to HuggingFace Spaces (Free)

1. Fork this repo to your HuggingFace account
2. Go to Settings → Variables and secrets, add:
   ```
   GEMINI_API_KEY=your_key_here
   OWNER_NAME=Your Name
   ```
3. Space auto-builds from Dockerfile and goes live

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini (free tier) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| RAG Framework | LangChain |
| API Server | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS + marked.js |
| Deployment | HuggingFace Spaces (Docker) |

---

## Suggested Questions For Employers

- "What AI projects has the candidate built?"
- "How does their experience apply to aged care settings?"
- "Explain how RAG differs from fine-tuning an LLM"
- "What is MCP and how does it relate to this system?"
- "Why should we hire this candidate for a health AI role?"

---

Disclaimer: This is a portfolio demonstration tool. Content is based on indexed documents and should not be used as medical advice. Always consult qualified health professionals for medical decisions.

Built to demonstrate real AI engineering skills — not just written on a resume.
