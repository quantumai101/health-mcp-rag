# 🧬 Health MCP RAG · AI Portfolio Showcase

**RAG-Powered AI Portfolio · Groq (Free) · ChromaDB · FastAPI · Railway**

```
YOUR DOCUMENTS → ChromaDB → Groq LLaMA 3.3 → Live AI Portfolio Interview
```

> This application IS the demonstration. Employers can chat with an AI that answers
> questions about your skills, experience, and health/AI knowledge — powered by
> real RAG architecture you built yourself.

---

## 🚀 Live Demo

[![Launch Portfolio](https://img.shields.io/badge/Launch-AI%20Portfolio-00d4ff?style=for-the-badge)](https://your-railway-url.up.railway.app)

---

## 💡 What This Proves To Employers

| Capability | Evidence |
|---|---|
| RAG architecture | ChromaDB vector store + semantic retrieval |
| LLM integration | Groq API (OpenAI-compatible interface) |
| API development | FastAPI with async endpoints |
| Health domain knowledge | Indexed health docs, aged care guidelines |
| Deployment | Railway cloud hosting, env var management |
| Python engineering | Clean, modular, production-grade code |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   HEALTH MCP RAG                    │
├─────────────────┬───────────────────────────────────┤
│   INGESTION     │           RETRIEVAL                │
│                 │                                   │
│  Your Docs      │  User Question                    │
│  (PDF/TXT/MD)   │       ↓                           │
│       ↓         │  Embed Query                      │
│  Chunk Text     │       ↓                           │
│       ↓         │  ChromaDB Similarity Search       │
│  Embed Chunks   │       ↓                           │
│       ↓         │  Top K Relevant Chunks            │
│  ChromaDB Store │       ↓                           │
│                 │  Groq LLaMA 3.3 + System Prompt   │
│                 │       ↓                           │
│                 │  Grounded AI Response             │
└─────────────────┴───────────────────────────────────┘
```

---

## 📁 Project Structure

```
health-mcp-rag/
├── app/
│   ├── server.py          ← FastAPI server (Groq + ChromaDB)
│   └── chat_ui.html       ← Dark tech chat interface
├── core/
│   └── ingest.py          ← Document indexing pipeline
├── data/                  ← YOUR DOCUMENTS GO HERE (gitignored)
│   ├── resumes/           ← CVs, cover letters
│   ├── health_docs/       ← Health guidelines, aged care docs
│   ├── ai_publications/   ← AI/ML papers
│   ├── transcripts/       ← YouTube video transcripts
│   └── notes/             ← Personal notes, project achievements
├── chroma_db/             ← Vector index (auto-generated, gitignored)
├── .env.example           ← Environment variable template
├── requirements.txt
├── Procfile               ← Railway deployment config
└── README.md
```

---

## ⚡ Quick Start (Local)

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/health-mcp-rag
cd health-mcp-rag
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
cp .env.example .env
# Edit .env — add your GROQ_API_KEY and OWNER_NAME
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 3. Add Your Documents
```
data/
├── resumes/       ← drop your CV PDFs here
├── health_docs/   ← health guidelines, aged care documents
├── transcripts/   ← paste YouTube transcript .txt files here
└── notes/         ← any .txt or .md notes
```

### 4. Index Your Documents
```bash
python core/ingest.py
```

### 5. Run the Server
```bash
python app/server.py
# Open: http://localhost:8000
```

---

## ☁️ Deploy to Railway (Free)

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add environment variables in Railway dashboard:
   ```
   GROQ_API_KEY=gsk_...
   OWNER_NAME=Your Name
   ```
4. Railway auto-detects `Procfile` and deploys
5. Generate a public domain in Settings → Domains

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (LLaMA 3.3 70B) — **free** |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| RAG Framework | LangChain |
| API Server | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Deployment | Railway |
| Voice (optional) | ElevenLabs + pyttsx3 |

---

## 📝 Suggested Questions For Employers

- *"What AI projects has the candidate built?"*
- *"How does their experience apply to aged care settings?"*
- *"Explain how RAG differs from fine-tuning an LLM"*
- *"What is MCP and how does it relate to this system?"*
- *"Why should we hire this candidate for a health AI role?"*

---

## ⚠️ Disclaimer

This is a portfolio demonstration tool. Content is based on indexed documents and
should not be used as medical advice. Always consult qualified health professionals
for medical decisions.

---

*Built to demonstrate real AI engineering skills — not just written on a resume.*
