#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   HEALTH-MCP-RAG · OFFICIAL DOCS INGESTION                  ║
║   Downloads & indexes official sources into ChromaDB        ║
║   APPENDS to existing index (does NOT wipe CV chunks)       ║
╚══════════════════════════════════════════════════════════════╝

USAGE:
    python core/ingest_official.py

Run AFTER ingest.py so your CV chunks are already indexed.
This script adds official docs on top of them.
"""

import os
import pathlib
import requests
import time
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ── IMPORTS ──────────────────────────────────────────────────────────────────
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from bs4 import BeautifulSoup
    from pypdf import PdfReader
except ImportError as e:
    print(f"❌ Missing packages: {e}")
    print("   Run:")
    print("   pip install langchain-chroma langchain-huggingface langchain-text-splitters langchain-core chromadb sentence-transformers pypdf bs4 requests")
    import sys; sys.exit(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "official_data"
PDF_DIR    = DATA_DIR / "pdf"
HTML_DIR   = DATA_DIR / "html"
CHROMA_DIR = pathlib.Path(os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db")))

PDF_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE    = 50

OFFICIAL_SOURCES = [
    {
        "url": "https://www.agedcarequality.gov.au/providers/quality-standards/strengthened-aged-care-quality-standards",
        "type": "html",
        "filename": "acqsc-strengthened-standards.html",
        "domain": "aged_care_quality",
    },
    {
        "url": "https://www.agedcarequality.gov.au/providers/quality-standards/guidance-and-resources",
        "type": "html",
        "filename": "acqsc-guidance-resources.html",
        "domain": "aged_care_quality",
    },
    {
        "url": "https://www.myagedcare.gov.au/aged-care-quality-standards",
        "type": "html",
        "filename": "myagedcare-standards.html",
        "domain": "aged_care_quality",
    },
    {
        "url": "https://huggingface.co/learn/cookbook/en/advanced_rag",
        "type": "html",
        "filename": "hf-advanced-rag.html",
        "domain": "rag_architecture",
    },
    {
        "url": "https://docs.mindset.ai/deploy/mcp/rag-mcpserver-integration",
        "type": "html",
        "filename": "mindset-rag-mcp.html",
        "domain": "rag_mcp_integration",
    },
]

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
def download_file(entry: dict) -> pathlib.Path:
    url       = entry["url"]
    file_type = entry["type"]
    filename  = entry["filename"]
    out_path  = (PDF_DIR if file_type == "pdf" else HTML_DIR) / filename

    if out_path.exists():
        print(f"  [SKIP] {filename} already downloaded")
        return out_path

    print(f"  [GET]  {url}")
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        print(f"  [SAVED] {filename}")
    except Exception as e:
        print(f"  ⚠️  Failed to download {url}: {e}")
        return None
    return out_path

# ── TEXT EXTRACTION ───────────────────────────────────────────────────────────
def extract_pdf(path: pathlib.Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"[PAGE {i+1}]\n{text}")
        return "\n\n".join(pages)
    except Exception as e:
        print(f"  ⚠️  PDF error {path.name}: {e}")
        return ""

def extract_html(path: pathlib.Path) -> str:
    try:
        raw  = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer"]):
            tag.decompose()
        lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"  ⚠️  HTML error {path.name}: {e}")
        return ""

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'═'*56}")
    print(f"  🌐 Health MCP RAG · Official Docs Ingestion")
    print(f"  🗄️  ChromaDB: {CHROMA_DIR}")
    print(f"  📌 Mode: APPEND (CV chunks preserved)")
    print(f"{'═'*56}\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    all_docs = []
    for entry in OFFICIAL_SOURCES:
        path = download_file(entry)
        if not path:
            continue

        text = extract_pdf(path) if entry["type"] == "pdf" else extract_html(path)
        if not text.strip():
            print(f"  ⚠️  Empty content: {entry['filename']}")
            continue

        chunks = splitter.create_documents(
            [text],
            metadatas=[{
                "source":   entry["url"],
                "filename": entry["filename"],
                "domain":   entry["domain"],
                "category": "official",
            }]
        )
        all_docs.extend(chunks)
        print(f"  ✅ {entry['filename']}: {len(chunks)} chunks [{entry['domain']}]")

    if not all_docs:
        print("\n  ❌ No content extracted.")
        return

    print(f"\n  📦 Total new chunks: {len(all_docs):,}")
    print(f"  🔄 Loading embedding model (all-MiniLM-L6-v2)…")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # APPEND to existing ChromaDB — do NOT wipe it
    print(f"  💾 Appending to existing ChromaDB index…")
    t0 = time.time()

    vs = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i:i+BATCH_SIZE]
        vs.add_documents(batch)
        pct = min(100, int((i + len(batch)) / len(all_docs) * 100))
        print(f"  ░{'█' * (pct//5)}{'░' * (20 - pct//5)}░ {pct}%", end="\r")

    elapsed = time.time() - t0
    total   = vs._collection.count()

    print(f"\n\n  ✅ Added {len(all_docs):,} official chunks in {elapsed:.1f}s")
    print(f"  📊 Total chunks in index: {total:,} (CVs + official docs)")
    print(f"  🚀 Restart server: python app/server.py\n")

if __name__ == "__main__":
    main()
