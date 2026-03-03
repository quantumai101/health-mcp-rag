#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   HEALTH-MCP-RAG · DOCUMENT INGESTION                       ║
║   Indexes your documents into ChromaDB for RAG retrieval    ║
╚══════════════════════════════════════════════════════════════╝

Supported input types:
  - PDF documents (resumes, health publications, guidelines)
  - TXT files (transcripts, notes, achievements)
  - MD  files (markdown docs, project notes)
  - CSV files (structured data)

USAGE:
    python core/ingest.py

Put your files in the ./data/ folder before running.
Sub-folders are supported:
    data/
    ├── resumes/          ← your CVs and cover letters
    ├── health_docs/      ← health guidelines, aged care publications
    ├── ai_publications/  ← AI/ML papers, technical docs
    ├── transcripts/      ← YouTube video transcripts
    └── notes/            ← personal notes, project achievements
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.getenv("DATA_PATH",   "./data"))
CHROMA_DIR = Path(os.getenv("CHROMA_PATH", "./chroma_db"))

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120
BATCH_SIZE    = 50

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}

# ── IMPORTS ──────────────────────────────────────────────────────────────────
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError as e:
    print(f"❌ Missing packages: {e}")
    print("   Run:")
    print("   pip install langchain-chroma langchain-huggingface langchain-text-splitters langchain-core chromadb sentence-transformers pdfplumber")
    sys.exit(1)


def load_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  ⚠️  Could not read {path.name}: {e}")
        return ""


def load_pdf(path: Path) -> str:
    try:
        import pdfplumber
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return "\n\n".join(text)
    except ImportError:
        print(f"  ⚠️  pdfplumber not installed — skipping {path.name}")
        print(f"       Run: pip install pdfplumber")
        return ""
    except Exception as e:
        print(f"  ⚠️  PDF error {path.name}: {e}")
        return ""


def load_csv(path: Path) -> str:
    try:
        import csv
        rows = []
        with open(path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(", ".join(row))
        return "\n".join(rows)
    except Exception as e:
        print(f"  ⚠️  CSV error {path.name}: {e}")
        return ""


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".csv":
        return load_csv(path)
    else:
        return load_txt(path)


def get_category(path: Path) -> str:
    """Infer document category from folder name for metadata tagging."""
    parts = path.parts
    for part in parts:
        pl = part.lower()
        if "resume"     in pl or "cv"          in pl: return "resume"
        if "health"     in pl or "aged"        in pl: return "health"
        if "ai"         in pl or "publication" in pl: return "ai_technical"
        if "transcript" in pl or "youtube"     in pl: return "transcript"
        if "note"       in pl or "project"     in pl: return "notes"
    return "general"


def main():
    print(f"\n{'═'*56}")
    print(f"  🧬 Health MCP RAG · Document Ingestion")
    print(f"  📁 Source:  {DATA_DIR.resolve()}")
    print(f"  🗄️  Output:  {CHROMA_DIR.resolve()}")
    print(f"{'═'*56}\n")

    # Ensure data dir exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Find all supported files
    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(DATA_DIR.rglob(f"*{ext}"))

    if not all_files:
        print(f"  ⚠️  No documents found in {DATA_DIR}")
        print(f"  💡 Add your files to the data/ folder:")
        print(f"       data/resumes/         ← CVs, cover letters")
        print(f"       data/health_docs/     ← health publications")
        print(f"       data/ai_publications/ ← AI/ML papers")
        print(f"       data/transcripts/     ← YouTube transcripts")
        print(f"       data/notes/           ← personal notes")
        return

    print(f"  📄 Found {len(all_files)} documents\n")

    # Load and chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_docs = []
    for i, file_path in enumerate(all_files, 1):
        rel = file_path.relative_to(DATA_DIR)
        print(f"  [{i:>3}/{len(all_files)}] {rel}", end=" ")

        text = load_document(file_path)
        if not text.strip():
            print("→ empty, skipped")
            continue

        category = get_category(file_path)
        chunks = splitter.create_documents(
            [text],
            metadatas=[{
                "source":   str(file_path),
                "filename": file_path.name,
                "category": category,
                "folder":   file_path.parent.name,
            }]
        )
        all_docs.extend(chunks)
        print(f"→ {len(chunks)} chunks [{category}]")

    if not all_docs:
        print("\n  ❌ No content extracted from any documents.")
        return

    print(f"\n  📦 Total chunks to index: {len(all_docs):,}")
    print(f"  🔄 Loading embedding model (all-MiniLM-L6-v2)…")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Wipe old index and rebuild
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(f"  🗑️  Cleared old index")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  💾 Indexing in batches of {BATCH_SIZE}…")
    t0 = time.time()

    vs = None
    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i:i+BATCH_SIZE]
        if vs is None:
            vs = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_DIR)
            )
        else:
            vs.add_documents(batch)
        pct = min(100, int((i + len(batch)) / len(all_docs) * 100))
        print(f"  ░{'█' * (pct//5)}{'░' * (20 - pct//5)}░ {pct}%", end="\r")

    elapsed = time.time() - t0
    final_count = vs._collection.count() if vs else 0

    print(f"\n\n  ✅ Indexed {final_count:,} chunks in {elapsed:.1f}s")
    print(f"  🚀 Start server: python app/server.py")
    print(f"  🌐 Open:         http://localhost:8000\n")

    # Summary by category
    from collections import Counter
    cats = Counter(d.metadata.get("category", "general") for d in all_docs)
    print("  📊 By category:")
    for cat, count in cats.most_common():
        print(f"       {cat:<20} {count:>5} chunks")
    print()


if __name__ == "__main__":
    main()
