FROM python:3.11-slim

WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app code
COPY --chown=user . $HOME/app

# Pre-download embedding model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Build chroma_db fresh on HF using its own chromadb version
RUN DATA_PATH=/app/Data python core/ingest.py

# HF Spaces requires port 7860
ENV PORT=7860

# Run the server
CMD ["python", "app/server.py"]