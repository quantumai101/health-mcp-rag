FROM python:3.11-slim

WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app code
COPY --chown=user . $HOME/app

# Build chroma_db fresh on HF using its own chromadb version
# This eliminates any version mismatch from local builds
RUN python core/ingest.py

# HF Spaces requires port 7860
# server.py reads PORT env var automatically
ENV PORT=7860

# Run the server
CMD ["python", "app/server.py"]
