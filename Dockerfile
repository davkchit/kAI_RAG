FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    qdrant-client \
    "fastembed[bm25]" \
    groq \
    httpx \
    python-dotenv

# Pre-bake BM25 model into the image (~few MB, no neural network)
RUN python -c "from fastembed import SparseTextEmbedding; list(SparseTextEmbedding('Qdrant/bm25').embed(['warmup']))"

CMD ["sh", "-c", "uvicorn scripts.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
