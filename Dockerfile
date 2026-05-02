FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    qdrant-client \
    groq \
    httpx \
    python-dotenv

COPY . /app

CMD ["sh", "-c", "uvicorn scripts.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
