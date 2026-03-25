FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    qdrant-client \
    fastembed \
    groq \
    python-dotenv

EXPOSE 8000
CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
