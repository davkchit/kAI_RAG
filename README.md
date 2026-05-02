# kAI — Студенческий ассистент НЧФ КНИТУ-КАИ

RAG-чатбот для ответов на вопросы о Набережночелнинском филиале КНИТУ-КАИ: правила приёма, контакты, стоимость обучения, регламенты и многое другое.

## Как это работает

Гибридный поиск (dense Jina embeddings + BM25) по базе университетских документов с маршрутизацией по намерению запроса. Ответы генерирует Groq (llama-3.3-70b-versatile). Векторная БД — Qdrant Cloud.

```
Вопрос → роутинг по ключевым словам → гибридный поиск в Qdrant → контекст → Groq LLM → ответ
```

## Стек

| Компонент | Технология |
|-----------|-----------|
| API | FastAPI |
| Эмбеддинги | Jina AI `jina-embeddings-v3` |
| Sparse search | BM25 (fastembed) |
| Векторная БД | Qdrant Cloud |
| LLM | Groq `llama-3.3-70b-versatile` |
| Деплой | Railway |

## Быстрый старт

### 1. Переменные окружения

Скопируй `.env.example` в `.env` и заполни:

```env
GROQ_API_KEY=       # console.groq.com
JINA_API_KEY=       # jina.ai
QDRANT_URL=         # Qdrant Cloud cluster URL (с портом :6333)
QDRANT_API_KEY=     # Qdrant Cloud API key
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Индексация документов

Положи PDF-файлы в папку `data/` и запусти:

```bash
python -X utf8 -m scripts.ingest
```

### 4. Запуск API

```bash
uvicorn scripts.api:app --host 0.0.0.0 --port 8000
```

Или через Docker:

```bash
docker compose up
```

## API

### `POST /ask`

```json
{
  "question": "Какой адрес филиала?"
}
```

```json
{
  "answer": "423814, РТ, г. Набережные Челны, ул. Академика Королева, д. 1.\n[src:university.pdf;p:4]"
}
```

## Деплой на Railway

1. Создай кластер на [cloud.qdrant.io](https://cloud.qdrant.io)
2. Прогони `ingest` локально, указав Qdrant Cloud в `.env`
3. Задеплой репозиторий на Railway
4. В Railway Variables выставь: `GROQ_API_KEY`, `JINA_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`

Railway автоматически подхватит `Dockerfile` и динамический `$PORT`.

## Качество (eval)

| Метрика | Результат |
|---------|-----------|
| Retrieval (топ-1 правильный документ) | 30/30 (100%) |
| Answer (ответ содержит ключевые факты) | 25/30 (83%) |

```bash
cd scripts && python -X utf8 eval_smoke.py
```

## Структура проекта

```
kAI/
├── data/               # PDF-документы для индексации
├── scripts/
│   ├── api.py          # FastAPI endpoint
│   ├── rag.py          # RAG-логика, роутинг, поиск
│   ├── ingest.py       # Индексация PDF в Qdrant
│   └── eval_smoke.py   # Smoke-тесты качества
├── eval_questions.jsonl
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
