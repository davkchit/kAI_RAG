# kAI RAG

Локальный RAG-проект для поиска ответов по PDF-документам КНИТУ-КАИ.

Проект индексирует документы через `OpenDataLoader PDF`, хранит чанки в `Qdrant`, использует hybrid retrieval (`dense + sparse + RRF`) и формирует финальный ответ через `Groq`.
Поверх RAG-ядра добавлен API-слой на `FastAPI` и рабочая интеграция с `n8n + Telegram`.

## Status

Текущий статус: сильный MVP.

Что уже есть:
- ingestion PDF -> JSON/Markdown -> chunks -> Qdrant
- hybrid retrieval на `intfloat/multilingual-e5-large` + `Qdrant/bm25`
- source citation в ответе
- простой routing по типу документа
- smoke eval для быстрой проверки качества
- API-эндпоинт `POST /ask` на `FastAPI`
- рабочая цепочка `Telegram -> n8n -> FastAPI -> Telegram`

Что еще не закончено:
- routing пока rule-based и довольно грубый
- admission-документы лучше разделить тоньше: `BO / АСП / СПО`
- нет UI
- API пока минимальный, без auth/rate-limit/наблюдаемости
- eval уже полезный, но еще не полноценный benchmark

## What It Does

- читает PDF из `data/`
- парсит документы через `OpenDataLoader PDF`
- собирает текст по страницам и индексирует его в `Qdrant`
- ищет релевантные чанки через hybrid search
- ограничивает контекст перед LLM
- возвращает короткий ответ с указанием источника
- отдает ответ через HTTP (`FastAPI`)
- может работать как backend для Telegram-бота через `n8n`

## Architecture

```text
PDF files
  -> OpenDataLoader PDF
  -> structured JSON / Markdown
  -> chunking
  -> Qdrant (dense + sparse)
  -> hybrid retrieval (RRF)
  -> lightweight routing by doc_group
  -> Groq / Llama 3.3 70B
  -> final answer with source citation
```

```text
Telegram user
  -> Telegram Trigger (n8n)
  -> HTTP Request to FastAPI (/ask)
  -> RAG pipeline (Qdrant + Groq)
  -> Telegram Send Message (n8n)
```

## Tech Stack

- Python
- OpenDataLoader PDF
- Qdrant
- `intfloat/multilingual-e5-large`
- `Qdrant/bm25`
- Groq `llama-3.3-70b-versatile`
- FastAPI + Uvicorn
- n8n
- Docker
- ngrok (для публичного webhook URL в Telegram)
- `langchain-text-splitters`

## Repository Layout

```text
kAI_RAG/
├─ data/
│  ├─ university.pdf
│  └─ ... local PDF files
├─ scripts/
│  ├─ ingest.py
│  ├─ rag.py
│  ├─ api.py
│  └─ eval_smoke.py
├─ Dockerfile
├─ .dockerignore
├─ eval_questions.jsonl
├─ .env.example
├─ .gitignore
└─ README.md
```

## Core Components

### `scripts/ingest.py`

Индексация базы знаний.

Скрипт:
- читает все PDF из `data/`
- конвертирует их через `OpenDataLoader PDF`
- вытаскивает текст из JSON-структуры
- режет документы на чанки
- пишет чанки в `Qdrant`
- добавляет metadata:
  - `source`
  - `page`
  - `chunk_index`
  - `parser`
  - `doc_group`

### `scripts/rag.py`

Поиск и генерация ответа.

Скрипт:
- выполняет hybrid retrieval
- использует простой routing по `doc_group`
- ограничивает контекст до `top-3` чанков
- отправляет контекст в Groq
- возвращает ответ в формате с citation: `[src:<source>;p:<page>]`

### `scripts/api.py`

Минимальный API-слой.

Скрипт:
- поднимает `FastAPI` приложение
- принимает `POST /ask` с полем `question`
- вызывает `ask_question(...)` из `scripts/rag.py`
- возвращает JSON-ответ вида `{"answer": "..."}`

### `scripts/eval_smoke.py`

Быстрая локальная проверка качества.

Скрипт прогоняет набор вопросов из `eval_questions.jsonl` и считает отдельно:
- retrieval accuracy
- answer accuracy

## Requirements

- Python 3.10+
- Java 11+
- Qdrant
- `GROQ_API_KEY`
- Docker (если запускать сервисы контейнерами)
- ngrok (если использовать Telegram Trigger в self-hosted n8n)

## Quick Start

### 1. Install dependencies

```powershell
pip install -U opendataloader-pdf qdrant-client groq python-dotenv langchain-text-splitters fastapi "uvicorn[standard]" fastembed
```

### 2. Check Java

`OpenDataLoader PDF` требует `Java 11+`.

```powershell
java -version
```

### 3. Start Qdrant

Самый простой локальный вариант:

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 4. Configure environment

Минимально нужен `.env` с такими ключами:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=http://localhost:6333
```

Примечание: `GEMINI_API_KEY` в `.env.example` сейчас не используется в основном пайплайне.

### 5. Put documents into `data/`

Скрипт индексации автоматически обработает все `*.pdf` в папке `data/`.

### 6. Run ingestion

```powershell
python scripts/ingest.py
```

### 7. Run RAG locally

```powershell
python scripts/rag.py
```

### 8. Run API locally

```powershell
uvicorn scripts.api:app --host 0.0.0.0 --port 8000
```

Проверка:

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/ask" -ContentType "application/json" -Body '{"question":"/start"}'
```

### 9. Run smoke eval

```powershell
python scripts/eval_smoke.py
```

## How to Add Documents

Как добавить новые документы в базу знаний:

1. Положить новые `*.pdf` в папку `data/`.
2. Запустить индексацию:

```powershell
python scripts/ingest.py
```

Важно:
- в `scripts/ingest.py` по умолчанию стоит `RECREATE_COLLECTION = True`
- это значит, что коллекция пересоздается и собирается заново из всех PDF в `data/`
- если нужен инкрементальный режим без удаления коллекции, временно поставь `RECREATE_COLLECTION = False`

После успешной индексации перезапуск `n8n` и `FastAPI` обычно не нужен: новые данные уже лежат в `Qdrant`.

## How to Add Eval Cases

Новые проверки добавляются строками в `eval_questions.jsonl` (одна строка = один JSON-объект):

```json
{"question":"Какой адрес филиала?","must_contain":["423814","Королева"],"expected_source":"university.pdf"}
```

Поля:
- `question`: вопрос для теста retrieval + answer
- `must_contain`: список подстрок, которые должны встретиться в ответе
- `expected_source`: ожидаемый источник top-1 retrieval

После добавления кейсов запусти:

```powershell
python scripts/eval_smoke.py
```

## Docker Compose (Recommended)

`docker-compose.yml` already includes `qdrant + fastapi + n8n`.

```powershell
docker compose up -d --build
```

If you already have old standalone containers on the same ports (`qdrant`, `n8n`, `kai-fastapi`), stop them first:

```powershell
docker stop qdrant n8n kai-fastapi
```

Stop all services:

```powershell
docker compose down
```

For Telegram webhook in self-hosted mode set `WEBHOOK_URL` in `.env`, for example:

```env
WEBHOOK_URL=https://your-ngrok-domain.ngrok-free.app/
```

## Docker FastAPI (Optional)

Сборка и запуск API в Docker:

```powershell
docker build -t kai-fastapi .
docker network create kai-net
docker network connect kai-net qdrant
docker run -d --name kai-fastapi --network kai-net -p 8000:8000 --env-file .env -e QDRANT_URL=http://qdrant:6333 kai-fastapi
```

Примечание: если сеть `kai-net` уже создана, команда `docker network create kai-net` вернет ошибку `already exists` — это нормально.

## n8n + Telegram Integration

Минимальный рабочий workflow:

1. `Telegram Trigger` (получение входящего сообщения)
2. `HTTP Request`:
   - Method: `POST`
   - URL: `http://host.docker.internal:8000/ask` (если FastAPI на хосте)
   - URL: `http://fastapi:8000/ask` (если FastAPI в Docker Compose)
   - Body: `{"question":"={{$json.message.text}}"}`
3. `Telegram -> Send Message`:
   - Chat ID: `{{$node["Telegram Trigger"].json.message.chat.id}}`
   - Text: `{{$node["HTTP Request"].json.answer}}`

Важные моменты:
- для Telegram webhook нужен публичный `HTTPS` URL
- в self-hosted схеме `ngrok` обычно поднимается на `n8n` (порт `5678`)
- `Telegram Trigger` не может одновременно слушать `test` и `production`

## Retrieval Strategy

Сейчас в проекте используется такой порядок:

1. Вопрос пользователя грубо классифицируется по группе документов.
2. Qdrant ищет кандидатов через dense + sparse retrieval.
3. Результаты объединяются через `RRF`.
4. В контекст LLM попадает только `top-3` чанка.
5. Модель отвечает только на основе найденного контекста.

Текущие группы документов:
- `branch`
- `admission`
- `regulations`

Для небольшого корпуса этого достаточно. При росте корпуса routing лучше дробить дальше, например на `admission_bo`, `admission_asp`, `admission_spo`.

## Evaluation

Проект уже содержит локальный smoke-eval.

На текущем наборе из 30 вопросов локально получается примерно:
- retrieval: `27/30`
- answers: `21/30`

Это не финальный benchmark, а быстрый рабочий индикатор, который помогает ловить регрессии после изменений в ingestion, retrieval и prompt.

## Known Limitations

- routing пока простой и основан на ключевых словах
- admission-вопросы иногда смешиваются между разными PDF
- answer quality зависит не только от retrieval, но и от лимитов/поведения Groq
- `client.add(...)` в Qdrant уже deprecated, но в текущем MVP еще работает
- в логах может появляться warning от `fastembed` про pooling у `multilingual-e5-large`
- при первом старте контейнера с FastAPI возможна задержка из-за загрузки embedding-модели

## Data and Git

- в репозитории хранится только `data/university.pdf` как открытый пример
- остальные PDF предполагаются локальными рабочими документами
- артефакты `odl_output/` и `odl_probe/` не должны коммититься

## Next Steps

Самые полезные следующие улучшения:
- разбить routing на более точные document groups
- добавить retrieval-only eval и более крупный eval set
- ввести score threshold и более строгий режим `не знаю`
- добавить `docker-compose` для `qdrant + n8n + fastapi`
- добавить UI и/или более богатый Telegram UX (кнопки, команды, fallback-ответы)
- усилить API-слой: auth, rate limit, логирование, healthchecks

## Summary

Если коротко: это локальный hybrid RAG по PDF-документам КНИТУ-КАИ с OpenDataLoader, Qdrant, Groq, FastAPI и рабочей Telegram-интеграцией через n8n.
