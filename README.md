# kAI RAG

Локальный RAG-проект для поиска ответов по PDF-документам КНИТУ-КАИ.

Проект индексирует документы через `OpenDataLoader PDF`, хранит чанки в `Qdrant`, использует hybrid retrieval (`dense + sparse + RRF`) и формирует финальный ответ через `Groq`.

## Status

Текущий статус: сильный MVP.

Что уже есть:
- ingestion PDF -> JSON/Markdown -> chunks -> Qdrant
- hybrid retrieval на `intfloat/multilingual-e5-large` + `Qdrant/bm25`
- source citation в ответе
- простой routing по типу документа
- smoke eval для быстрой проверки качества

Что еще не закончено:
- routing пока rule-based и довольно грубый
- admission-документы лучше разделить тоньше: `BO / АСП / СПО`
- нет API-слоя и UI
- eval уже полезный, но еще не полноценный benchmark

## What It Does

- читает PDF из `data/`
- парсит документы через `OpenDataLoader PDF`
- собирает текст по страницам и индексирует его в `Qdrant`
- ищет релевантные чанки через hybrid search
- ограничивает контекст перед LLM
- возвращает короткий ответ с указанием источника

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

## Tech Stack

- Python
- OpenDataLoader PDF
- Qdrant
- `intfloat/multilingual-e5-large`
- `Qdrant/bm25`
- Groq `llama-3.3-70b-versatile`
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
│  └─ eval_smoke.py
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

## Quick Start

### 1. Install dependencies

```powershell
pip install -U opendataloader-pdf qdrant-client groq python-dotenv langchain-text-splitters
```

### 2. Check Java

`OpenDataLoader PDF` требует `Java 11+`.

```powershell
java -version
```

### 3. Start Qdrant

Самый простой локальный вариант:

```powershell
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 4. Configure environment

Минимально нужен `.env` с таким ключом:

```env
GROQ_API_KEY=your_groq_api_key
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

### 8. Run smoke eval

```powershell
python scripts/eval_smoke.py
```

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

## Data and Git

- в репозитории хранится только `data/university.pdf` как открытый пример
- остальные PDF предполагаются локальными рабочими документами
- артефакты `odl_output/` и `odl_probe/` не должны коммититься

## Next Steps

Самые полезные следующие улучшения:
- разбить routing на более точные document groups
- добавить retrieval-only eval и более крупный eval set
- ввести score threshold и более строгий режим `не знаю`
- вынести проект в API-слой
- добавить UI или Telegram-бота

## Summary

Если коротко: это локальный hybrid RAG по PDF-документам КНИТУ-КАИ с OpenDataLoader, Qdrant, Groq и базовым quality loop через eval.
