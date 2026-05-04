import os

import httpx
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient

load_dotenv()

DENSE_DIM = 1024
DENSE_VECTOR_NAME = "dense"
COLLECTION_NAME = "university_docs_odl"
CANDIDATE_LIMIT = 20
CONTEXT_LIMIT = 5

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

client_groq = Groq(api_key=GROQ_API_KEY)

client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30,
    check_compatibility=False,
)

_SYSTEM_PROMPT = """ты — kAI, помощник нчф книту-каи. отвечаешь тепло и по-дружески.

правила:
- отвечай только на основе context
- если ответа в context нет — напиши: "Извини, у меня нет информации по этому вопросу. Лучше уточни в приёмной комиссии 💙"
- если вопрос про адреса, местонахождение или как добраться — перечисли ВСЕ адреса из контекста, не выбирай один
- не говори "согласно документам", "в тексте указано" и т.п.
- не придумывай факты
- в конце ответа на новой строке укажи источник: [src:<source>;p:<page>]
- если ответа в context нет — источник не пиши

стиль:
- тон лёгкий, живой, как помощь другу
- можно добавить "удачи!", "не переживай", "всё получится"
- без канцелярского тона

💙: ставь в конце примерно каждого второго ответа, не два раза подряд в одном
"""

_CHITCHAT_SYSTEM_PROMPT = """ты — kAI, дружелюбный помощник нчф книту-каи в набережных челнах.
отвечай тепло и живо. если пишут просто поздороваться или поблагодарить — ответь естественно.
если спрашивают что ты умеешь — скажи что помогаешь с вопросами об университете: приём, документы, контакты, расписание и т.д.
никогда не говори "у меня нет информации" в ответ на приветствие."""

_CHITCHAT_TRIGGERS = (
    "привет", "здравствуй", "добрый", "хай", "хелло", "ку ", "дарова", "приветствую",
    "пока", "до свидан", "спасибо", "благодар", "спс", "пасиб", "благодарю",
    "кто ты", "что ты умеешь", "ты бот", "ты умный", "что можешь", "помоги мне",
    "/start",
    "окей", "понятно", "хорошо", "ясно", "понял", "ок",
    "молодец", "класс", "круто", "супер", "отлично", "здорово",
)


def get_dense_embedding(text: str) -> list[float]:
    response = httpx.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        json={"model": "jina-embeddings-v3", "input": [text], "task": "retrieval.query"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def rerank(question: str, hits: list, top_n: int = CONTEXT_LIMIT) -> list:
    if not hits:
        return []
    documents = [hit.payload.get("document") or hit.payload.get("text", "") for hit in hits]
    response = httpx.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": question,
            "documents": documents,
            "top_n": top_n,
        },
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()["results"]
    return [hits[r["index"]] for r in results]


def is_chitchat(text: str) -> bool:
    if text.strip() == "/start":
        return True
    lowered = text.lower().strip()
    if len(lowered) > 60:
        return False
    return any(trigger in lowered for trigger in _CHITCHAT_TRIGGERS)


def search_candidates(question: str, collection_name: str, limit: int = CANDIDATE_LIMIT) -> list:
    search_text = f"{question} НЧФ КНИТУ-КАИ набережные челны"
    dense_vec = get_dense_embedding(search_text)
    response = client.query_points(
        collection_name=collection_name,
        query=dense_vec,
        using=DENSE_VECTOR_NAME,
        with_payload=True,
        limit=limit,
    )
    return response.points


def ask_question(question: str) -> str:
    if is_chitchat(question):
        completion = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _CHITCHAT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content

    candidates = search_candidates(question, COLLECTION_NAME)
    if not candidates:
        return "Извини, у меня нет информации по этому вопросу. Лучше уточни в приёмной комиссии 💙"

    hits = rerank(question, candidates)
    if not hits:
        return "Извини, у меня нет информации по этому вопросу. Лучше уточни в приёмной комиссии 💙"

    context_parts = []
    for hit in hits:
        source = hit.payload.get("source", "Неизвестный источник")
        page = hit.payload.get("page")
        content = hit.payload.get("document") or hit.payload.get("text", "")
        if not content:
            continue
        if page:
            context_parts.append(f"--- ИСТОЧНИК: {source}, стр. {page} ---\n{content}")
        else:
            context_parts.append(f"--- ИСТОЧНИК: {source} ---\n{content}")

    if not context_parts:
        return "Извини, у меня нет информации по этому вопросу. Лучше уточни в приёмной комиссии 💙"

    context = "\n\n".join(context_parts)
    prompt = f"Контекст из документов:\n{context}\n\nВопрос студента: {question}"

    completion = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    try:
        print(ask_question("привет"))
    except Exception as e:
        print(f"Произошла ошибка: {e}")
