import os

import httpx
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient, models

load_dotenv()

DENSE_VECTOR_NAME = "dense"
COLLECTION_NAME = "university_docs_odl"
CANDIDATE_LIMIT = 40
CONTEXT_LIMIT = 7

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
- если чего-то нет (общежитие, центр, услуга) — пиши "отсутствует", не "нет"
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

_CLASSIFIER_PROMPT = """You classify questions about НЧФ КНИТУ-КАИ university branch.
Reply with ONLY one of these exact codes (no other text):

branch
admission_bo
admission_asp
admission_spo
regulations
general

Rules:
- branch: anything about the branch itself — its address, director, rector, phone, email, social media, dormitory, military center, academic department (УМО), partners, KAMAZ, tuition price, budget seats, passing scores (проходные баллы), license number, accreditation number, ranking (рейтинг), founding year, building schedule
- admission_bo: anything about bachelor / specialist / master (магистратура) degree — whether it exists, EGE scores, entrance exams, enrollment documents, how to apply. NOTE: магистратура is NOT аспирантура — it is admission_bo
- admission_asp: аспирантура (PhD-level) admission only — NOT магистратура
- admission_spo: secondary vocational (СПО) admission
- regulations: student rules — expulsion, transfer, reinstatement, academic leave, suspension, termination of studies
- general: everything else

Output ONLY the code."""

_CHITCHAT_TRIGGERS = (
    "привет", "здравствуй", "добрый", "хай", "хелло", "ку ", "дарова", "приветствую",
    "пока", "до свидан", "спасибо", "благодар", "спс", "пасиб", "благодарю",
    "кто ты", "что ты умеешь", "ты бот", "ты умный", "что можешь", "помоги мне",
    "/start",
    "окей", "понятно", "хорошо", "ясно", "понял", "ок",
    "молодец", "класс", "круто", "супер", "отлично", "здорово",
)

_KNOWN_TOPICS = {"branch", "admission_bo", "admission_asp", "admission_spo", "regulations"}


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


def classify_query(question: str) -> str | None:
    """Returns topic string, or None for global search."""
    try:
        completion = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _CLASSIFIER_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=5,
        )
        topic = completion.choices[0].message.content.strip().lower().split()[0]
        return topic if topic in _KNOWN_TOPICS else None
    except Exception:
        return None  # fallback to global search on any error


def build_qdrant_filter(topic: str | None) -> models.Filter | None:
    if not topic:
        return None
    if topic == "branch":
        return models.Filter(must=[
            models.FieldCondition(key="doc_scope", match=models.MatchValue(value="branch"))
        ])
    if topic == "regulations":
        return models.Filter(must=[
            models.FieldCondition(key="doc_group", match=models.MatchValue(value="regulations"))
        ])
    # Admission: include both the specific admission PDF and the branch overview
    # (branch overview has passing scores, seat counts that complement admission rules)
    level = topic.split("_")[-1]  # bo / asp / spo
    return models.Filter(
        should=[
            models.Filter(must=[
                models.FieldCondition(key="doc_group", match=models.MatchValue(value="admission")),
                models.FieldCondition(key="program_level", match=models.MatchValue(value=level)),
            ]),
            models.Filter(must=[
                models.FieldCondition(key="doc_scope", match=models.MatchValue(value="branch")),
            ]),
        ],
    )


def is_chitchat(text: str) -> bool:
    if text.strip() == "/start":
        return True
    lowered = text.lower().strip()
    if len(lowered) > 60:
        return False
    return any(trigger in lowered for trigger in _CHITCHAT_TRIGGERS)


def search_candidates(question: str, collection_name: str, limit: int = CANDIDATE_LIMIT,
                      query_filter: models.Filter | None = None) -> list:
    dense_vec = get_dense_embedding(f"{question} НЧФ КНИТУ-КАИ набережные челны")
    response = client.query_points(
        collection_name=collection_name,
        query=dense_vec,
        using=DENSE_VECTOR_NAME,
        query_filter=query_filter,
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

    topic = classify_query(question)
    query_filter = build_qdrant_filter(topic)

    candidates = search_candidates(question, COLLECTION_NAME, query_filter=query_filter)
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
