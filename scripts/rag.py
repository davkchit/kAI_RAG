import os

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient, models

load_dotenv()

DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "qdrant/bm25"
COLLECTION_NAME = "university_docsNEW"
CANDIDATE_LIMIT = 20

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY)

client = QdrantClient(
    url="http://localhost:6333",
    timeout=30,
    trust_env=False,
    check_compatibility=False,
)
client.set_model(DENSE_MODEL)
client.set_sparse_model(SPARSE_MODEL)

DENSE_VECTOR_NAME = next(iter(client.get_fastembed_vector_params().keys()))
SPARSE_VECTOR_NAME = next(iter(client.get_fastembed_sparse_vector_params().keys()))

def hybrid_search(question: str, collection_name: str, limit: int = 5):
    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=question, model=DENSE_MODEL),
                using=DENSE_VECTOR_NAME,
                limit=CANDIDATE_LIMIT,
            ),
            models.Prefetch(
                query=models.Document(text=question, model=SPARSE_MODEL),
                using=SPARSE_VECTOR_NAME,
                limit=CANDIDATE_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit,
    )
    return response.points


def ask_question(question: str):
    search_results = hybrid_search(question, COLLECTION_NAME)

    if not search_results:
        return "Извини, у меня нет информации по этому вопросу. Попробуй переформулировать."

    context_parts = []
    for hit in search_results:
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
        return "Извини, у меня нет информации по этому вопросу. Попробуй переформулировать."

    context = "\n\n".join(context_parts)
    prompt = f"Контекст из документов:\n{context}\n\nВопрос студента: {question}"

    completion = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """ты — kAI, помощник нчф книту каи.
правила:

- если пользователь пишет /start — ответь: Давай же начнем наше общение! Я всегда на связи, спрашивай 💙"
- отвечай только на основе context.
- если ответа в context нет — напиши: "Извини, у меня нет информации по этому вопросу. Попробуй переформулировать."
- не начинай ответы с приветствий и не представляйся.

- не говори "согласно документам", "в тексте указано" и т.п.

- не придумывай фактов.

- если ответ найден, в конце ответа на новой строке укажи только один источник в формате [src:<source>;p:<page>]
- используй только источник, который уже есть в context
- не придумывай источник
- если ответа в context нет, источник не пиши




стиль:

- отвечай коротко, ясно и дружелюбно.



предложение помощи:

- не предлагай помощь в каждом ответе.

- иногда можно добавить короткую фразу с предложением помощи (примерно в 1 из 4 ответов).

- если добавляешь предложение помощи — поставь 💙 в самом конце.

- если предложения помощи нет — 💙 использовать нельзя.
""",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    try:
        print(ask_question("меня звать биллиан ауто дес Аутоим. Desno, sir vi achete nah billie?"))
    except Exception as e:
        print(f"Произошла ошибка: {e}")
