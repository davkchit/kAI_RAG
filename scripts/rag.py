import os
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()


model = SentenceTransformer("BAAI/bge-m3")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY)

qdrant = QdrantClient(

    url="http://localhost:6333",

    timeout=30,

    trust_env=False,

    check_compatibility=False,

)
collection_name = "university_docs2"

def ask_question(question: str):
    query_vector = model.encode(question)
    response = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5
    )

    search_results = response.points
    # --- ШАГ 3: Собираем контекст из найденных результатов ---
    if not search_results:
        return "В базе данных нет информации по этому вопросу. 💙"

    context_parts = []
    for hit in search_results:
        # Достаем текст, который мы сохраняли в payload при загрузке
        text = hit.payload.get("text", "[Текст не найден]")
        context_parts.append(text)
    
    context = "\n\n".join(context_parts)

    # --- ШАГ 4: Формируем запрос для Llama через Groq ---
    prompt = f"Контекст из документов:\n{context}\n\nВопрос студента: {question}"

    response = client_groq.chat.completions.create(
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



стиль:

- отвечай коротко, ясно и дружелюбно.



предложение помощи:

- не предлагай помощь в каждом ответе.

- иногда можно добавить короткую фразу с предложением помощи (примерно в 1 из 4 ответов).

- если добавляешь предложение помощи — поставь 💙 в самом конце.

- если предложения помощи нет — 💙 использовать нельзя."""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# Тестовый запуск
try:
    print(ask_question("есть ли у вас программа 'СТАРТ'"))
except Exception as e:
    print(f"Произошла ошибка: {e}")