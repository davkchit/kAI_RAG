import os

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient, models

load_dotenv()

DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_NAME = "university_docs_odl"
CANDIDATE_LIMIT = 20
ROUTE_LIMIT = 3
CONTEXT_LIMIT = 3

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY)

client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    timeout=30,
    trust_env=False,
    check_compatibility=False,
)
client.set_model(DENSE_MODEL)
client.set_sparse_model(SPARSE_MODEL)

DENSE_VECTOR_NAME = next(iter(client.get_fastembed_vector_params().keys()))
SPARSE_VECTOR_NAME = next(iter(client.get_fastembed_sparse_vector_params().keys()))


def normalize_text(text: str):
    return text.lower().replace("ё", "е")


def contains_any(text: str, keywords):
    return any(keyword in text for keyword in keywords)


def build_filter(criteria):
    if not criteria:
        return None

    return models.Filter(
        must=[
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in criteria.items()
        ]
    )


def expand_query(question: str, expansion: str = ""):
    if not expansion:
        return question
    return f"{question}\n{expansion}"


def add_route(question: str, routes, seen, name: str, filters=None, expansion: str = "", bonus: float = 0.0):
    normalized_filters = tuple(sorted((filters or {}).items()))
    route_key = (name, normalized_filters)

    if route_key in seen:
        return

    seen.add(route_key)
    routes.append(
        {
            "name": name,
            "filters": filters,
            "search_text": expand_query(question=question, expansion=expansion),
            "bonus": bonus,
        }
    )


def build_route_plan(question: str):
    lowered = normalize_text(question)
    routes = []
    seen = set()

    branch_core_keywords = (
        "филиал",
        "нчф",
        "челн",
        "королев",
        "приемной комиссии филиала",
        "приемной директора филиала",
    )
    branch_leadership_keywords = (
        "директор",
        "ректор",
        "руковод",
    )
    branch_contact_keywords = (
        "телефон",
        "контакт",
        "адрес",
        "сайт",
        "email",
        "почт",
        "соцсет",
        "telegram",
        "vk",
        "приемной",
    )
    branch_office_keywords = (
        "учебно-методическ",
        "зорина",
        "справк",
        "академическ",
        "отпуск",
    )
    branch_admission_keywords = (
        "проходной балл",
        "бюджетных мест",
        "коммерческих мест",
        "бюджетный фонд",
        "коммерческий фонд",
        "средняя стоимость",
        "минимальный порог",
        "стоимость обучения",
    )
    branch_status_keywords = (
        "основан",
        "лиценз",
        "аккредитац",
        "рейтинг",
        "минтруд",
        "трудоустройств",
    )
    branch_partner_keywords = (
        "камаз",
        "генеральный партнер",
        "инженерная школа",
        "школы №30",
        "школа №30",
        "партнер",
    )
    branch_campus_keywords = (
        "общежит",
        "военный учебный центр",
    )
    bo_keywords = (
        "магистрат",
        "бакалавр",
        "специалитет",
        "егэ",
        "внутренн",
        "направлен",
        "организац",
        "особая квота",
        "отдельная квота",
        "бакалавриата",
    )
    asp_keywords = (
        "аспирант",
        "аспирантур",
        "научно-педагогических",
        "специальная дисциплина",
        "вступительн",
    )
    spo_keywords = (
        "спо",
        "среднего профессионального",
        "среднее профессиональное",
    )
    regulation_keywords = (
        "образовательных отношен",
        "отчисл",
        "перевод",
        "восстанов",
        "приостанов",
        "прекращени",
        "порядок",
    )

    is_branch_query = any(
        contains_any(lowered, keywords)
        for keywords in (
            branch_core_keywords,
            branch_leadership_keywords,
            branch_contact_keywords,
            branch_office_keywords,
            branch_admission_keywords,
            branch_status_keywords,
            branch_partner_keywords,
            branch_campus_keywords,
        )
    )

    if contains_any(lowered, regulation_keywords):
        add_route(
            question,
            routes,
            seen,
            name="regulations",
            filters={"doc_group": "regulations"},
            expansion="образовательные отношения перевод восстановление отчисление приказ книту-каи",
            bonus=0.15,
        )

    elif contains_any(lowered, spo_keywords):
        add_route(
            question,
            routes,
            seen,
            name="admission_spo",
            filters={"doc_group": "admission", "program_level": "spo"},
            expansion="правила приема спо среднее профессиональное образование книту-каи",
            bonus=0.15,
        )

    elif contains_any(lowered, asp_keywords) and "магистрат" not in lowered:
        add_route(
            question,
            routes,
            seen,
            name="admission_asp",
            filters={"doc_group": "admission", "program_level": "asp"},
            expansion="правила приема аспирантура вступительное испытание минимальное количество баллов книту-каи",
            bonus=0.15,
        )

    elif is_branch_query:
        branch_has_special_focus = False
        office_focus = contains_any(lowered, branch_office_keywords)

        if contains_any(lowered, branch_leadership_keywords):
            branch_has_special_focus = True
            add_route(
                question,
                routes,
                seen,
                name="branch_leadership",
                filters={"doc_scope": "branch"},
                expansion="высшее руководство ректор директор филиала",
                bonus=0.2,
            )

        if contains_any(lowered, branch_contact_keywords) and not office_focus:
            branch_has_special_focus = True
            add_route(
                question,
                routes,
                seen,
                name="branch_contacts",
                filters={"doc_scope": "branch"},
                expansion="контактная и справочная информация приемная комиссия приемная директора официальный сайт email telegram vk",
                bonus=0.2,
            )

        if office_focus:
            branch_has_special_focus = True
            add_route(
                question,
                routes,
                seen,
                name="branch_office",
                filters={"doc_scope": "branch"},
                expansion="учебно-методический отдел начальник отдела зорина ирина владимировна ivzorina 8(963)123-46-97",
                bonus=0.22,
            )

        if contains_any(lowered, branch_admission_keywords):
            branch_has_special_focus = True
            add_route(
                question,
                routes,
                seen,
                name="branch_admission",
                filters={"doc_scope": "branch"},
                expansion="контрольные цифры приема бюджетный фонд коммерческий фонд проходные баллы стоимость обучения минимальный порог 68 400 172 000",
                bonus=0.22,
            )

        if contains_any(lowered, branch_status_keywords):
            branch_has_special_focus = True
            add_route(
                question,
                routes,
                seen,
                name="branch_status",
                filters={"doc_scope": "branch"},
                expansion="основан 18 октября 2001 лицензия аккредитация 44-е место рейтинг минтруда",
                bonus=0.22,
            )

        if contains_any(lowered, branch_partner_keywords):
            branch_has_special_focus = True
            add_route(
                question,
                routes,
                seen,
                name="branch_partnership",
                filters={"doc_scope": "branch"},
                expansion="пао камаз генеральный партнер инженерная школа школа №30",
                bonus=0.22,
            )

        if contains_any(lowered, branch_campus_keywords):
            branch_has_special_focus = True
            add_route(
                question,
                routes,
                seen,
                name="branch_campus",
                filters={"doc_scope": "branch"},
                expansion="общежитие военный учебный центр",
                bonus=0.18,
            )

        if not branch_has_special_focus:
            add_route(
                question,
                routes,
                seen,
                name="branch_core",
                filters={"doc_scope": "branch"},
                expansion="набережночелнинский филиал книту-каи",
                bonus=0.12,
            )

    elif contains_any(lowered, bo_keywords):
        bo_expansion = "правила приема бакалавриат специалитет магистратура книту-каи"

        if "какое образование" in lowered and "магистрат" in lowered:
            bo_expansion += " для поступления в магистратуру требуется высшее образование"

        add_route(
            question,
            routes,
            seen,
            name="admission_bo",
            filters={"doc_group": "admission", "program_level": "bo"},
            expansion=bo_expansion,
            bonus=0.15,
        )

    elif contains_any(lowered, ("прием", "абитури", "зачисл", "документ")):
        add_route(
            question,
            routes,
            seen,
            name="admission_generic",
            filters={"doc_group": "admission"},
            expansion="правила приема книту-каи",
            bonus=0.08,
        )

    add_route(
        question,
        routes,
        seen,
        name="global_fallback",
        filters=None,
        expansion="книту-каи",
        bonus=0.0,
    )

    return routes

def build_hit_key(hit):
    if hit.id is not None:
        return hit.id

    payload = hit.payload or {}
    return (
        payload.get("source"),
        payload.get("page"),
        payload.get("chunk_index"),
    )


def run_hybrid_query(search_text: str, collection_name: str, limit: int = 5, route_filter=None):
    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=search_text, model=DENSE_MODEL),
                using=DENSE_VECTOR_NAME,
                filter=route_filter,
                limit=CANDIDATE_LIMIT,
            ),
            models.Prefetch(
                query=models.Document(text=search_text, model=SPARSE_MODEL),
                using=SPARSE_VECTOR_NAME,
                filter=route_filter,
                limit=CANDIDATE_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit,
    )
    return response.points


def hybrid_search(question: str, collection_name: str, limit: int = 5):
    routes = build_route_plan(question)
    scored_hits = {}

    for route_index, route in enumerate(routes):
        if route["name"] == "global_fallback" and len(scored_hits) >= CONTEXT_LIMIT:
            continue

        route_limit = limit if route["name"] == "global_fallback" else ROUTE_LIMIT
        hits = run_hybrid_query(
            search_text=route["search_text"],
            collection_name=collection_name,
            limit=max(route_limit, 1),
            route_filter=build_filter(route["filters"]),
        )

        for hit in hits:
            key = build_hit_key(hit)
            weighted_score = float(hit.score or 0.0) + route["bonus"]

            current = scored_hits.get(key)
            if current is None or weighted_score > current[0]:
                scored_hits[key] = (weighted_score, hit, route_index)

    ranked_hits = sorted(
        scored_hits.values(),
        key=lambda item: (item[0], -item[2]),
        reverse=True,
    )

    return [hit for _, hit, _ in ranked_hits[:limit]]


def ask_question(question: str):
    if question.strip() == "/start":
        return "Давай же начнем наше общение! Я всегда на связи, спрашивай 💙"

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

    context_parts = context_parts[:CONTEXT_LIMIT]
    context = "\n\n".join(context_parts)
    prompt = f"Контекст из документов:\n{context}\n\nВопрос студента: {question}"

    completion = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """ты — kAI, помощник нчф книту каи.
правила:

- если пользователь пишет /start — ответь: Давай же начнем наше общение! Я всегда на связи, спрашивай 💙
- отвечай только на основе context.
- если ответа в context нет — напиши: \"Извини, у меня нет информации по этому вопросу. Попробуй переформулировать.\"
- не начинай ответы с приветствий и не представляйся.
- не говори \"согласно документам\", \"в тексте указано\" и т.п.
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
        print(ask_question("привет"))
    except Exception as e:
        print(f"Произошла ошибка: {e}")


