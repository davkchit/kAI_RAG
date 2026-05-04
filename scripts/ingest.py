import json
import os
import shutil
from pathlib import Path

import httpx
import opendataloader_pdf
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

load_dotenv()

DENSE_DIM = 1024  # jina-embeddings-v3
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
COLLECTION_NAME = "university_docs_odl"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("odl_output")
RECREATE_COLLECTION = True
UPLOAD_BATCH_SIZE = 64
FILTERABLE_FIELDS = (
    "source",
    "doc_group",
    "doc_type",
    "doc_scope",
    "program_level",
)

JINA_API_KEY = os.getenv("JINA_API_KEY")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=60,
    separators=["\n\n", "\n", ". ", " "],
)

sparse_model = SparseTextEmbedding("Qdrant/bm25")

client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30,
    check_compatibility=False,
)


def get_jina_embeddings(texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        json={"model": "jina-embeddings-v3", "input": texts, "task": "retrieval.passage"},
        timeout=120,
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


def setup_collection():
    if RECREATE_COLLECTION and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Старая коллекция '{COLLECTION_NAME}' удалена.")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=DENSE_DIM,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams()
            },
        )
        print(f"Коллекция '{COLLECTION_NAME}' создана.")
    else:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует.")

    ensure_payload_indexes()


def ensure_payload_indexes():
    for field_name in FILTERABLE_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def build_preview_text(pages, page_limit: int = 2, max_chars: int = 2500):
    preview_parts = []
    for _, page_text in pages[:page_limit]:
        compact = " ".join(page_text.split()).strip()
        if compact:
            preview_parts.append(compact)
    return " ".join(preview_parts)[:max_chars].lower()


def infer_document_profile(pdf_name: str, preview_text: str):
    lowered_name = pdf_name.lower()
    combined = f"{lowered_name}\n{preview_text}"

    profile = {
        "doc_group": "other",
        "doc_type": "reference",
        "doc_scope": "university",
        "program_level": "generic",
        "doc_title": pdf_name,
    }

    if any(
        keyword in combined
        for keyword in (
            "набережночелнинский филиал",
            "нчф",
            "филиал книту-каи",
            "university.pdf",
        )
    ):
        profile.update(
            {
                "doc_group": "branch",
                "doc_type": "overview",
                "doc_scope": "branch",
                "program_level": "branch",
            }
        )
        return profile

    if "правила приема" in combined:
        profile.update(
            {
                "doc_group": "admission",
                "doc_type": "rules",
                "doc_scope": "university",
            }
        )
        if "правила приема асп" in lowered_name:
            profile["program_level"] = "asp"
        elif "правила приема спо" in lowered_name:
            profile["program_level"] = "spo"
        elif "правила приема bo" in lowered_name:
            profile["program_level"] = "bo"
        elif any(keyword in combined for keyword in ("аспирантур", "аспирант", "научно-педагогических кадров")):
            profile["program_level"] = "asp"
        elif any(keyword in combined for keyword in ("среднего профессионального", "правила приема спо", "правила приема spo", "спо")):
            profile["program_level"] = "spo"
        elif any(keyword in combined for keyword in ("бакалавриат", "специалитет", "магистратур", "правила приема bo", "правила приема во", " bo")):
            profile["program_level"] = "bo"
        return profile

    if any(
        keyword in combined
        for keyword in (
            "образовательных отношений",
            "приостанов",
            "прекращени",
            "отчисл",
            "перевод",
            "восстанов",
            "порядок оформления",
        )
    ):
        profile.update(
            {
                "doc_group": "regulations",
                "doc_type": "regulation",
                "doc_scope": "university",
            }
        )
        return profile

    return profile


def table_to_text(node):
    rows = node.get("rows", [])
    if not rows:
        return str(node.get("content", "")).strip()

    row_texts = []
    for row in rows:
        cell_values = []
        for cell in row.get("cells", []):
            parts = []
            for kid in cell.get("kids", []):
                if isinstance(kid, dict):
                    text = str(kid.get("content", "")).strip()
                    if text:
                        parts.append(text)
            cell_text = " ".join(parts).strip()
            cell_values.append(cell_text)
        if any(cell_values):
            row_texts.append(" | ".join(cell_values))

    return "\n".join(row_texts).strip()


def walk_elements(node):
    if isinstance(node, list):
        for item in node:
            yield from walk_elements(item)
        return

    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    page_number = node.get("page number")

    if node_type in {"heading", "paragraph", "caption", "list item"}:
        content = str(node.get("content", "")).strip()
        if page_number and content:
            yield page_number, content

    if node_type == "table":
        table_text = table_to_text(node)
        if page_number and table_text:
            yield page_number, table_text

    yield from walk_elements(node.get("kids", []))
    yield from walk_elements(node.get("list items", []))


def extract_pages_from_json(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    pages = {}

    for page_number, text in walk_elements(data.get("kids", [])):
        pages.setdefault(page_number, []).append(text)

    result = []
    for page_number in sorted(pages):
        page_text = "\n\n".join(pages[page_number]).strip()
        if page_text:
            result.append((page_number, page_text))

    return result


def load_faq(faq_path: Path):
    """Load FAQ entries from data/faq.md — each ## section is one entry."""
    if not faq_path.exists():
        return [], [], []

    text = faq_path.read_text(encoding="utf-8")
    sections = text.split("\n## ")

    embed_texts = []
    display_texts = []
    metadatas = []

    for section in sections:
        section = section.strip().lstrip("# ").strip()
        if not section:
            continue

        lines = section.split("\n", 1)
        question = lines[0].strip()
        answer = lines[1].strip() if len(lines) > 1 else ""

        if not answer:
            continue

        embed_text = f"{question}\n{answer}"
        embed_texts.append(embed_text)
        display_texts.append(answer)
        metadatas.append({
            "source": "faq.md",
            "page": None,
            "chunk_index": 0,
            "parser": "faq",
            "doc_group": "faq",
            "doc_type": "faq",
            "doc_scope": "university",
            "program_level": "generic",
            "doc_title": question,
        })

    print(f"FAQ: загружено {len(embed_texts)} записей из {faq_path.name}")
    return embed_texts, display_texts, metadatas


def upload_batches(embed_texts: list[str], display_texts: list[str], metadatas: list[dict]):
    """embed_texts — contextualized text for embedding; display_texts — original chunk for LLM context."""
    total = len(embed_texts)
    point_id = 0

    for start in range(0, total, UPLOAD_BATCH_SIZE):
        end = min(start + UPLOAD_BATCH_SIZE, total)
        batch_embed = embed_texts[start:end]
        batch_display = display_texts[start:end]
        batch_meta = metadatas[start:end]

        print(f"Получаю эмбеддинги для чанков {start + 1}–{end} из {total}...")

        dense_vecs = get_jina_embeddings(batch_embed)
        sparse_vecs = list(sparse_model.embed(batch_embed))

        points = []
        for i, (display_doc, meta, d_vec, s_vec) in enumerate(
            zip(batch_display, batch_meta, dense_vecs, sparse_vecs)
        ):
            points.append(
                models.PointStruct(
                    id=point_id + i,
                    vector={
                        DENSE_VECTOR_NAME: d_vec,
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=s_vec.indices.tolist(),
                            values=s_vec.values.tolist(),
                        ),
                    },
                    payload={"document": display_doc, **meta},
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        point_id += len(batch_display)
        print(f"  Загружено.")


def main():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("В папке data нет PDF-файлов.")
        return

    setup_collection()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Dfile.encoding=UTF-8")

    opendataloader_pdf.convert(
        input_path=[str(pdf) for pdf in pdf_files],
        output_dir=str(OUTPUT_DIR),
        format="json,markdown",
        use_struct_tree=True,
    )

    all_embed_texts = []
    all_display_texts = []
    all_metadatas = []

    for pdf in pdf_files:
        json_path = OUTPUT_DIR / f"{pdf.stem}.json"

        if not json_path.exists():
            print(f"Не найден JSON-результат для {pdf.name}")
            continue

        pages = extract_pages_from_json(json_path)
        preview_text = build_preview_text(pages)
        doc_profile = infer_document_profile(pdf.name, preview_text)
        print(f"PDF обработан OpenDataLoader: {pdf.name}")

        for page_number, page_text in pages:
            chunks = splitter.split_text(page_text)

            for i, chunk_text in enumerate(chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                # Contextual chunking: prefix with document title and page for richer embedding
                embed_text = f"[{doc_profile['doc_title']}, стр. {page_number}]\n{chunk_text}"

                all_embed_texts.append(embed_text)
                all_display_texts.append(chunk_text)
                all_metadatas.append(
                    {
                        "source": pdf.name,
                        "page": page_number,
                        "chunk_index": i,
                        "parser": "opendataloader",
                        **doc_profile,
                    }
                )

    # Load FAQ entries if present
    faq_embed, faq_display, faq_meta = load_faq(DATA_DIR / "faq.md")
    all_embed_texts.extend(faq_embed)
    all_display_texts.extend(faq_display)
    all_metadatas.extend(faq_meta)

    if not all_embed_texts:
        print("Не найдено документов для индексации.")
        return

    print(f"Подготовлено {len(all_embed_texts)} чанков для индексации.")
    upload_batches(all_embed_texts, all_display_texts, all_metadatas)
    print("Готово! Данные проиндексированы.")


if __name__ == "__main__":
    main()
