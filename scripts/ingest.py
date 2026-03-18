import os
import fitz
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import uuid
from qdrant_client.models import PointStruct
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY)
model = SentenceTransformer("BAAI/bge-m3")
pdf_files = list(Path("data").glob("*.pdf"))
collection_name = "university_docs2"

client = QdrantClient(
    url="http://localhost:6333",
    timeout=30,
    trust_env=False,
    check_compatibility=False,
)

if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1024, 
                distance=Distance.COSINE
            ),
        )
        print(f"Готово, '{collection_name}'  создана!")
else:
    print(f"уже есть '{collection_name}' ")

def extract_textANDtables_from_pdf(pdf):
    doc = fitz.open(pdf)
    total_text = ""
    for page in doc:
        page_text = page.get_text("text")
        tabs = page.find_tables()
        md_tables = ""
        if tabs.tables:
            for table in tabs.tables:
                data = table.extract()
                if not data: continue
                table_str = "\n[ДАННЫЕ ИЗ ТАБЛИЦЫ]:\n"
                for i, row in enumerate(data):
                    clean_row = [str(c).replace("\n", " ") if c else "" for c in row]
                    table_str += "| " + " | ".join(clean_row) + " |\n"
                    if i == 0:
                        table_str += "| " + " | ".join(["---"] * len(row)) + " |\n"
                md_tables += table_str + "\n"
        total_text += f"{page_text}\n{md_tables}\n"

    doc.close()
    return total_text

all_docs = []

for pdf in pdf_files:
    text = extract_textANDtables_from_pdf(pdf)
    all_docs.append({
        "text": text,
        "source": pdf.name
    })
    print(f"Пдф файл прочитан {pdf.name}")

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,     
    chunk_overlap=120,    
    separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)

    points = []
    for chunk_text, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),      # Уникальный ID
                vector=vector.tolist(),    # Превращаем numpy-массив в обычный список
                payload={                  # Твои метаданные
                    "text": chunk_text, 
                    "source": pdf.name
                }
            )
        )
    
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Добавлено {len(points)} чанков из {pdf.name}")
    else:
        print(f"Пропуск {pdf.name}: не найдено текста для индексации.")
