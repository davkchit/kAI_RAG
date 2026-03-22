import fitz
from pathlib import Path
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_files = list(Path("data").glob("*.pdf"))
splitter = RecursiveCharacterTextSplitter(
chunk_size=600,     
chunk_overlap=120,    
separators=["\n\n", "\n", ". ", " "]
)
client = QdrantClient(
    url="http://localhost:6333",
    timeout=30,
    trust_env=False,
    check_compatibility=False,
)
DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "qdrant/bm25"
COLLECTION_NAME = "university_docsNEW"
client.set_model(DENSE_MODEL)
client.set_sparse_model(SPARSE_MODEL)

if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=client.get_fastembed_vector_params(),
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
        )
        print(f"Готово, '{COLLECTION_NAME}'  создана!")
else:
    print(f"уже есть '{COLLECTION_NAME}' ")

def extract_textANDtables_from_pdf(pdf):
    doc = fitz.open(pdf)
    pages = []

    for page_index, page in enumerate(doc, start=1):
        page_text = page.get_text("text").strip()
        if page_text:
            pages.append((page_index, page_text))
    doc.close()
    return pages



for pdf in pdf_files:
    documents = []
    metadatas = []

    pages = extract_textANDtables_from_pdf(pdf)
    print(f"Пдф файл прочитан {pdf.name}")

    for page_number, page_text in pages:
        chunks = splitter.split_text(page_text)

        for i, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            documents.append(chunk_text)
            metadatas.append(
                {
                    "source": pdf.name,
                    "page": page_number,
                    "chunk_index": i,
                    "parser": "fitz",
                }
            )

    if documents:
        print(f"Загружаю {len(documents)} чанков в Qdrant...")
        client.add(
            collection_name=COLLECTION_NAME,
            documents=documents,
            metadata=metadatas,
            batch_size=32,
        )
        print("Успех! Данные проиндексированы для будущего hybrid search.")
    else:
        print("Ошибка: не найдено документов для индексации.")
