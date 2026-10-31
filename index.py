from pathlib import Path

import chromadb
from ollama import embed


# ======================================
# CONFIGURATION
# ======================================

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
DB_PATH = ROOT / "chroma_db"

EMBED_MODEL = "embeddinggemma:300m"
COLLECTION_NAME = "cuda_knowledge_v02"


# ======================================
# CHROMA DATABASE
# ======================================

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)


# ======================================
# LOAD AND CHUNK DOCUMENTS
# ======================================

def load_documents():
    chunks = []

    for file in sorted(DOCS_DIR.glob("*.txt")):

        text = file.read_text(encoding="utf-8")

        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        for index, paragraph in enumerate(paragraphs, 1):

            chunks.append({
                "id": f"{file.name}:{index:04d}",
                "source": file.name,
                "chunk": index,
                "text": paragraph
            })

    return chunks


# ======================================
# INDEX DOCUMENTS
# ======================================

def index_documents():

    print("\nLoading documents...")

    documents = load_documents()

    if not documents:
        print("No documents found!")
        return

    print(f"Found {len(documents)} chunks.")

    # Đọc những tài liệu đã lưu trong DB
    existing = collection.get(
        include=["documents"]
    )

    existing_docs = dict(
        zip(existing["ids"], existing["documents"])
    )

    # Chỉ lấy các chunk mới hoặc đã thay đổi
    changed = [
        doc for doc in documents
        if existing_docs.get(doc["id"]) != doc["text"]
    ]

    current_ids = {doc["id"] for doc in documents}

    deleted_ids = [
        doc_id
        for doc_id in existing["ids"]
        if doc_id not in current_ids
    ]

    if changed:

        print(f"Embedding {len(changed)} chunks...")

        response = embed(
            model=EMBED_MODEL,
            input=[doc["text"] for doc in changed]
        )

        collection.upsert(
            ids=[doc["id"] for doc in changed],

            embeddings=[
                list(vector)
                for vector in response.embeddings
            ],

            documents=[
                doc["text"] for doc in changed
            ],

            metadatas=[
                {
                    "source": doc["source"],
                    "chunk": doc["chunk"]
                }
                for doc in changed
            ]
        )

    # Xóa các chunk không còn trong tài liệu
    if deleted_ids:
        collection.delete(ids=deleted_ids)

    print("\nINDEXING COMPLETED!")
    print(f"New/updated chunks: {len(changed)}")
    print(f"Deleted chunks: {len(deleted_ids)}")
    print(f"Total stored: {collection.count()}")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    index_documents()
