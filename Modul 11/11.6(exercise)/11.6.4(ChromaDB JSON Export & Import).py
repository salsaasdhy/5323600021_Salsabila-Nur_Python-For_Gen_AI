{
    "id": "doc_0",
    "text": "RAG connects LLMs to external knowledge bases.",
    "embedding": [0.12, 0.34, 0.56],
    "metadata": {
        "category": "technique"
    }
}


import chromadb
import json


# --- Persistent client ---
client = chromadb.PersistentClient(
    path="./export_db"
)

collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)


# --- Export collection to JSON ---
def export_collection(collection, output_file):

    data = collection.get(
        include=["documents", "embeddings", "metadatas"]
    )

    exported_data = []

    for i, doc_id in enumerate(data["ids"]):

        exported_data.append({
            "id": doc_id,
            "text": data["documents"][i],
            "embedding": data["embeddings"][i].tolist(),
            "metadata": data["metadatas"][i]
        })

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            exported_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Exported {len(exported_data)} documents.")


# --- Import JSON back to ChromaDB ---
def import_collection(collection, input_file):

    with open(input_file, "r", encoding="utf-8") as file:
        imported_data = json.load(file)

    ids = [item["id"] for item in imported_data]

    documents = [
        item["text"]
        for item in imported_data
    ]

    embeddings = [
        item["embedding"]
        for item in imported_data
    ]

    metadatas = [
        item["metadata"]
        for item in imported_data
    ]

    if imported_data:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print(f"Imported {len(imported_data)} documents.")


# --- Example documents ---
documents = [
    "RAG connects LLMs to external knowledge bases.",
    "Vector databases store numerical embeddings.",
    "Prompt engineering guides LLM responses."
]

ids = ["doc_0", "doc_1", "doc_2"]

metadatas = [
    {"category": "RAG"},
    {"category": "Vector Database"},
    {"category": "Prompt Engineering"}
]

# Generate embeddings
from embedding_config import embed

embeddings = embed(documents)

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

# --- Export ---
export_collection(
    collection,
    "chromadb_backup.json"
)

# --- Import into another collection ---
backup_collection = client.get_or_create_collection(
    name="restored_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

import_collection(
    backup_collection,
    "chromadb_backup.json"
)

print(
    "Restored collection size:",
    backup_collection.count()
)
