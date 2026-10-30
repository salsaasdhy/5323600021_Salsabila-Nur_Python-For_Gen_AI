
import chromadb
from chromadb.config import Settings
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# --- Load environment variables ---
load_dotenv(Path(__file__).resolve().parent / ".env")

# --- OpenRouter client ---
openrouter_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# --- Embedding helper ---
def embed(texts: list[str]) -> list[list[float]]:
    resp = openrouter_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=texts
    )
    return [
        e.embedding
        for e in sorted(resp.data, key=lambda x: x.index)
    ]


# --- Persistent client (saves to disk) ---
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# --- Get or create a collection ---
collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)

# --- Add documents with pre-computed embeddings ---
documents = [
    "RAG connects LLMs to external knowledge bases at inference time.",
    "Fine-tuning adapts model weights using task-specific training data.",
    "Prompt engineering guides LLM outputs without any model training.",
    "Vector databases index embeddings for fast similarity search.",
    "Agents use LLMs to plan and execute multi-step tasks with tools.",
]

ids = [f"doc_{i}" for i in range(len(documents))]

metadatas = [
    {"category": "technique", "index": i}
    for i in range(len(documents))
]

embeddings = embed(documents)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)

print(f"Collection size: {collection.count()}")


# --- Query ---
query = "How do I add knowledge to an LLM without retraining?"

q_embed = embed([query])[0]

results = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    include=["documents", "distances", "metadatas"],
)

print("\nTop results:")

for doc, dist, meta in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0],
):
    print(f"[{dist:.4f}] {doc}")
    print(f"Category: {meta['category']}")
# --- Filter by metadata before similarity ranking ---
results_filtered = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    where={"category": {"$eq": "technique"}},  # pre-filter
    include=["documents", "distances"],
)

print("\nFiltered results:")

for doc, dist in zip(
    results_filtered["documents"][0],
    results_filtered["distances"][0],
):
    print(f"[{dist:.4f}] {doc}")


# --- Update a document ---
updated_document = (
    "RAG retrieves external documents and injects them "
    "into the LLM context."
)

collection.update(
    ids=["doc_0"],
    documents=[updated_document],
    embeddings=embed([updated_document]),
)


# --- Delete a document ---
collection.delete(ids=["doc_4"])

print(f"After delete, count: {collection.count()}")