
import chromadb
import hashlib

from embedding_config import embed


class TenantVectorStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./tenant_db"
        )

    # --- Get collection for a specific tenant ---
    def get_tenant_collection(self, tenant_id):

        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("Invalid tenant ID")

        # Generate a safe and unique collection name
        tenant_hash = hashlib.sha256(
            tenant_id.encode()
        ).hexdigest()[:16]

        collection_name = f"tenant_{tenant_hash}"

        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # --- Add documents ---
    def add(self, tenant_id, documents, ids):

        if len(documents) != len(ids):
            raise ValueError("Documents and IDs must have the same length")

        collection = self.get_tenant_collection(tenant_id)

        embeddings = embed(documents)

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )

        print(f"Added documents for tenant: {tenant_id}")

    # --- Search documents ---
    def search(self, tenant_id, query, top_k=3):

        collection = self.get_tenant_collection(tenant_id)

        query_embedding = embed([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"]
        )

        return results


# --- Example usage ---
store = TenantVectorStore()

# Tenant 1
store.add(
    tenant_id="mahasiswa_001",
    ids=["doc_1", "doc_2"],
    documents=[
        "Pomodoro helps students manage study time.",
        "Time management improves learning productivity."
    ]
)

# Tenant 2
store.add(
    tenant_id="mahasiswa_002",
    ids=["doc_1", "doc_2"],
    documents=[
        "Machine learning uses data to train models.",
        "Deep learning uses neural networks."
    ]
)

# Search tenant 1
results = store.search(
    tenant_id="mahasiswa_001",
    query="How can students manage study time?"
)

print("\n--- Tenant 1 Search Results ---")

for doc, distance in zip(
    results["documents"][0],
    results["distances"][0]
):
    print(f"Document: {doc}")
    print(f"Distance: {distance:.4f}")
    print()

# Search tenant 2
results = store.search(
    tenant_id="mahasiswa_002",
    query="How does machine learning work?"
)

print("\n--- Tenant 2 Search Results ---")

for doc, distance in zip(
    results["documents"][0],
    results["distances"][0]
):
    print(f"Document: {doc}")
    print(f"Distance: {distance:.4f}")
    print()
