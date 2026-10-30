import chromadb
from embedding_config import embed
class MultiCollectionSearch:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./multi_collection_db"
        )

    # --- Search across multiple collections ---
    def search(self, collection_names, query, top_k=5):

        query_embedding = embed([query])[0]

        all_results = []

        for collection_name in collection_names:

            collection = self.client.get_collection(
                name=collection_name
            )

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )

            for i, doc in enumerate(results["documents"][0]):

                distance = results["distances"][0][i]

                all_results.append({
                    "collection": collection_name,
                    "document": doc,
                    "distance": distance,
                    "score": 1 - distance,
                    "metadata": results["metadatas"][0][i]
                })

        # --- Merge and rank results ---
        all_results.sort(
            key=lambda x: x["distance"]
        )

        return all_results[:top_k]


# --- Example usage ---
client = chromadb.PersistentClient(
    path="./multi_collection_db"
)

# Create example collections
collection1 = client.get_or_create_collection(
    name="ai_knowledge"
)

collection2 = client.get_or_create_collection(
    name="rag_knowledge"
)
# Add documents
docs1 = [
    "Artificial intelligence enables machines to perform tasks.",
    "Machine learning is a branch of artificial intelligence."
]

docs2 = [
    "RAG retrieves external knowledge for LLMs.",
    "Vector databases store embeddings for similarity search."
]

collection1.upsert(
    ids=["ai_1", "ai_2"],
    documents=docs1,
    embeddings=embed(docs1)
)

collection2.upsert(
    ids=["rag_1", "rag_2"],
    documents=docs2,
    embeddings=embed(docs2)
)

# --- Search ---
searcher = MultiCollectionSearch()

results = searcher.search(
    collection_names=["ai_knowledge", "rag_knowledge"],
    query="How can an LLM access external knowledge?",
    top_k=3
)

print("\n--- Unified Search Results ---")

for result in results:
    print(f"Collection: {result['collection']}")
    print(f"Document: {result['document']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Distance: {result['distance']:.4f}")
    print()
