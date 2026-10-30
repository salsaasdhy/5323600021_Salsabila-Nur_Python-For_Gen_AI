
from pathlib import Path
from dotenv import load_dotenv
import os
import re

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv(Path(__file__).resolve().parent / ".env")

Settings.llm = OpenAILike(
    model="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    context_window=128000,
    is_chat_model=True,
    max_tokens=3000
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1"
)

documents = [
    Document(text="HNSW uses a hierarchical graph for approximate nearest-neighbor search."),
    Document(text="IVF clusters vectors to reduce the number of comparisons."),
    Document(text="Product Quantization compresses vectors to save memory."),
    Document(text="Cosine similarity compares vector directions."),
    Document(text="Reranking improves retrieval by reordering candidate passages."),
    Document(text="Metadata filtering restricts search by attributes."),
    Document(text="Hybrid retrieval combines keyword and vector search."),
    Document(text="Chunking divides large documents into smaller passages."),
    Document(text="Embeddings represent text in vector space."),
    Document(text="Exact search compares a query with every stored vector."),
    Document(text="Image compression reduces file size."),
    Document(text="Web development uses HTML, CSS, and JavaScript.")
]

index = VectorStoreIndex.from_documents(documents)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10
)

def rerank(question: str, top_k: int = 3) -> list:
    candidates = retriever.retrieve(question)
    scored_nodes = []

    for node in candidates:
        prompt = f"""
Rate the relevance of the passage to the question from 0 to 10.
Return only one integer from 0 to 10.

Question: {question}

Passage:
{node.text}
"""
        response = Settings.llm.complete(prompt)
        match = re.search(r"\b(10|[0-9])\b", str(response))
        score = int(match.group(1)) if match else 0

        scored_nodes.append({
            "score": score,
            "text": node.text,
            "metadata": node.metadata
        })

    scored_nodes.sort(key=lambda item: item["score"], reverse=True)
    return scored_nodes[:top_k]

if __name__ == "__main__":
    question = "Which techniques improve vector search retrieval?"
    results = rerank(question)

    for rank, item in enumerate(results, start=1):
        print(f"\nRank {rank} | Score: {item['score']}/10")
        print(item["text"])
