
from pathlib import Path
from dotenv import load_dotenv
import os

from llama_index.core import (
    VectorStoreIndex, SummaryIndex, Document, Settings
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv(Path(__file__).resolve().parent / ".env")

Settings.llm = OpenAILike(
    model="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    context_window=128000,
    is_chat_model=True,
    max_tokens=1000
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1"
)

paragraphs = [
    "HNSW is a graph-based algorithm used for approximate nearest-neighbor search.",
    "IVF divides vectors into clusters to reduce the search space.",
    "Product Quantization compresses vectors to reduce memory usage.",
    "Exact nearest-neighbor search compares a query against all stored vectors.",
    "Vector embeddings represent text as numerical vectors.",
    "Cosine similarity measures the angle-based similarity between vectors.",
    "Chunking splits long documents into smaller searchable passages.",
    "Metadata can filter documents by year, topic, or source.",
    "Reranking reorders retrieved passages based on query relevance.",
    "Hybrid search combines lexical retrieval with vector similarity."
]

documents = [
    Document(text=p, metadata={"paragraph": i + 1})
    for i, p in enumerate(paragraphs)
]

vector_index = VectorStoreIndex.from_documents(documents)
summary_index = SummaryIndex.from_documents(documents)

vector_engine = vector_index.as_query_engine(similarity_top_k=3)
summary_engine = summary_index.as_query_engine()

questions = [
    "Which algorithm compresses vectors?",
    "What does IVF do?",
    "How does cosine similarity measure similarity?",
    "Explain the main stages of a vector search system.",
    "Summarize the techniques described in all ten paragraphs."
]

for i, question in enumerate(questions, start=1):
    print(f"\nQUESTION {i}: {question}")

    vector_answer = vector_engine.query(question)
    summary_answer = summary_engine.query(question)

    print("\n[VectorStoreIndex]")
    print(str(vector_answer))

    print("\n[SummaryIndex]")
    print(str(summary_answer))
