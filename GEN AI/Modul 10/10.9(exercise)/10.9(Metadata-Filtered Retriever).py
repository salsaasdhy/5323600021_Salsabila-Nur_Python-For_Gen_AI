
from pathlib import Path
from dotenv import load_dotenv
import os

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

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

documents = [
    Document(
        text="Embedding models map text into numerical vectors.",
        metadata={"year": 2024, "topic": "embeddings"}
    ),
    Document(
        text="Vector similarity helps retrieve related passages.",
        metadata={"year": 2024, "topic": "embeddings"}
    ),
    Document(
        text="Transformer models use attention mechanisms.",
        metadata={"year": 2024, "topic": "transformers"}
    ),
    Document(
        text="Older embedding methods use word-level representations.",
        metadata={"year": 2023, "topic": "embeddings"}
    ),
]

index = VectorStoreIndex.from_documents(documents)

filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="year",
            value=2024,
            operator=FilterOperator.EQ
        ),
        MetadataFilter(
            key="topic",
            value="embeddings",
            operator=FilterOperator.EQ
        )
    ],
    condition="and"
)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    filters=filters
)

results = retriever.retrieve(
    "How do embedding models work?"
)

for node in results:
    print("Metadata:", node.metadata)
    print("Text:", node.text)
    print()
