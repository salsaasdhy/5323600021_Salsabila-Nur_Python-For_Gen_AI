
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from pathlib import Path
from dotenv import load_dotenv

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

PERSIST_DIR = str(Path(__file__).resolve().parent / "llama_index_store")

# Build and save index
documents = [
    Document(text="LlamaIndex persists indexes to disk as JSON files."),
    Document(text="Loading a persisted index avoids re-embedding on every run."),
]

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=PERSIST_DIR)
print(f"Index saved to {PERSIST_DIR}")

# Load index back - no re-embedding needed
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
loaded_index = load_index_from_storage(storage_context)
query_engine = loaded_index.as_query_engine()
response = query_engine.query("How does LlamaIndex handle persistence?")
print("After reload:", response)
