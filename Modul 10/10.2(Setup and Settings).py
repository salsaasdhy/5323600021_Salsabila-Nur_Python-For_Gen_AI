
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
# Configure global settings - applied to all indexes and query engines

Settings.llm = OpenAI(
    model="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=1024,
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
)

Settings.chunk_size = 512  # tokens per chunk
Settings.chunk_overlap = 64  # token overlap between chunks


print("Konfigurasi Modul 10 berhasil dijalankan!")
