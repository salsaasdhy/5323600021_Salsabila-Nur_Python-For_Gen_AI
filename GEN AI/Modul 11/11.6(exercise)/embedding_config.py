from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(Path(__file__).resolve().parent / ".env")

openrouter_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

def embed(texts: list[str]) -> list[list[float]]:
    response = openrouter_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=texts
    )
    return [
        item.embedding
        for item in sorted(response.data, key=lambda x: x.index)
    ]