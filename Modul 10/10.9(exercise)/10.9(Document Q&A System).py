
from pathlib import Path
from dotenv import load_dotenv
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
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

DATA_DIR = Path(__file__).resolve().parent / "txt_documents"

def build_index():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = list(DATA_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"Tambahkan file .txt terlebih dahulu ke {DATA_DIR}"
        )

    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".txt"],
        filename_as_id=True
    ).load_data()

    return VectorStoreIndex.from_documents(documents)

index = build_index()
query_engine = index.as_query_engine(similarity_top_k=3)

def ask(question: str) -> dict:
    response = query_engine.query(question)

    sources = sorted({
        Path(node.node.metadata.get("file_name", "unknown")).name
        for node in response.source_nodes
    })

    return {
        "answer": str(response),
        "source_filenames": sources
    }

if __name__ == "__main__":
    result = ask("What information is available in these documents?")
    print(result)
