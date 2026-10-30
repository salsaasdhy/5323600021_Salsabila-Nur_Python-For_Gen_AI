from llama_index.core import SummaryIndex, Document, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from pathlib import Path
from dotenv import load_dotenv
import os

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
# Simulate a long document split into multiple sections
sections = [
    Document(text="Introduction: This report analyses the performance of five LLMs on coding, reasoning, and language tasks."),
    Document(text="Section 1 - Coding: Claude Sonnet 4.5 scored highest on HumanEval with 94.2%. GPT-4o followed at 90.1%."),
    Document(text="Section 2 - Reasoning: GPT-4o led on GSM8K with 97.1%. Claude Sonnet 4.5 achieved 96.8%."),
    Document(text="Section 3 - Language: Both models performed similarly on MMLU, within 1% of each other."),
    Document(text="Conclusion: For coding tasks, prefer Claude Sonnet 4.5. For balanced performance, GPT-4o is the safer choice."),
]

index = SummaryIndex.from_documents(sections)

query_engine = index.as_query_engine()

# SummaryIndex reads ALL nodes, not just top-k - good for document-wide questions
response = query_engine.query(
    "Which model should I use for a coding project?"
)

print(response)