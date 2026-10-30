
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Pastikan API key OpenRouter tersedia
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY tidak ditemukan. "
        "Periksa file .env kamu."
    )

# Model pertama melalui OpenRouter
llm_oai = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.7,
    max_tokens=1024,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Model kedua melalui OpenRouter
llm_ant = ChatOpenAI(
    model="anthropic/claude-sonnet-4.5",
    temperature=0.7,
    max_tokens=1024,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Kedua model menerima tipe pesan yang sama
messages = [
    SystemMessage(
        content="You are a concise technical assistant."
    ),
    HumanMessage(
        content="What is LangChain Expression Language?"
    ),
]

# 1. Invoke: menghasilkan satu respons
response = llm_ant.invoke(messages)
print("INVOKE:")
print(response.content)

# 2. Stream: menampilkan respons secara bertahap
print("\nSTREAM:")
for chunk in llm_ant.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()

# 3. Batch: memproses beberapa input
print("\nBATCH:")
responses = llm_ant.batch([
    [HumanMessage(content="What is RAG?")],
    [HumanMessage(content="What is a vector database?")],
])

for r in responses:
    print(r.content[:80])
