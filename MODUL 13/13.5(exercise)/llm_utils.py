# pip install langgraph langchain-openai python-dotenv
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # butuh OPENROUTER_API_KEY di .env


def get_llm(model: str = "openai/gpt-4o-mini", max_tokens: int = 500, temperature: float = 0):
    """LLM lewat OpenRouter. max_tokens dibatasi supaya kredit yang dicadangkan kecil."""
    return ChatOpenAI(
        model=model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        max_tokens=max_tokens,
        temperature=temperature,
    )