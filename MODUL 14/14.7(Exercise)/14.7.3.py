import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM  # Import LLM dari crewai
from crewai.tools import tool

# 1. Deteksi path .env (Mencari di folder 14.7(Exercise) atau folder MODUL 14)
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env" if (script_dir / ".env").exists() else script_dir.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(f"OPENROUTER_API_KEY tidak ditemukan! Path yang diperiksa: {env_path}")

# 2. Inisialisasi LLM menggunakan class LLM dari CrewAI
llm = LLM(
    model="openrouter/openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# 3. Definisi Custom Mock Search Tool
@tool("Mock Web Search Tool")
def mock_web_search(query: str) -> str:
    """Useful to search technical information on the web given a query string."""
    print(f"\n[TOOL LOG] Searching for query: '{query}'")
    query_lower = query.lower()
    
    if "fine-tuning" in query_lower or "rag" in query_lower:
        return (
            "Mock Web Result: Fine-Tuning modifies the internal parameters of an LLM for specific style/format, "
            "whereas RAG injects dynamic external document context into the prompt at query time."
        )
    return f"Mock Web Result: Found general information for topic '{query}'."

# 4. Inisialisasi Agent
researcher = Agent(
    role="Tech Research Analyst",
    goal="Answer complex technical questions by searching external resources.",
    backstory="A research assistant who relies on web search tools to gather precise factual data.",
    tools=[mock_web_search],
    llm=llm,
    verbose=True  # verbose=True agar pemanggilan Tool terlihat di konsol
)

# 5. Inisialisasi Task
research_task = Task(
    description="Research the core structural differences between Fine-Tuning and RAG using the search tool.",
    expected_output="A concise paragraph summarizing the findings obtained from the search tool.",
    agent=researcher
)

# 6. Run Crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential
)

print("=== EXERCISE 3: TESTING CUSTOM SEARCH TOOL ===")
result = crew.kickoff()
print("\n=== FINAL ANSWER ===")
print(result)