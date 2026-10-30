import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Crew, Process, Agent, Task, LLM

# 1. Load file .env yang ada di folder MODUL 14
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env" if (script_dir / ".env").exists() else script_dir.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# 2. Ambil API Key OpenRouter
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(f"OPENROUTER_API_KEY tidak ditemukan di {env_path}")

# 3. Inisialisasi LLM via OpenRouter
llm = LLM(
    model="openrouter/openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# 4. Inisialisasi Agents
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate information on AI topics.",
    backstory="Senior AI research analyst with expertise in LLM systems.",
    llm=llm,
    verbose=False
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write clear technical content for developers.",
    backstory="Technical writer with a software engineering background.",
    llm=llm,
    verbose=False
)

reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate and publication-ready.",
    backstory="Meticulous editor who reviews technical AI content.",
    llm=llm,
    verbose=False
)

# 5. Inisialisasi Tasks
research_task = Task(
    description="Riset poin-poin utama perkembangan arsitektur LLM modern.",
    expected_output="Daftar poin-poin hasil riset.",
    agent=researcher
)

writing_task = Task(
    description="Tulis draf artikel teknis singkat berdasarkan hasil riset.",
    expected_output="Artikel teknis 2 paragraf.",
    agent=writer
)

review_task = Task(
    description="Review artikel teknis agar bebas kesalahan dan siap terbit.",
    expected_output="Artikel teknis final.",
    agent=reviewer
)

# 6. Buat Crew dengan Process.hierarchical (Wajib menyertakan manager_llm)
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.hierarchical,  # Menggunakan proses hierarki
    manager_llm=llm,               # LLM yang bertindak sebagai Manajer
    verbose=False
)

print("Memulai eksekusi Hierarchical CrewAI...")
result = crew.kickoff()

print("\n=== CREW OUTPUT ===")
print(result)