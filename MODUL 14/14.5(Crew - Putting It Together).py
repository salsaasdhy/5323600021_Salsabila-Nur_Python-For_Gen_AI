import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Crew, Process, Agent, Task, LLM  # Pastikan LLM di-import di sini

# 1. Load file .env di folder MODUL 14
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY tidak ditemukan di .env!")

llm = LLM(
    model="openrouter/openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)
# 3. Setup Agents
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate information on AI topics.",
    backstory="Senior AI research analyst with expertise in LLM systems.",
    llm=llm,
    verbose=False,
    max_iter=5,
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write clear technical content for developers.",
    backstory="Technical writer with a software engineering background.",
    llm=llm,
    verbose=False,
    max_iter=5,
)

reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate and publication-ready.",
    backstory="Meticulous editor who reviews technical AI content.",
    llm=llm,
    verbose=False,
    max_iter=5,
)

# 4. Setup Tasks
research_task = Task(
    description="Lakukan riset singkat tentang tren perkembangan LLM modern.",
    expected_output="Daftar poin-poin penting tren LLM.",
    agent=researcher
)

writing_task = Task(
    description="Tulis draf artikel teknis singkat berdasarkan hasil riset.",
    expected_output="Artikel teknis singkat 2 paragraf.",
    agent=writer
)

review_task = Task(
    description="Review dan pastikan artikel teknis akurat serta siap dipublikasikan.",
    expected_output="Artikel teknis final.",
    agent=reviewer
)

# 5. Jalankan Crew
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=False,
    memory=False,
)

print("Memulai eksekusi CrewAI...")
result = crew.kickoff()

print("\n=== CREW OUTPUT ===")
print(result)

print("\n=== TASK OUTPUTS ===")
for task in crew.tasks:
    print(f"\n--- {task.agent.role} ---")
    if task.output:
        print(str(task.output)[:200], "...")