import os
import time
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM  # Import LLM dari crewai

# 1. Deteksi path .env (mencari di folder 14.7(Exercise) atau folder MODUL 14)
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

# 3. Fungsi pembangun pipeline
def build_pipeline():
    a1 = Agent(role="Analyst", goal="Identify 3 key AI trends.", backstory="Data Analyst", llm=llm, verbose=False)
    a2 = Agent(role="Writer", goal="Draft a summary of the trends.", backstory="Tech Writer", llm=llm, verbose=False)
    a3 = Agent(role="Editor", goal="Polish the summary into bullet points.", backstory="Senior Editor", llm=llm, verbose=False)

    t1 = Task(description="Identify 3 major AI trends in 2026.", expected_output="3 trends list.", agent=a1)
    t2 = Task(description="Summarize the 3 trends into a short paragraph.", expected_output="Summary paragraph.", agent=a2)
    t3 = Task(description="Polish the summary into publication-ready bullet points.", expected_output="Polished bullet points.", agent=a3)

    return [a1, a2, a3], [t1, t2, t3]

# 4. Test Process.sequential
print("Running Process.sequential...")
agents_seq, tasks_seq = build_pipeline()
crew_seq = Crew(agents=agents_seq, tasks=tasks_seq, process=Process.sequential, verbose=False)

start_seq = time.time()
out_seq = crew_seq.kickoff()
dur_seq = time.time() - start_seq

# 5. Test Process.hierarchical (Membutuhkan manager_llm)
print("Running Process.hierarchical...")
agents_hier, tasks_hier = build_pipeline()
crew_hier = Crew(
    agents=agents_hier, 
    tasks=tasks_hier, 
    process=Process.hierarchical,
    manager_llm=llm,  # LLM yang sama digunakan sebagai manajer
    verbose=False
)

start_hier = time.time()
out_hier = crew_hier.kickoff()
dur_hier = time.time() - start_hier

# 6. Output Perbandingan
print("\n" + "="*50)
print("=== EXERCISE 4: PROCESS COMPARISON ===")
print("="*50)
print(f"1. Process.sequential Execution Time : {dur_seq:.2f} seconds")
print(f"2. Process.hierarchical Execution Time: {dur_hier:.2f} seconds")