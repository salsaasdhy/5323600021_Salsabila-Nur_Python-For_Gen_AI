import os
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

# 3. Python Snippet dengan celah keamanan dan efisiensi rendah
code_snippet = """
def fetch_and_process_user_data(user_id, input_filter):
    import sqlite3
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    
    # Raw SQL execution
    query = f"SELECT * FROM users WHERE id = '{user_id}' AND status = '{input_filter}'"
    cursor.execute(query)
    records = cursor.fetchall()
    
    # Inefficient nested loops
    duplicates = []
    for item1 in records:
        for item2 in records:
            if item1[0] == item2[0] and item1 != item2:
                duplicates.append(item1)
                
    return records, duplicates
"""

# 4. Inisialisasi Agents
security_reviewer = Agent(
    role="AppSec Engineer",
    goal="Identify security vulnerabilities and risks in Python code.",
    backstory="A cybersecurity specialist focused on OWASP standards and secure coding practices.",
    llm=llm,
    verbose=False
)

performance_reviewer = Agent(
    role="Performance Optimization Engineer",
    goal="Identify performance bottlenecks, memory leaks, and high algorithmic complexity.",
    backstory="An engineer obsessed with time complexity, SQL optimization, and Python runtime efficiency.",
    llm=llm,
    verbose=False
)

lead_reviewer = Agent(
    role="Tech Lead",
    goal="Synthesize security and performance feedback into a unified review report with refactored code.",
    backstory="A veteran Engineering Lead who balances code quality, security, and developer productivity.",
    llm=llm,
    verbose=False
)

# 5. Inisialisasi Tasks
sec_task = Task(
    description=f"Analyze the following code for security vulnerabilities:\n\n```python\n{code_snippet}\n```",
    expected_output="List of security risks (e.g., SQL Injection) with severity levels and mitigation advice.",
    agent=security_reviewer
)

perf_task = Task(
    description=f"Analyze the following code for performance inefficiencies:\n\n```python\n{code_snippet}\n```",
    expected_output="List of performance issues (e.g., O(N^2) complexity, unindexed queries) and optimization suggestions.",
    agent=performance_reviewer
)

lead_task = Task(
    description="Combine the security and performance findings into a final Code Review Report. Include a refactored version of the code that resolves all issues.",
    expected_output="A structured report with 'Security Audit', 'Performance Audit', and 'Refactored Python Code'.",
    agent=lead_reviewer
)

# 6. Run Crew
crew = Crew(
    agents=[security_reviewer, performance_reviewer, lead_reviewer],
    tasks=[sec_task, perf_task, lead_task],
    process=Process.sequential,
    verbose=False
)

print("=== EXERCISE 2: CODE REVIEW CREW ===")
result = crew.kickoff()

print("\n" + "="*40 + " CODE REVIEW REPORT " + "="*40 + "\n")
print(result)