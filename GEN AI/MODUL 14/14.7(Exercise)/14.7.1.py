import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM  # Gunakan LLM dari crewai

# 1. Deteksi path .env (Mendukung pencarian di folder MODUL 14)
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env" if (script_dir / ".env").exists() else script_dir.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(f"OPENROUTER_API_KEY tidak ditemukan! Path yang diperiksa: {env_path}")

# 2. Inisialisasi LLM CrewAI dengan OpenRouter Auto-Free
llm = LLM(
    model="openrouter/openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# 3. Setup Agents
researcher = Agent(
    role="AI Research Specialist",
    goal="Gather technical facts, advantages, and trade-offs between Fine-Tuning and RAG.",
    backstory="An AI expert who specializes in evaluating LLM customization techniques.",
    llm=llm,
    verbose=False
)

outline_writer = Agent(
    role="Technical Content Strategist",
    goal="Structure research findings into a logical, well-organized blog post outline.",
    backstory="A seasoned content architect who structures complex technical concepts into engaging blog frameworks.",
    llm=llm,
    verbose=False
)

full_writer = Agent(
    role="Senior Technical Writer",
    goal="Expand the blog post outline into a comprehensive, developer-friendly article.",
    backstory="A technical writer with software engineering experience who produces clear, production-grade technical articles.",
    llm=llm,
    verbose=False
)

# 4. Setup Tasks
topic = "The practical guide to fine-tuning vs RAG"

research_task = Task(
    description=f"Conduct in-depth technical research on '{topic}'. Focus on core differences, use cases, costs, and key decision factors.",
    expected_output="Bullet points of key technical comparison metrics, pros/cons, and implementation scenarios.",
    agent=researcher
)

outline_task = Task(
    description="Take the research output and construct a structured blog outline containing an Introduction, Core Comparison, Decision Framework, and Conclusion.",
    expected_output="A structured blog post outline with section headings and brief descriptions for each section.",
    agent=outline_writer
)

writing_task = Task(
    description="Expand the outline into a full technical blog post. Ensure clear explanations, professional tone, and actionable insights for engineers.",
    expected_output="A complete, publication-ready technical blog post in Markdown format.",
    agent=full_writer
)

# 5. Run Crew
crew = Crew(
    agents=[researcher, outline_writer, full_writer],
    tasks=[research_task, outline_task, writing_task],
    process=Process.sequential,
    verbose=False
)

print("=== EXERCISE 14.7.1: EXECUTING CONTENT PIPELINE ===")
result = crew.kickoff()

print("\n" + "="*40 + " FINAL POST " + "="*40 + "\n")
print(result)