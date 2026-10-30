from crewai import Agent, LLM
import os
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    max_tokens=1500,
)

# Research specialist
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate and up-to-date information on AI topics.",
    backstory="""You are a senior AI research analyst with 10 years of experience
    reading and synthesising academic papers, technical blogs, and release notes.
    You are known for being precise, citing sources, and never making things up.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,  # this agent does not delegate to others
    max_iter=5,              # max reasoning iterations before stopping
)

# Writing specialist
writer = Agent(
    role="Technical Content Writer",
    goal="Write clear, accurate, and engaging technical content for developers.",
    backstory="""You are a technical writer with a background in software engineering.
    You excel at turning complex AI concepts into practical, well-structured guides.
    You always write in active voice, use concrete examples, and avoid jargon.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# Quality reviewer
reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate, complete, and publication-ready.",
    backstory="""You are a meticulous editor who reviews technical content for
    accuracy, clarity, and completeness. You check facts, flag unsupported claims,
    and ensure the writing matches the target audience's level.""",
    llm=llm,
    verbose=True,
    allow_delegation=True,  # reviewer can delegate back to writer for revisions
)
#untuk membuktikan bahwa code berfungsi:
print(researcher.role)
print(writer.role)
print(reviewer.role)
