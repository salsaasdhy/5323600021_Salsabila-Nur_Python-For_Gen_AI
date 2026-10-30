from crewai import Task
from crewai import Agent  # assume agents from 14.2 are defined


# Each task has: description, expected_output, agent
def make_tasks(researcher, writer, reviewer):

    research_task = Task(
        description="""Research the current state of RAG (Retrieval-Augmented Generation) in 2024-2025.
Cover:
1. What RAG is and how it works
2. The main variants (naive RAG, advanced RAG, modular RAG)
3. Common production challenges and how practitioners solve them
4. Three real-world use cases with specific results

Be specific and factual. Avoid vague generalities.""",

        expected_output="""A structured research report with:
- A clear definition and architecture overview
- Descriptions of the three RAG variants with key differences
- Three specific production challenges with concrete solutions
- Three real-world use cases with measurable outcomes

Format: Markdown with headers.""",

        agent=researcher,
    )

    writing_task = Task(
        description="""Using the research report provided, write a developer guide titled
'Building Production RAG Systems'.

The guide should:
- Be written for senior developers with Python experience
- Include a conceptual overview, implementation considerations, and common pitfalls
- Use practical Python code examples where appropriate
- Explain how to evaluate and monitor RAG systems
- Include recommendations for production deployment

Use the research report as the primary source.""",

        expected_output="""A comprehensive developer guide titled
'Building Production RAG Systems'.

The guide must include:
- Introduction and conceptual overview
- RAG architecture and workflow
- Implementation considerations
- Practical Python code examples
- Common pitfalls and solutions
- Evaluation and monitoring strategies
- Production deployment recommendations

Format: Markdown with clear headers and code blocks.""",

        agent=writer,
    )

    review_task = Task(
        description="""Review the developer guide produced by the writer.

Evaluate:
1. Technical accuracy
2. Completeness of explanations
3. Quality and correctness of Python examples
4. Practical usefulness for senior developers
5. Clarity and organization

Identify inaccuracies, missing information, and areas for improvement.
Provide specific recommendations.""",

        expected_output="""A structured review report containing:
- Overall assessment
- Technical accuracy findings
- Missing or incomplete information
- Issues in Python code examples
- Specific recommendations for improvement

Format: Markdown with headers.""",

        agent=reviewer,
    )

    return research_task, writing_task, review_task
