
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_tokens=500,
)

parser = StrOutputParser()

# 1. Extract 3 key facts (LLM call pertama)
facts_prompt = ChatPromptTemplate.from_template(
    """Extract exactly 3 key facts from this document.
    Return each fact on a separate line, numbered 1 to 3.

    Document:
    {document}"""
)

facts_chain = facts_prompt | llm | parser

# 2. Generate a quiz question for each fact (LLM call kedua)
quiz_prompt = ChatPromptTemplate.from_template(
    """Create one clear quiz question for each of these 3 facts.
    Return only the questions, one per line, numbered 1 to 3.

    Facts:
    {facts}"""
)

quiz_chain = quiz_prompt | llm | parser

# 3. Combine both calls
chain = (
    {"facts": facts_chain, "document": lambda x: x}
    | quiz_chain
    | (lambda result: [
        line.split(". ", 1)[-1].strip()
        for line in result.splitlines()
        if line.strip()
    ])
)

document = """
RAG stands for Retrieval-Augmented Generation.
It uses retrieval to find relevant external information.
ChromaDB is a vector database used to store and search embeddings.
"""

questions = chain.invoke(document)
print(questions)
print(type(questions))  # <class 'list'>
