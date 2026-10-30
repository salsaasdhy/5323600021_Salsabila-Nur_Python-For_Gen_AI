
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
from dotenv import load_dotenv
load_dotenv()

# 1. Documents
docs = [
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It connects LLMs to external knowledge bases."),
    Document(page_content="The retrieval step uses embeddings and vector search to find relevant documents for a given query."),
    Document(page_content="The generation step takes the retrieved context and the user question to produce a grounded answer."),
    Document(page_content="Vector databases like ChromaDB, Pinecone, and Weaviate store embeddings for fast similarity search."),
    Document(page_content="Fine-tuning bakes knowledge into model weights. RAG retrieves knowledge at inference time. Both have trade-offs."),
]

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(docs)

# 3. Embed and store in Chroma (in-memory)
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain the answer, say "I don't have that information."
Context:
{context}"""),
    ("human", "{question}"),
])

# 5. Chain
llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_tokens=500,
)
parser = StrOutputParser()

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | parser
)

# 6. Response-quality filter
quality_prompt = ChatPromptTemplate.from_template(
    """Evaluate how well the answer is grounded in the provided context.

Give a score from 0 to 10:
- 0 means the answer is unsupported by the context.
- 5 means the answer is partially supported.
- 10 means every factual claim is supported by the context.

Return only one integer from 0 to 10.

Context:
{context}

Answer:
{answer}"""
)

quality_chain = quality_prompt | llm | parser


def rag_with_quality_filter(question: str) -> dict:
    # Retrieve documents
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)

    # Generate answer using the retrieved context
    answer = rag_chain.invoke(question)

    # Score groundedness using a second LLM call
    score_text = quality_chain.invoke({
        "context": context,
        "answer": answer,
    })

    # Extract score
    match = re.search(r"\b(10|[0-9])\b", score_text)
    score = int(match.group(1)) if match else None

    return {
        "answer": answer,
        "score": score,
    }


# 7. Ask questions
questions = [
    "What does RAG stand for?",
    "How is RAG different from fine-tuning?",
    "What is the capital of France?",  # Not in context
]

for q in questions:
    result = rag_with_quality_filter(q)
    print(f"Q: {q}")
    print(f"A: {result['answer']}")
    print(f"Groundedness score: {result['score']}/10\n")
