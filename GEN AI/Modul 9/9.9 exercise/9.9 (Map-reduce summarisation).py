import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_tokens=500,
)

parser = StrOutputParser()

# 1. Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)

# 2. Summarise each chunk (map)
map_prompt = ChatPromptTemplate.from_template(
    """Summarise the following text clearly and concisely.

    Text:
    {text}"""
)

map_chain = map_prompt | llm | parser

# 3. Summarise the summaries (reduce)
reduce_prompt = ChatPromptTemplate.from_template(
    """Combine the following partial summaries into one coherent summary.
    Preserve the most important information and remove repetition.

    Partial summaries:
    {summaries}"""
)

reduce_chain = reduce_prompt | llm | parser


def map_reduce_summarise(document):
    chunks = splitter.split_text(document)

    # Map: summarise each chunk
    summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    # Reduce: combine all partial summaries
    final_summary = reduce_chain.invoke(
        {"summaries": "\n\n".join(summaries)}
    )

    return final_summary



long_document = """
Retrieval-Augmented Generation (RAG) is a technique that combines
information retrieval with text generation. It allows a language model
to retrieve relevant documents from an external knowledge source before
answering a question.

The retrieval process uses embeddings to represent text as vectors.
A vector database stores these embeddings and searches for relevant
information based on the user's query.

After retrieving the relevant documents, the language model uses the
retrieved context to generate an answer. RAG can help improve factual
accuracy and allows a system to use updated external information
without retraining the entire model.

Unlike traditional fine-tuning, RAG retrieves information at inference
time. Fine-tuning changes model weights using additional training data.
Both approaches have advantages and limitations depending on the task.
"""

summary = map_reduce_summarise(long_document)

print("Final Summary:")
print(summary)

