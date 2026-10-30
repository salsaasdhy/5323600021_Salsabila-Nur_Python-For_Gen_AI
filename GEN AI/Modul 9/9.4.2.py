
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="openai/gpt-4o", api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", max_tokens=500)

# Run two chains in parallel and combine results
summarise_prompt = ChatPromptTemplate.from_messages([
("system", "Summarise this in one sentence."),
("human", "{text}"),])

keywords_prompt = ChatPromptTemplate.from_messages([
("system", "Extract 3 keywords from this text. Return them comma-separated."),
("human", "{text}"),])

parser = StrOutputParser()

parallel_chain = RunnableParallel(
summary = summarise_prompt | llm | parser,
keywords = keywords_prompt | llm | parser,
)

result = parallel_chain.invoke({
"text": "Retrieval-Augmented Generation (RAG) improves LLM answers by fetching relevant documents at inference time."
})

print("Summary: ", result["summary"])
print("Keywords:", result["keywords"])
