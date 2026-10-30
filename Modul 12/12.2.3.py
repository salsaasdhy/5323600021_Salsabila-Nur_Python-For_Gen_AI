from transformers import pipeline

# Summarisation
summariser = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device=-1,
)
article = """
Large language models have fundamentally changed how software is built.
Developers now reach for AI APIs as naturally as they reach for database clients.
The shift from rule-based systems to neural language models has accelerated
since the release of GPT-3 in 2020, and the pace has only increased with
instruction-tuned models like ChatGPT, Claude, and Gemini. These models can
write code, explain concepts, extract information, and reason through complex
problems - tasks that previously required specialised rule-based systems or
significant human effort.
"""
summary = summariser(article, max_length=60, min_length=20, do_sample=False)
print(summary[0]["summary_text"])