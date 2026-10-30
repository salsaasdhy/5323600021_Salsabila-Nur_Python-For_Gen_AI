from transformers import pipeline

# Named entity recognition
ner = pipeline(
    task="ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
    device=-1,
)

text = "Anthropic was founded in San Francisco by Dario Amodei and Daniela Amodei."
entities = ner(text)

for ent in entities:
    print(f"{ent['word']!r:25s} →{ent['entity_group']} ({ent['score']:.3f})")

from transformers import pipeline

# Text generation (small local model)
generator = pipeline(
    task="text-generation",
    model="distilgpt2",  # small model - runs on CPU
    device=-1,
)

output = generator(
    "The future of AI agents is",
    max_new_tokens=50,
    num_return_sequences=2,
    temperature=0.8,
    do_sample=True,
    pad_token_id=50256,  # suppress warning for GPT-2
)

for i, seq in enumerate(output):
    print(f"Sequence{i+1}:{seq['generated_text']}")