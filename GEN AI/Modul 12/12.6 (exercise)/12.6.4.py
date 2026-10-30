# pip install peft transformers torch datasets accelerate
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

BASE_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = BASE_DIR / "lora_adapter"
MODEL_NAME = "distilgpt2"
MAX_LEN = 80

torch.manual_seed(42)

# ── Dataset: 20 pasangan Q&A (topik: RAG & vector database) ───────────────────
QA_PAIRS = [
    ("What is RAG?", "RAG stands for Retrieval-Augmented Generation."),
    ("What is an embedding?", "An embedding is a dense vector that represents text meaning."),
    ("What is a vector database?", "A vector database stores embeddings for similarity search."),
    ("What is ChromaDB?", "ChromaDB is an open-source vector database for local development."),
    ("What is Pinecone?", "Pinecone is a managed vector database with no infrastructure work."),
    ("What is Weaviate?", "Weaviate is a vector database that supports hybrid search."),
    ("What is pgvector?", "pgvector is a PostgreSQL extension for storing vectors."),
    ("What is cosine similarity?", "Cosine similarity measures the angle between two vectors."),
    ("What is chunking?", "Chunking splits long documents into smaller pieces."),
    ("What is a retriever?", "A retriever finds the documents most relevant to a query."),
    ("What is hybrid search?", "Hybrid search combines BM25 keyword search with vector search."),
    ("What is BM25?", "BM25 is a keyword ranking function based on term frequency."),
    ("What is top-k?", "Top-k is the number of closest results returned by a query."),
    ("What is a hallucination?", "A hallucination is a confident but false answer from a model."),
    ("What is fine-tuning?", "Fine-tuning trains a pre-trained model on new data."),
    ("What is LoRA?", "LoRA trains small low-rank adapters instead of all model weights."),
    ("What is a token?", "A token is a small piece of text that a model reads."),
    ("What is a prompt?", "A prompt is the input text that instructs a language model."),
    ("What is LangChain?", "LangChain is a framework for building LLM applications."),
    ("What is a context window?", "A context window is the maximum number of tokens a model can read."),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def normalize(s: str) -> str:
    return " ".join(s.lower().strip().rstrip(".").split())


def answer(model, question: str) -> str:
    inputs = tokenizer(build_prompt(question), return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,  # greedy: jawaban deterministik, cocok untuk verifikasi
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def evaluate(model, title: str, verbose: bool = True) -> float:
    model.eval()
    correct = 0
    print(f"\n=== {title} ===")
    for q, gold in QA_PAIRS:
        pred = answer(model, q)
        ok = normalize(pred) == normalize(gold)
        correct += ok
        if verbose:
            print(f"[{'OK ' if ok else 'XX '}] {q}\n       -> {pred}")
    acc = correct / len(QA_PAIRS)
    print(f"Akurasi: {correct}/{len(QA_PAIRS)} ({acc:.0%})")
    return acc


# ── 1. Load base model ────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
evaluate(model, "SEBELUM fine-tuning (base model)", verbose=False)

# ── 2. Configure LoRA ─────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=True,  # layer GPT-2 bertipe Conv1D
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# ── 3. Prepare dataset ────────────────────────────────────────────────────────
texts = [f"{build_prompt(q)} {a}{tokenizer.eos_token}" for q, a in QA_PAIRS]


def tokenize(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    # padding tidak ikut dihitung loss (-100); EOS asli tetap dilatih agar model berhenti
    enc["labels"] = [
        [tok if mask == 1 else -100 for tok, mask in zip(ids, masks)]
        for ids, masks in zip(enc["input_ids"], enc["attention_mask"])
    ]
    return enc


tokenized = Dataset.from_dict({"text": texts}).map(tokenize, batched=True)

# ── 4. Training arguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(BASE_DIR / "lora_output"),
    num_train_epochs=60,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    fp16=False,
    logging_steps=25,
    save_strategy="no",  # hemat disk, adapter disimpan manual di langkah 6
    report_to="none",
    seed=42,
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
trainer = Trainer(model=peft_model, args=training_args, train_dataset=tokenized)
trainer.train()

# ── 6. Save adapter only ──────────────────────────────────────────────────────
peft_model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print("LoRA adapter saved.")

# ── 7. Load adapter for inference & verify ────────────────────────────────────
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
finetuned = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
evaluate(finetuned, "SESUDAH fine-tuning (adapter dimuat ulang dari disk)")