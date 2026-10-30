import time
from transformers import pipeline

class ModelBenchmark:
    """Jalankan prompt yang sama di dua pipeline text-generation, bandingkan latensi & panjang output."""

    def __init__(self, model_a: str = "distilgpt2", model_b: str = "gpt2",
                 max_new_tokens: int = 40):
        self.names = (model_a, model_b)
        self.max_new_tokens = max_new_tokens
        self.pipes = {
            name: pipeline(task="text-generation", model=name, device=-1)
            for name in self.names
        }

    def _run_one(self, name: str, prompt: str) -> dict:
        pipe = self.pipes[name]
        tok = pipe.tokenizer
        start = time.perf_counter()
        out = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # greedy supaya hasil dua kali jalan konsisten
            return_full_text=False,
            pad_token_id=tok.eos_token_id,
        )
        latency = time.perf_counter() - start
        text = out[0]["generated_text"]
        return {
            "latency": latency,
            "tokens": len(tok.encode(text)),
            "chars": len(text),
            "text": text,
        }

    def run(self, prompts: list[str]) -> list[dict]:
        # warm-up: panggilan pertama selalu lebih lambat, jangan ikut dihitung
        for name in self.names:
            self._run_one(name, "Hello")

        rows = []
        for prompt in prompts:
            row = {"prompt": prompt}
            for name in self.names:
                row[name] = self._run_one(name, prompt)
            rows.append(row)
        self.print_table(rows)
        return rows

    def print_table(self, rows: list[dict]):
        a, b = self.names
        header = (f"{'#':>2} {'Prompt':<30} | {a + ' (s)':>14} {'tok':>4} {'chars':>5} "
                  f"| {b + ' (s)':>14} {'tok':>4} {'chars':>5}")
        print(header)
        print("-" * len(header))
        for i, r in enumerate(rows, 1):
            p = r["prompt"][:28] + (".." if len(r["prompt"]) > 28 else "")
            print(f"{i:>2} {p:<30} | {r[a]['latency']:>14.3f} {r[a]['tokens']:>4} {r[a]['chars']:>5} "
                  f"| {r[b]['latency']:>14.3f} {r[b]['tokens']:>4} {r[b]['chars']:>5}")
        print("-" * len(header))
        n = len(rows)
        avg = lambda m, k: sum(r[m][k] for r in rows) / n
        print(f"{'':>2} {'RATA-RATA':<30} | {avg(a, 'latency'):>14.3f} {avg(a, 'tokens'):>4.0f} {avg(a, 'chars'):>5.0f} "
              f"| {avg(b, 'latency'):>14.3f} {avg(b, 'tokens'):>4.0f} {avg(b, 'chars'):>5.0f}")


PROMPTS = [
    "The key advantage of local LLMs is",
    "Retrieval-Augmented Generation works by",
    "A vector database is used for",
    "Fine-tuning a language model means",
    "The difference between training and inference is",
    "Tokenization is the process of",
    "An embedding is",
    "Large language models can be used to",
    "The main risk of hallucination is",
    "To reduce the cost of running an LLM, you can",
]

if __name__ == "__main__":
    ModelBenchmark("distilgpt2", "gpt2").run(PROMPTS)