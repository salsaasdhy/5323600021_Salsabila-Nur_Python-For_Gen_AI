from transformers import AutoTokenizer


class TokenBudgetChecker:
    """Buang teks yang melebihi batas token, kembalikan sisanya beserta jumlah token."""

    def __init__(self, model_name: str = "gpt2", max_tokens: int = 50):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.rejected: list[tuple[str, int]] = []

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def filter(self, texts: list[str], max_tokens: int | None = None) -> list[tuple[str, int]]:
        limit = self.max_tokens if max_tokens is None else max_tokens
        kept, self.rejected = [], []
        for text in texts:
            n = self.count(text)
            (kept if n <= limit else self.rejected).append((text, n))
        return kept


if __name__ == "__main__":
    texts = [
        "Short text.",
        "RAG retrieves documents and adds them to the prompt before generation.",
        "word " * 200,
        "Vector databases store embeddings for similarity search.",
    ]
    checker = TokenBudgetChecker("gpt2", max_tokens=20)
    kept = checker.filter(texts)

    print(f"Batas: {checker.max_tokens} token")
    print("Lolos:")
    for text, n in kept:
        print(f"  [{n:>3} token] {text[:60]}")
    print("Dibuang:")
    for text, n in checker.rejected:
        print(f"  [{n:>3} token] {text[:60]}")