import numpy as np
import pandas as pd
from pathlib import Path
import re


# ============================================================
# 1. LOAD CSV LLM BENCHMARK
# ============================================================

print("=" * 60)
print("1. LLM BENCHMARK ANALYSIS")
print("=" * 60)

# Folder tempat file Python berada
BASE_DIR = Path(__file__).parent

csv_file = BASE_DIR / "llm_benchmark.csv"


# ------------------------------------------------------------
# Buat CSV contoh jika belum tersedia
# Minimal 20 baris dan 4 model
# ------------------------------------------------------------

if not csv_file.exists():

    benchmark_data = [
        ["GPT-4o", "Question Answering", 0.91, 420],
        ["GPT-4o", "Summarization", 0.88, 450],
        ["GPT-4o", "Classification", 0.94, 390],
        ["GPT-4o", "Reasoning", 0.89, 520],
        ["GPT-4o", "Translation", 0.93, 410],

        ["Claude Sonnet", "Question Answering", 0.93, 450],
        ["Claude Sonnet", "Summarization", 0.92, 470],
        ["Claude Sonnet", "Classification", 0.91, 430],
        ["Claude Sonnet", "Reasoning", 0.94, 500],
        ["Claude Sonnet", "Translation", 0.95, 440],

        ["Gemini 1.5", "Question Answering", 0.89, 380],
        ["Gemini 1.5", "Summarization", 0.90, 400],
        ["Gemini 1.5", "Classification", 0.92, 370],
        ["Gemini 1.5", "Reasoning", 0.87, 460],
        ["Gemini 1.5", "Translation", 0.91, 390],

        ["Llama 3", "Question Answering", 0.86, 330],
        ["Llama 3", "Summarization", 0.85, 350],
        ["Llama 3", "Classification", 0.88, 320],
        ["Llama 3", "Reasoning", 0.84, 410],
        ["Llama 3", "Translation", 0.87, 340],
    ]

    benchmark_df = pd.DataFrame(
        benchmark_data,
        columns=["model", "task", "score", "latency"]
    )

    benchmark_df.to_csv(csv_file, index=False)

    print(f"CSV contoh dibuat: {csv_file}")


# ------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------

df = pd.read_csv(csv_file)

print("\nData benchmark:")
print(df.to_string(index=False))


# ------------------------------------------------------------
# Mean score per model
# ------------------------------------------------------------

mean_score = (
    df.groupby("model")["score"]
    .mean()
    .sort_values(ascending=False)
)

print("\nMean score per model:")
print(mean_score.round(3))


# ------------------------------------------------------------
# Best-performing task per model
# ------------------------------------------------------------

best_task = (
    df.loc[
        df.groupby("model")["score"].idxmax(),
        ["model", "task", "score"]
    ]
    .reset_index(drop=True)
)

print("\nBest-performing task per model:")
print(best_task.to_string(index=False))


# ------------------------------------------------------------
# Correlation score dan latency
# ------------------------------------------------------------

correlation = df["score"].corr(df["latency"])

print(
    f"\nCorrelation antara score dan latency: "
    f"{correlation:.3f}"
)


# ============================================================
# 2. NORMALISE EMBEDDINGS
# ============================================================

print("\n" + "=" * 60)
print("2. EMBEDDING NORMALISATION")
print("=" * 60)


def normalise_embeddings(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalise setiap baris pada matrix.
    """

    matrix = np.asarray(matrix, dtype=float)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    # Hindari pembagian dengan nol
    norms = np.where(norms == 0, 1, norms)

    return matrix / norms


# Contoh matrix embedding
embeddings = np.array([
    [3, 4, 0],
    [1, 2, 2],
    [5, 0, 12],
    [2, 3, 6]
], dtype=float)


normalised = normalise_embeddings(embeddings)

print("\nOriginal embeddings:")
print(embeddings)

print("\nNormalised embeddings:")
print(np.round(normalised, 4))


# Verifikasi bahwa setiap row memiliki norm = 1
row_norms = np.linalg.norm(normalised, axis=1)

print("\nRow norms setelah normalisasi:")
print(np.round(row_norms, 6))

print(
    "\nSemua row norm = 1.0:",
    np.allclose(row_norms, 1.0)
)


# ============================================================
# 3. ANALISIS FILE TXT
# ============================================================

print("\n" + "=" * 60)
print("3. TXT FILE ANALYSIS")
print("=" * 60)


txt_folder = BASE_DIR / "txt_files"

# Buat folder contoh jika belum ada
txt_folder.mkdir(exist_ok=True)


# Buat beberapa file contoh jika folder masih kosong
if not list(txt_folder.glob("*.txt")):

    sample_files = {
        "document1.txt":
            "Artificial intelligence is a technology. "
            "It can process data and learn from examples.",

        "document2.txt":
            "Machine learning is a branch of artificial intelligence. "
            "Models learn patterns from data. "
            "These patterns can be used for prediction.",

        "document3.txt":
            "Deep learning uses neural networks. "
            "Neural networks contain multiple layers.",

        "document4.txt":
            "Natural language processing allows computers "
            "to understand human language.",

        "document5.txt":
            "Computer vision enables computers to understand "
            "images and videos. "
            "It is widely used in artificial intelligence."
    }

    for filename, content in sample_files.items():
        (txt_folder / filename).write_text(
            content,
            encoding="utf-8"
        )

    print("File TXT contoh telah dibuat.")


def analyse_txt_folder(folder_path: str | Path) -> pd.DataFrame:
    """
    Membaca semua file .txt dalam folder dan mengembalikan
    DataFrame dengan kolom:
    filename, char_count, word_count, sentence_count
    """

    folder = Path(folder_path)

    results = []

    for file_path in folder.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        # Jumlah karakter
        char_count = len(text)

        # Jumlah kata
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)

        # Jumlah kalimat
        sentences = re.findall(
            r"[^.!?]+[.!?]+",
            text
        )

        sentence_count = len(sentences)

        results.append({
            "filename": file_path.name,
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count
        })

    result_df = pd.DataFrame(results)

    # Sort berdasarkan jumlah kata terbesar
    result_df = result_df.sort_values(
        by="word_count",
        ascending=False
    ).reset_index(drop=True)

    return result_df


txt_df = analyse_txt_folder(txt_folder)

print("\nHasil analisis file TXT:")
print(txt_df.to_string(index=False))


# ============================================================
# 4. PAIRWISE COSINE SIMILARITY
# ============================================================

print("\n" + "=" * 60)
print("4. COSINE SIMILARITY")
print("=" * 60)


corpus = [
    "Artificial intelligence is transforming technology.",
    "Artificial intelligence is changing modern technology.",
    "Cats are common household animals.",
    "Dogs are popular household pets.",
    "Machine learning is a part of artificial intelligence."
]


# ------------------------------------------------------------
# Hash-based mock embeddings
# ------------------------------------------------------------

def mock_embedding(
    text: str,
    dimension: int = 128
) -> np.ndarray:
    """
    Membuat embedding sederhana berbasis hash.
    Digunakan hanya untuk simulasi.
    """

    seed = abs(hash(text)) % (2**32)

    rng = np.random.default_rng(seed)

    vector = rng.normal(
        size=dimension
    )

    return vector


# Buat embedding untuk setiap string
embeddings = np.array([
    mock_embedding(text)
    for text in corpus
])


# Normalisasi embedding
embeddings = normalise_embeddings(embeddings)


# ------------------------------------------------------------
# Cosine similarity matrix
# ------------------------------------------------------------

similarity_matrix = embeddings @ embeddings.T


print("\nCosine similarity matrix:")
print(
    np.round(similarity_matrix, 4)
)


# ------------------------------------------------------------
# Cari pasangan dengan similarity tertinggi
# ------------------------------------------------------------

highest_similarity = -1
best_pair = None

for i in range(len(corpus)):

    for j in range(i + 1, len(corpus)):

        similarity = similarity_matrix[i, j]

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_pair = (i, j)


print("\nPasangan dengan similarity tertinggi:")
print(f"String 1: {corpus[best_pair[0]]}")
print(f"String 2: {corpus[best_pair[1]]}")
print(f"Similarity: {highest_similarity:.4f}")


# ============================================================
# SELESAI
# ============================================================

print("\n" + "=" * 60)
print("SEMUA TUGAS SELESAI")
print("=" * 60)
