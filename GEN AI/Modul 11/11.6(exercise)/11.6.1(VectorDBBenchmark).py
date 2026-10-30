
import chromadb
import time
import numpy as np

from embedding_config import embed


class VectorDBBenchmark:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./benchmark_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="benchmark_collection",
            metadata={"hnsw:space": "cosine"}
        )

    # --- Insert 1,000 mock documents ---
    def insert_documents(self, total=1000):
        documents = [
            f"Document {i}: This is mock information "
            f"about artificial intelligence and vector databases."
            for i in range(total)
        ]

        ids = [f"doc_{i}" for i in range(total)]

        embeddings = []

        # Embedding in batches
        for i in range(0, total, 100):
            batch = documents[i:i + 100]
            embeddings.extend(embed(batch))

        # Upsert allows rerunning the benchmark
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )

        print(f"Inserted documents: {self.collection.count()}")

    # --- Run 100 queries and measure latency ---
    def run_queries(self, total_queries=100):
        queries = [
            f"How does artificial intelligence work? Query {i}"
            for i in range(total_queries)
        ]

        # Prepare embeddings before measuring query latency
        query_embeddings = []

        for i in range(0, total_queries, 50):
            batch = queries[i:i + 50]
            query_embeddings.extend(embed(batch))

        latencies = []
        total_results = 0

        for query_embedding in query_embeddings:
            start = time.perf_counter()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "distances"]
            )

            end = time.perf_counter()

            latencies.append(end - start)
            total_results += len(results["ids"][0])

        # --- Calculate benchmark metrics ---
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        total_time = sum(latencies)

        throughput = (
            total_results / total_time
            if total_time > 0 else 0
        )

        print("\n--- VectorDB Benchmark Results ---")
        print(f"Total queries: {total_queries}")
        print(f"Average latency: {avg_latency * 1000:.2f} ms")
        print(f"P95 latency: {p95_latency * 1000:.2f} ms")
        print(f"Results per second: {throughput:.2f}")


# --- Main ---
benchmark = VectorDBBenchmark()

benchmark.insert_documents(total=1000)
benchmark.run_queries(total_queries=100)
