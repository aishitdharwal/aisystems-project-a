"""
Ingest documents into pgvector.

Session 3 update: supports multiple chunking strategies.
A/B test them against each other using the eval harness.

Run:
    python scripts/ingest.py                          # fixed_size (Week 1 default)
    python scripts/ingest.py --strategy sentence_aware
    python scripts/ingest.py --strategy sliding_window
    python scripts/ingest.py --compare               # show chunk stats for all strategies
"""
import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from chunker import chunk_document, compare_strategies, STRATEGIES

load_dotenv()

client = OpenAI()

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")


def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5433"),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD", "workshop123"),
        dbname=os.getenv("PG_DATABASE", "acmera_kb"),
    )
    register_vector(conn)
    return conn


def embed_texts(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


def ingest(strategy="fixed_size"):
    print(f"Ingesting with strategy: {strategy}\n")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks;")

    doc_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
    total_chunks = 0

    for filepath in doc_files:
        doc_name = os.path.basename(filepath)
        with open(filepath, "r") as f:
            content = f.read()

        chunks = chunk_document(content, strategy=strategy)
        print(f"  {doc_name}: {len(chunks)} chunks")

        for batch_start in range(0, len(chunks), 20):
            batch = chunks[batch_start:batch_start + 20]
            embeddings = embed_texts(batch)

            for i, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                chunk_index = batch_start + i
                metadata = json.dumps({
                    "doc_name": doc_name,
                    "chunk_index": chunk_index,
                    "strategy": strategy,
                })
                cur.execute(
                    """INSERT INTO chunks (doc_name, chunk_index, content, embedding, metadata)
                       VALUES (%s, %s, %s, %s::vector, %s)""",
                    (doc_name, chunk_index, chunk, embedding, metadata),
                )

        total_chunks += len(chunks)

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone: {len(doc_files)} documents, {total_chunks} chunks (strategy: {strategy})")


def show_comparison():
    """Print chunk count and size stats for each strategy across all docs."""
    doc_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
    totals = {s: {"count": 0, "chars": 0} for s in STRATEGIES}

    for filepath in doc_files:
        with open(filepath) as f:
            text = f.read()
        stats = compare_strategies(text)
        for strategy, s in stats.items():
            totals[strategy]["count"] += s["count"]
            totals[strategy]["chars"] += s["count"] * s["avg_size"]

    print("\nChunking strategy comparison across corpus:\n")
    print(f"  {'Strategy':<20} {'Total chunks':>13} {'Avg chars':>10}")
    print(f"  {'-'*20} {'-'*13} {'-'*10}")
    for strategy, totals_s in totals.items():
        avg = totals_s["chars"] // totals_s["count"] if totals_s["count"] else 0
        marker = "  ← Week 1 baseline" if strategy == "fixed_size" else ""
        print(f"  {strategy:<20} {totals_s['count']:>13} {avg:>10}{marker}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest corpus into pgvector")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="fixed_size",
        help="Chunking strategy (default: fixed_size)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show chunk statistics for all strategies without ingesting",
    )
    args = parser.parse_args()

    if args.compare:
        show_comparison()
    else:
        ingest(strategy=args.strategy)
