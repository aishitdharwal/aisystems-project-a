"""
RAG pipeline — Session 3 update: Hybrid Search (BM25 + Dense + RRF).

New in Session 3:
  - build_bm25_index()    — loads all chunks, builds in-memory BM25 index
  - bm25_retrieve()       — lexical retrieval using BM25Okapi
  - reciprocal_rank_fusion() — combines dense + BM25 rankings (RRF formula)
  - hybrid_retrieve()     — full hybrid pipeline
  - ask() now accepts mode="dense" (default) or mode="hybrid"
  - ask_hybrid()          — convenience wrapper for hybrid mode

Why hybrid? Dense embeddings are strong on semantics but miss exact keyword
matches. BM25 is strong on keywords but blind to meaning. RRF fusion gets both.

Run:
    python scripts/rag.py                          # dense (default)
    python scripts/rag.py --mode hybrid            # hybrid
"""
import os
import sys
import json
import time
import argparse
from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import psycopg2
from pgvector.psycopg2 import register_vector
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

client = OpenAI()
langfuse = Langfuse()

TOP_K = 5
BM25_CANDIDATES = TOP_K * 3   # retrieve more candidates before RRF fusion
GENERATION_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful customer support assistant for Acmera, an Indian e-commerce company.
Answer the customer's question based on the provided context from our documentation.

Rules:
- Only answer based on the provided context. If the context doesn't contain enough information, say so.
- Be specific and cite relevant policy details (days, amounts, conditions).
- If the question involves membership tiers, check the context for tier-specific policies.
- Be concise but thorough.

Context from Acmera documentation:
{context}"""


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


# =========================================================================
# DENSE RETRIEVAL (Week 1 baseline)
# =========================================================================

@observe(name="query_embedding")
def embed_query(query):
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    return response.data[0].embedding


@observe(name="retrieval_dense")
def retrieve(query_embedding, top_k=TOP_K):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, doc_name, chunk_index, content, metadata,
                  1 - (embedding <=> %s::vector) AS similarity
           FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s""",
        (query_embedding, query_embedding, top_k),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0], "doc_name": row[1], "chunk_index": row[2],
            "content": row[3],
            "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            "similarity": round(float(row[5]), 4),
        })
    cur.close()
    conn.close()

    langfuse_context.update_current_observation(metadata={
        "mode": "dense", "top_k": top_k,
        "results": [{"doc_name": r["doc_name"], "similarity": r["similarity"]} for r in results],
    })
    return results


# =========================================================================
# BM25 RETRIEVAL (Session 3)
# =========================================================================

def _load_all_chunks():
    """Fetch all chunks from DB for BM25 index construction."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, doc_name, chunk_index, content, metadata FROM chunks ORDER BY id")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    chunks = []
    for row in rows:
        chunks.append({
            "id": row[0], "doc_name": row[1], "chunk_index": row[2],
            "content": row[3],
            "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
        })
    return chunks


def build_bm25_index():
    """
    Load all chunks from DB and build an in-memory BM25 index.

    Returns:
        (bm25, all_chunks) — the BM25 index and the full chunk list it maps to
    """
    all_chunks = _load_all_chunks()
    tokenized_corpus = [chunk["content"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, all_chunks


def bm25_retrieve(query, bm25, all_chunks, top_k=BM25_CANDIDATES):
    """
    Score all chunks with BM25 and return the top_k.
    Adds 'bm25_score' key to each result dict.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    results = []
    for idx, score in ranked:
        if score > 0:
            chunk = all_chunks[idx].copy()
            chunk["bm25_score"] = round(float(score), 4)
            chunk["similarity"] = 0.0  # filled in if also in dense results
            results.append(chunk)

    return results


# =========================================================================
# RECIPROCAL RANK FUSION (Session 3)
# =========================================================================

def reciprocal_rank_fusion(dense_results, bm25_results, top_k=TOP_K, k=60):
    """
    Merge dense and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF score = Σ 1 / (k + rank_in_list)

    k=60 is the standard constant that dampens the impact of very high ranks.
    A chunk that's #1 in both lists scores much higher than #1 in one list only.
    """
    scores = {}  # chunk id → accumulated RRF score
    chunk_map = {}  # chunk id → chunk dict

    for rank, chunk in enumerate(dense_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for cid, rrf_score in ranked:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = round(rrf_score, 6)
        results.append(chunk)

    return results


# =========================================================================
# HYBRID RETRIEVE (Session 3)
# =========================================================================

@observe(name="retrieval_hybrid")
def hybrid_retrieve(query, query_embedding, top_k=TOP_K):
    """
    Full hybrid pipeline:
      1. Dense retrieval (3× top_k candidates)
      2. BM25 retrieval (3× top_k candidates)
      3. RRF fusion → top_k final results
    """
    bm25, all_chunks = build_bm25_index()

    dense_results = retrieve.__wrapped__(query_embedding, top_k=BM25_CANDIDATES)
    bm25_results = bm25_retrieve(query, bm25, all_chunks, top_k=BM25_CANDIDATES)
    fused = reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_k)

    langfuse_context.update_current_observation(metadata={
        "mode": "hybrid",
        "dense_candidates": len(dense_results),
        "bm25_candidates": len(bm25_results),
        "fused_results": len(fused),
        "results": [{"doc_name": r["doc_name"], "rrf_score": r.get("rrf_score")} for r in fused],
    })
    return fused


# =========================================================================
# CONTEXT ASSEMBLY
# =========================================================================

@observe(name="context_assembly")
def assemble_context(retrieved_chunks):
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['doc_name']}, Chunk {chunk['chunk_index']}]\n{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    langfuse_context.update_current_observation(metadata={
        "num_chunks": len(retrieved_chunks),
        "total_context_chars": len(context),
    })
    return context


# =========================================================================
# GENERATION
# =========================================================================

@observe(name="generation")
def generate(query, context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model=GENERATION_MODEL, messages=messages, temperature=0, max_tokens=1000,
    )
    answer = response.choices[0].message.content
    langfuse_context.update_current_observation(
        input=messages, output=answer,
        metadata={"model": GENERATION_MODEL,
                  "prompt_tokens": response.usage.prompt_tokens,
                  "completion_tokens": response.usage.completion_tokens},
        usage={"input": response.usage.prompt_tokens,
               "output": response.usage.completion_tokens,
               "total": response.usage.total_tokens, "unit": "TOKENS"},
    )
    return answer


# =========================================================================
# PUBLIC API
# =========================================================================

@observe(name="rag_pipeline")
def ask(query, mode="dense"):
    """
    Ask a question through the RAG pipeline.

    Args:
        query: The user question
        mode: "dense" (default, Week 1 baseline) or "hybrid" (Session 3)

    Returns:
        dict with query, answer, retrieved_chunks, context, trace_id, elapsed_seconds
    """
    start_time = time.time()
    langfuse_context.update_current_trace(
        input=query, metadata={"pipeline": f"rag_{mode}", "top_k": TOP_K}
    )

    query_embedding = embed_query(query)

    if mode == "hybrid":
        retrieved_chunks = hybrid_retrieve(query, query_embedding)
    else:
        retrieved_chunks = retrieve(query_embedding)

    context = assemble_context(retrieved_chunks)
    answer = generate(query, context)

    elapsed = round(time.time() - start_time, 2)
    langfuse_context.update_current_trace(
        output=answer, metadata={"elapsed_seconds": elapsed, "mode": mode}
    )
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.flush()

    return {
        "query": query, "answer": answer,
        "retrieved_chunks": retrieved_chunks, "context": context,
        "trace_id": trace_id, "elapsed_seconds": elapsed, "mode": mode,
    }


def ask_hybrid(query):
    """Convenience wrapper — hybrid retrieval mode."""
    return ask(query, mode="hybrid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    parser.add_argument("--query", default="What is the standard return window for products?")
    args = parser.parse_args()

    result = ask(args.query, mode=args.mode)
    print(f"\nQuery: {result['query']}")
    print(f"Mode:  {result['mode']}")
    print(f"Answer: {result['answer']}")
    print(f"Trace: {result['trace_id']}")
    print(f"Time: {result['elapsed_seconds']}s")
    for i, c in enumerate(result["retrieved_chunks"]):
        score_key = "rrf_score" if "rrf_score" in c else "similarity"
        print(f"  [{i+1}] {c['doc_name']} (chunk {c['chunk_index']}) — {score_key}: {c.get(score_key)}")
