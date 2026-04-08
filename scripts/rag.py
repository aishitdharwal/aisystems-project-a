"""
RAG pipeline — Session 4: Re-ranking + Context Engineering.

Full pipeline progression:
  Week 1: ask()          — dense retrieval → generate
  Week 2: ask()          — same API, hybrid mode added
  Session 3: ask_hybrid() — hybrid retrieval (BM25 + dense + RRF)
  Session 4: ask_advanced() — hybrid + rerank + context assembly (dedup + expand + compress)

Session 4 additions:
  - ask_advanced()       — full 4-stage pipeline
  - run_ab_test()        — compare naive vs advanced on your eval set

Run:
    python scripts/rag.py                              # dense (default)
    python scripts/rag.py --mode hybrid               # Session 3 hybrid
    python scripts/rag.py --mode advanced             # Session 4 full pipeline
    python scripts/rag.py --mode advanced --query "..."
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
BM25_CANDIDATES = TOP_K * 3
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
# BM25 + HYBRID (Session 3)
# =========================================================================

def _load_all_chunks():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, doc_name, chunk_index, content, metadata FROM chunks ORDER BY id")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{
        "id": row[0], "doc_name": row[1], "chunk_index": row[2],
        "content": row[3],
        "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
    } for row in rows]


def build_bm25_index():
    all_chunks = _load_all_chunks()
    tokenized = [c["content"].lower().split() for c in all_chunks]
    return BM25Okapi(tokenized), all_chunks


def bm25_retrieve(query, bm25, all_chunks, top_k=BM25_CANDIDATES):
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        if score > 0:
            chunk = all_chunks[idx].copy()
            chunk["bm25_score"] = round(float(score), 4)
            chunk["similarity"] = 0.0
            results.append(chunk)
    return results


def reciprocal_rank_fusion(dense_results, bm25_results, top_k=TOP_K, k=60):
    scores = {}
    chunk_map = {}
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


@observe(name="retrieval_hybrid")
def hybrid_retrieve(query, query_embedding, top_k=TOP_K):
    bm25, all_chunks = build_bm25_index()
    # Call retrieve without the @observe decorator wrapper to avoid double-tracing
    dense = retrieve.__wrapped__(query_embedding, top_k=BM25_CANDIDATES)
    bm25_results = bm25_retrieve(query, bm25, all_chunks, top_k=BM25_CANDIDATES)
    fused = reciprocal_rank_fusion(dense, bm25_results, top_k=top_k)
    langfuse_context.update_current_observation(metadata={
        "mode": "hybrid", "fused": len(fused),
        "results": [{"doc_name": r["doc_name"], "rrf_score": r.get("rrf_score")} for r in fused],
    })
    return fused


# =========================================================================
# CONTEXT ASSEMBLY (shared)
# =========================================================================

@observe(name="context_assembly")
def assemble_context(retrieved_chunks):
    parts = [
        f"[Source: {c['doc_name']}, Chunk {c['chunk_index']}]\n{c['content']}"
        for c in retrieved_chunks
    ]
    context = "\n\n---\n\n".join(parts)
    langfuse_context.update_current_observation(metadata={
        "num_chunks": len(retrieved_chunks), "total_chars": len(context),
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
        usage={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens,
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
        mode: "dense" | "hybrid" | "advanced"
    """
    start_time = time.time()
    langfuse_context.update_current_trace(input=query, metadata={"pipeline": f"rag_{mode}"})

    query_embedding = embed_query(query)

    if mode == "advanced":
        return _ask_advanced(query, query_embedding, start_time)
    elif mode == "hybrid":
        retrieved_chunks = hybrid_retrieve(query, query_embedding)
    else:
        retrieved_chunks = retrieve(query_embedding)

    context = assemble_context(retrieved_chunks)
    answer = generate(query, context)

    elapsed = round(time.time() - start_time, 2)
    langfuse_context.update_current_trace(output=answer, metadata={"elapsed_seconds": elapsed})
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.flush()

    return {
        "query": query, "answer": answer,
        "retrieved_chunks": retrieved_chunks, "context": context,
        "trace_id": trace_id, "elapsed_seconds": elapsed, "mode": mode,
    }


def _ask_advanced(query, query_embedding, start_time):
    """
    Full Session 4 pipeline:
      hybrid retrieve → rerank → dedup + expand + compress → generate
    """
    from reranker import rerank
    from context_assembler import assemble_advanced

    # Stage 1: Hybrid retrieval (wider candidate set for re-ranker)
    candidates = hybrid_retrieve(query, query_embedding, top_k=TOP_K * 2)

    # Stage 2: Re-rank with cross-encoder
    reranked = rerank(query, candidates, top_n=TOP_K + 2)

    # Stage 3: Context assembly (dedup + expand + compress)
    context, final_chunks = assemble_advanced(
        reranked, conn_fn=get_connection, max_chars=4000, expand_window=1
    )

    # Stage 4: Generate
    answer = generate(query, context)

    elapsed = round(time.time() - start_time, 2)
    langfuse_context.update_current_trace(
        output=answer, metadata={"elapsed_seconds": elapsed, "mode": "advanced",
                                 "stages": {"candidates": len(candidates),
                                            "reranked": len(reranked),
                                            "final": len(final_chunks)}}
    )
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.flush()

    return {
        "query": query, "answer": answer,
        "retrieved_chunks": final_chunks, "context": context,
        "trace_id": trace_id, "elapsed_seconds": elapsed, "mode": "advanced",
        "pipeline_stages": {"candidates": len(candidates), "reranked": len(reranked),
                            "final": len(final_chunks)},
    }


def ask_hybrid(query):
    return ask(query, mode="hybrid")


def ask_advanced(query):
    return ask(query, mode="advanced")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "hybrid", "advanced"], default="dense")
    parser.add_argument("--query", default="What is the return window for premium members during Diwali sale?")
    args = parser.parse_args()

    result = ask(args.query, mode=args.mode)
    print(f"\nQuery: {result['query']}")
    print(f"Mode:  {result['mode']}")
    print(f"Time:  {result['elapsed_seconds']}s")
    if "pipeline_stages" in result:
        s = result["pipeline_stages"]
        print(f"Stages: {s['candidates']} candidates → {s['reranked']} reranked → {s['final']} final")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nChunks used:")
    for i, c in enumerate(result["retrieved_chunks"], 1):
        score = c.get("rerank_score", c.get("rrf_score", c.get("similarity", 0)))
        tag = " [expanded]" if c.get("is_expanded") else ""
        print(f"  [{i}] {c['doc_name']} chunk {c['chunk_index']}{tag} — {score:.4f}")
