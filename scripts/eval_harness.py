"""
Evaluation Harness — Sessions 1 & 2 Starter

SESSION 1 functions (implement during Session 1 homework):
  1. check_retrieval_hit() — is the expected source in the top-K results?
  2. calculate_mrr() — how high is the first relevant chunk ranked?
  3. judge_faithfulness() — is the answer grounded in the context? (LLM-as-judge)
  4. judge_correctness() — does the answer match the expected answer? (LLM-as-judge)
  5. run_eval() — orchestrate everything and produce a scorecard

SESSION 2 functions (implement during Session 2 homework):
  6. run_stratified_eval() — break down scores by category and difficulty
  7. attach_langfuse_scores() — attach eval scores to LangFuse traces
  8. save_baseline() — save current scores as baseline_scores.json

Run: python scripts/eval_harness.py
Run with options:
  python scripts/eval_harness.py --include-hard
  python scripts/eval_harness.py --save-baseline
  python scripts/eval_harness.py --category membership
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

SCRIPT_DIR = os.path.dirname(__file__)


# =========================================================================
# GOLDEN DATASET
# =========================================================================

def load_golden_dataset():
    """Load the golden dataset from JSON file."""
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(path):
        print("No golden_dataset.json found. Create one first!")
        return []
    with open(path) as f:
        return json.load(f)


# =========================================================================
# SESSION 1: RETRIEVAL METRICS
# =========================================================================

def check_retrieval_hit(retrieved_chunks, expected_source):
    """
    Is the expected source document in the retrieved chunks?
    Returns True/False.
    """
    if expected_source == "N/A":
        return True
    return any(c["doc_name"] == expected_source for c in retrieved_chunks)


def calculate_mrr(retrieved_chunks, expected_source):
    """
    Mean Reciprocal Rank — how high is the first relevant chunk?
    Position 1 → 1.0, Position 3 → 0.33, Not found → 0.0

    Formula: 1 / rank_of_first_relevant_chunk
    """
    if expected_source == "N/A":
        return 1.0
    for i, chunk in enumerate(retrieved_chunks):
        if chunk["doc_name"] == expected_source:
            return round(1.0 / (i + 1), 4)
    return 0.0


# =========================================================================
# SESSION 1: GENERATION METRICS (LLM-as-Judge)
# =========================================================================

def judge_faithfulness(query, answer, context):
    """
    Is the answer grounded in the retrieved context?
    Uses GPT-4o-mini as a judge with a structured rubric.
    Returns: {"score": 1-5, "reason": "explanation"}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are a strict evaluation judge. Assess whether the answer is grounded in the context.

Score 1-5:
5 = Fully grounded — every claim explicitly supported
4 = Mostly grounded — 1 minor inference
3 = Partially grounded — some claims not in context
2 = Poorly grounded — significant claims missing, or PII/internal data revealed
1 = Not grounded — fabricated information

Score 2 or lower if answer reveals customer PII or internal company data.
Respond ONLY with JSON: {"score": N, "reason": "brief explanation"}"""
            },
            {"role": "user", "content": f"QUERY: {query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"}
        ],
    )
    try:
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"score": 3, "reason": "parse error"}


def judge_correctness(query, answer, expected_answer):
    """
    Does the answer match the expected answer?
    Uses GPT-4o-mini as a judge.
    Returns: {"score": 1-5, "reason": "explanation"}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are a strict evaluation judge. Compare generated answer against expected answer.

Score 1-5:
5 = Perfect — all key points accurate
4 = Good — most points, minor omissions
3 = Partial — some points missing
2 = Poor — misses most points or significant errors
1 = Wrong — contradicts expected or reveals sensitive data

Respond ONLY with JSON: {"score": N, "reason": "brief explanation"}"""
            },
            {"role": "user", "content": f"QUERY: {query}\n\nEXPECTED:\n{expected_answer}\n\nGENERATED:\n{answer}"}
        ],
    )
    try:
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"score": 3, "reason": "parse error"}


# =========================================================================
# SESSION 1: EVAL RUNNER
# =========================================================================

def run_eval(include_hard=False):
    """
    Run the full evaluation:
    1. Load golden dataset (+ hard queries if --include-hard)
    2. Run each query through the RAG pipeline via ask()
    3. Score retrieval (hit rate, MRR)
    4. Score generation (faithfulness, correctness)
    5. Print scorecard
    6. Save results to eval_results.json
    """
    from rag import ask

    dataset = load_golden_dataset()
    if not dataset:
        print("No golden dataset found.")
        return

    results = []
    retrieval_hits = 0
    mrr_scores = []
    faithfulness_scores = []
    correctness_scores = []

    for i, q in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {q['query'][:60]}...")
        result = ask(q["query"])

        hit = check_retrieval_hit(result["retrieved_chunks"], q["expected_source"])
        mrr = calculate_mrr(result["retrieved_chunks"], q["expected_source"])
        if hit:
            retrieval_hits += 1
        mrr_scores.append(mrr)

        faith = judge_faithfulness(q["query"], result["answer"], result["context"])
        faithfulness_scores.append(faith["score"])

        correct = judge_correctness(q["query"], result["answer"], q["expected_answer"])
        correctness_scores.append(correct["score"])

        results.append({
            "id": q["id"],
            "query": q["query"],
            "difficulty": q.get("difficulty", "easy"),
            "category": q.get("category", "general"),
            "retrieval_hit": hit,
            "mrr": mrr,
            "faithfulness_score": faith["score"],
            "faithfulness_reason": faith["reason"],
            "correctness_score": correct["score"],
            "correctness_reason": correct["reason"],
            "answer": result["answer"],
            "trace_id": result.get("trace_id"),
        })

    total = len(dataset)
    hit_rate = retrieval_hits / total * 100
    avg_mrr = sum(mrr_scores) / total * 100
    avg_faith = sum(faithfulness_scores) / total / 5 * 100
    avg_correct = sum(correctness_scores) / total / 5 * 100

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total queries:       {total}")
    print(f"Retrieval hit rate:  {hit_rate:.1f}%")
    print(f"MRR:                 {avg_mrr:.1f}%")
    print(f"Faithfulness:        {avg_faith:.1f}%")
    print(f"Correctness:         {avg_correct:.1f}%")
    print("="*50)

    output = {
        "summary": {
            "total_queries": total,
            "retrieval_hit_rate": round(hit_rate, 1),
            "avg_mrr": round(avg_mrr, 1),
            "avg_faithfulness": round(avg_faith, 1),
            "avg_correctness": round(avg_correct, 1),
        },
        "results": results,
    }
    with open(os.path.join(SCRIPT_DIR, "..", "eval_results.json"), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nResults saved to eval_results.json")


# =========================================================================
# SESSION 2: STRATIFIED EVALUATION
# =========================================================================

def run_stratified_eval(results):
    """
    Break down eval scores by category and by difficulty.

    For categories: group results by result["category"], compute
    hit_rate, faithfulness, correctness per group, print a table.

    For difficulty: group by result["difficulty"] (easy/medium/hard),
    compute correctness per group, print a table.

    The key insight: 87% overall might hide 40% on membership queries.
    Stratification surfaces this.

    TODO: Implement in Session 2 homework.
    """
    pass


# =========================================================================
# SESSION 2: LANGFUSE SCORE ATTACHMENT
# =========================================================================

def attach_langfuse_scores(trace_id, faithfulness_result, correctness_result, retrieval_hit):
    """
    Attach eval scores to a LangFuse trace so they're queryable in the dashboard.

    Use langfuse.score() with:
      - name="faithfulness", value=faithfulness_result["score"] / 5
      - name="correctness", value=correctness_result["score"] / 5
      - name="retrieval_hit", value=1.0 if retrieval_hit else 0.0

    After attaching, you can filter in LangFuse:
    "Show me all traces where faithfulness < 0.6"

    TODO: Implement in Session 2 homework.
    """
    pass


# =========================================================================
# SESSION 2: SAVE BASELINE
# =========================================================================

def save_baseline(summary_scores, category_breakdown):
    """
    Save current eval scores as baseline_scores.json.
    This becomes the regression anchor — future evals compare against it.

    summary_scores should include: retrieval_hit_rate, avg_faithfulness, avg_correctness
    category_breakdown: per-category correctness scores

    TODO: Implement in Session 2 homework.
    """
    pass


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-hard", action="store_true",
                        help="Include hard queries that expose system failures")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current scores as baseline_scores.json")
    parser.add_argument("--category", type=str,
                        help="Filter to a specific category (e.g. 'membership')")
    args = parser.parse_args()

    run_eval(include_hard=args.include_hard)
