"""
Synthetic Question Generator — Session 2 Starter

Generates evaluation questions from corpus documents using GPT-4o-mini.
We'll build this together in Session 2.

Functions to implement:
  1. generate_questions() — call GPT-4o-mini with a persona prompt, return parsed questions
  2. assign_ids() — give sequential IDs that don't clash with existing dataset
  3. critique_questions() — auto-rate each question for realism and difficulty
  4. main() — wire everything together with CLI args

Run: python scripts/synthetic_generator.py
Run with options:
  python scripts/synthetic_generator.py --doc 02_premium_membership.md
  python scripts/synthetic_generator.py --persona frustrated
  python scripts/synthetic_generator.py --merge
"""
import os
import sys
import json
import argparse
import glob

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

SCRIPT_DIR = os.path.dirname(__file__)
CORPUS_DIR = os.path.join(SCRIPT_DIR, "..", "corpus")

# =========================================================================
# PERSONA PROMPTS
# =========================================================================
# Three personas — each reveals different failure modes in the system:
#
# standard   — Clean customer language, direct questions.
#              Reveals: basic retrieval gaps and generation errors.
#
# frustrated — Emotional, informal, vocabulary doesn't match docs.
#              Reveals: vocabulary mismatch failures. Real users sound like this.
#
# mismatch   — Deliberately avoids all vocabulary from the source document.
#              Reveals: whether embeddings can handle true paraphrase.
# =========================================================================

PERSONA_PROMPTS = {
    "standard": """You are generating evaluation questions for a customer support RAG system.

Given the documentation below, generate {count} customer support questions.

Rules:
- Use natural customer language — NOT the exact phrasing from the document
- Each question should be answerable from this document alone
- Mix difficulty: some lookups, some requiring careful reading
- expected_answer should be specific and include key details (numbers, dates, conditions)
- Use category names from: returns, shipping, payments, warranty, membership, orders, products, account, rewards, promotions, sustainability, business

Document: {doc_name}
Content:
{doc_text}

Respond ONLY with a valid JSON array. Each object must have:
  id, query, expected_answer, expected_source, difficulty (easy/medium/hard), category""",

    "frustrated": """You are generating adversarial evaluation questions from FRUSTRATED customers.

These customers are upset, under time pressure, or venting emotionally.
Language is informal and does NOT match the documentation vocabulary.

Document: {doc_name}
Content:
{doc_text}

Generate {count} questions a frustrated customer might actually ask.

Examples of frustrated phrasing:
- "why cant i return this thing already" instead of "what is the return window"
- "they charged me wrong" instead of "refund process"

Rules:
- NO documentation vocabulary in the queries
- Questions must still be answerable from this document
- expected_answer should be calm and correct

Respond ONLY with a valid JSON array. Each object must have:
  id, query, expected_answer, expected_source, difficulty (easy/medium/hard), category""",

    "mismatch": """You are generating vocabulary-mismatch evaluation questions for a RAG stress test.

Use completely DIFFERENT vocabulary than the source document.
Same meaning, different words. Tests whether embeddings handle paraphrase.

Document: {doc_name}
Content:
{doc_text}

Generate {count} questions where vocabulary does NOT match the document.

Examples:
- "send it back" instead of "return"
- "get my cash" instead of "refund"
- "loyalty tiers" instead of "Premium membership levels"

Respond ONLY with a valid JSON array. Each object must have:
  id, query, expected_answer, expected_source, difficulty (easy/medium/hard), category"""
}


# =========================================================================
# FUNCTIONS TO IMPLEMENT IN SESSION 2
# =========================================================================

def generate_questions(doc_name: str, doc_text: str, persona: str = "standard", count: int = 5) -> list:
    """
    Generate synthetic questions for a single document using the specified persona.

    Steps:
    1. Select the right prompt from PERSONA_PROMPTS using the persona key
    2. Format the prompt with doc_name, doc_text[:3000], and count
    3. Call GPT-4o-mini with temperature=0.8 (higher = more varied questions)
    4. Strip ```json fences and parse the JSON response
    5. Add expected_source = doc_name and persona = persona to each question
    6. Return the list of question dicts

    TODO: Implement in Session 2.
    """
    pass


def assign_ids(questions: list, existing_dataset: list, prefix: str = "s") -> list:
    """
    Assign sequential IDs (e.g. s001, s002) that don't clash with existing dataset IDs.

    Steps:
    1. Collect all existing IDs into a set
    2. For each question, find the next available ID with the given prefix
    3. Set question["id"] = that ID
    4. Return the updated list

    TODO: Implement in Session 2.
    """
    pass


def load_golden_dataset() -> list:
    """Load the golden dataset from JSON file."""
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_golden_dataset(dataset: list):
    """Save the golden dataset back to JSON file."""
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dataset)} entries to golden_dataset.json")


def critique_questions(questions: list) -> list:
    """
    Use GPT-4o-mini to rate each question on realism and difficulty.
    Adds a 'critique' field to each question dict.

    Judge prompt should ask for:
      - realism (1-5): does it sound like a real customer?
      - difficulty_actual (1-5): how hard is it for the RAG system?
      - flag: "keep" / "rewrite" / "drop"
      - note: one line explanation

    Returns: {"score": N, "reason": "..."}

    TODO: Implement in Session 2 (stretch).
    """
    pass


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Synthetic question generator for RAG eval")
    parser.add_argument("--doc", type=str, help="Specific corpus doc (e.g. 02_premium_membership.md)")
    parser.add_argument("--all-docs", action="store_true", help="Generate from all corpus documents")
    parser.add_argument("--persona", type=str, default="standard",
                        choices=["standard", "frustrated", "mismatch"])
    parser.add_argument("--count", type=int, default=5, help="Questions per document")
    parser.add_argument("--critique", action="store_true", help="Auto-critique generated questions")
    parser.add_argument("--merge", action="store_true", help="Merge into golden_dataset.json")
    args = parser.parse_args()

    print("Synthetic generator skeleton loaded.")
    print("Functions to implement: generate_questions, assign_ids, critique_questions")
    print("\nWe'll build these together in Session 2.")


if __name__ == "__main__":
    main()
