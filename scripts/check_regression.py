"""
Regression Checker — Session 2 Starter

Compares current eval scores against a saved baseline.
Flags any metric that drops more than the threshold.

Functions to implement:
  1. load_baseline() — load baseline_scores.json
  2. load_current() — load eval_results.json (note: scores are under "summary" key)
  3. check_regression() — compare metric by metric, return list of regressions
  4. display_results() — print a clear pass/fail table with deltas

Run: python scripts/check_regression.py
Run with options:
  python scripts/check_regression.py --threshold 3.0
  python scripts/check_regression.py --baseline scripts/baseline_scores.json
"""
import os
import sys
import json
import argparse

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = os.path.dirname(__file__)

DEFAULT_BASELINE = os.path.join(SCRIPT_DIR, "baseline_scores.json")
DEFAULT_CURRENT = os.path.join(SCRIPT_DIR, "..", "eval_results.json")
DEFAULT_THRESHOLD = 5.0  # percentage points


# =========================================================================
# FUNCTIONS TO IMPLEMENT IN SESSION 2
# =========================================================================

def load_baseline(path: str) -> dict:
    """
    Load baseline scores from JSON file.
    Returns the parsed dict.

    TODO: Implement in Session 2.
    """
    pass


def load_current(path: str) -> dict:
    """
    Load current eval results from JSON file.
    Note: eval_results.json wraps scores inside a "summary" key.
    If "summary" is present, return data["summary"]. Otherwise return data directly.

    TODO: Implement in Session 2.
    """
    pass


def check_regression(current: dict, baseline: dict, threshold: float = DEFAULT_THRESHOLD) -> list:
    """
    Compare current scores against baseline for these metrics:
      - retrieval_hit_rate
      - avg_faithfulness
      - avg_correctness

    For each metric:
      - Calculate delta = current[metric] - baseline[metric]
      - Mark as regression if delta < -threshold

    Returns a list of dicts:
      [{"metric": "...", "baseline": N, "current": N, "delta": N, "is_regression": bool}]

    TODO: Implement in Session 2.
    """
    pass


def display_results(regressions: list, threshold: float):
    """
    Print a clear comparison table showing baseline vs current for each metric.
    Show PASS (green) or REGRESSION (red) for each.
    Print a headline: ✅ NO REGRESSION or ❌ REGRESSION DETECTED.

    TODO: Implement in Session 2.
    """
    pass


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Regression checker for RAG eval")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE)
    parser.add_argument("--current", type=str, default=DEFAULT_CURRENT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Regression threshold in percentage points (default: 5.0)")
    args = parser.parse_args()

    print("Regression checker skeleton loaded.")
    print("Functions to implement: load_baseline, load_current, check_regression, display_results")
    print("\nWe'll build these together in Session 2.")


if __name__ == "__main__":
    main()
