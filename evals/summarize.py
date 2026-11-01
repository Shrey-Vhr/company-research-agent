"""Milestone 5 summary: scores the saved run, joins your hand grades, prints and saves two Markdown tables.

Run from the project root:
    python -m evals.summarize
"""

import csv
import json
import re
import sys
from pathlib import Path

from evals.run_evals import GRADING_IDS, QUESTIONS_FILE, RESULTS_DIR, VERSIONS, score_numbers

STAMP = "2026-10-30_192104"
RUN_FILE = RESULTS_DIR / f"run_{STAMP}.json"
GRADING_FILE = RESULTS_DIR / f"grading_{STAMP}.csv"
SUMMARY_FILE = RESULTS_DIR / f"summary_{STAMP}.md"

# FILL THESE IN: price in dollars per 1 million tokens
INPUT_PRICE_PER_M = 0.1
OUTPUT_PRICE_PER_M = 0.5

N_NUMBER = 11  # q01-q11 are scored by score_numbers
N_TEXT = 9  # q12-q20 are graded by hand


PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")


def score_numbers_percent_ok(answer, item):
    """Like score_numbers, but a ratio answer written as '22.9%' also counts as 0.229.
    Only for items with a plain-ratio tolerance (no 'unit'); percentage-point items (q07, q08, q11) stay strict."""
    if "unit" in item["tolerance"]:
        return score_numbers(answer, item)
    as_ratios = " ".join(str(float(x) / 100) for x in PERCENT.findall(answer))
    return score_numbers(answer + " " + as_ratios, item)


def load_passes(questions_by_id, results):
    """Return (strict, lenient), each {(id, version): True/False}.
    q01-q11: score_numbers (strict) and score_numbers_percent_ok (lenient). q12-q20: your 'pass' column in both."""
    passes, lenient = {}, {}
    for r in results:
        if r["id"] not in GRADING_IDS:
            item = questions_by_id[r["id"]]
            key = (r["id"], r["version"])
            passes[key] = bool(r["answer"]) and score_numbers(r["answer"], item)
            lenient[key] = bool(r["answer"]) and score_numbers_percent_ok(r["answer"], item)
    with open(GRADING_FILE, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            passes[(row["id"], row["version"])] = lenient[(row["id"], row["version"])] = row["pass"].strip() == "1"
    return passes, lenient


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # answers contain the rupee sign
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions_by_id = {q["id"]: q for q in questions}
    results = json.loads(RUN_FILE.read_text(encoding="utf-8"))["results"]
    passes, lenient = load_passes(questions_by_id, results)

    metrics, failures = [], []
    for version in VERSIONS:
        runs = [r for r in results if r["version"] == version]
        n = len(runs)
        num_ok = sum(passes[(r["id"], version)] for r in runs if r["id"] not in GRADING_IDS)
        text_ok = sum(passes[(r["id"], version)] for r in runs if r["id"] in GRADING_IDS)
        avg_in = sum(r["input_tokens"] for r in runs) / n
        avg_out = sum(r["output_tokens"] for r in runs) / n
        num_ok2 = sum(lenient[(r["id"], version)] for r in runs if r["id"] not in GRADING_IDS)
        cost = avg_in * INPUT_PRICE_PER_M / 1e6 + avg_out * OUTPUT_PRICE_PER_M / 1e6
        metrics.append([version, f"{num_ok}/{N_NUMBER}", f"{num_ok2}/{N_NUMBER}", f"{text_ok}/{N_TEXT}",
                        f"{num_ok + text_ok}/{N_NUMBER + N_TEXT}", f"{num_ok2 + text_ok}/{N_NUMBER + N_TEXT}",
                        f"{sum(r['seconds'] for r in runs) / n:.1f}", f"{avg_in:.0f}", f"{avg_out:.0f}",
                        f"${cost:.4f}", sum(1 for r in runs if r["error"])])
        for r in runs:
            if not passes[(r["id"], version)]:
                answer = (r["answer"] or r["error"] or "").replace("\n", " ").replace("|", "\\|")
                failures.append([r["id"], version, "yes" if lenient[(r["id"], version)] else "no", answer[:150]])

    out = "## Results by version\n\n"
    out += table(["version", "numbers strict (of 11)", "numbers % tolerant (of 11)", "text/names/refusals (of 9)",
                  "overall strict (of 20)", "overall % tolerant (of 20)", "avg seconds",
                  "avg input tokens", "avg output tokens", "avg cost per question", "runs with error"], metrics)
    out += "\n\n## Failed questions (strict scoring)\n\n"
    out += table(["id", "version", "passes % tolerant", "answer (first 150 chars)"], failures) if failures else "No failures."
    print(out)
    SUMMARY_FILE.write_text(out + "\n", encoding="utf-8")
    print(f"\nSaved {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
