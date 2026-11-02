"""Milestone 5 summary: scores the saved run, joins your hand grades, prints and saves three Markdown tables.
Works for a run with several repeats: metrics are averaged over all repeats.

Run from the project root:
    python -m evals.summarize [stamp]      (default stamp below)
"""

import csv
import json
import re
import sys
from collections import Counter

from evals.run_evals import (
    GRADING_IDS,
    QUESTIONS_FILE,
    RESULTS_DIR,
    VERSIONS,
    score_numbers,
)

STAMP = sys.argv[1] if len(sys.argv) > 1 else "2026-11-01_195047"
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
    """Return (strict, lenient), each {(id, version, repeat): True/False}.
    q01-q11: score_numbers (strict) and score_numbers_percent_ok (lenient). q12-q20: your 'pass' column in both.
    The grading CSV has no repeat column: rows are in run order, so the nth row of an (id, version) is repeat n."""
    passes, lenient = {}, {}
    answers = {
        (r["id"], r["version"], r["repeat"]): r["answer"] or r["error"] or ""
        for r in results
    }
    for r in results:
        if r["id"] not in GRADING_IDS:
            item = questions_by_id[r["id"]]
            key = (r["id"], r["version"], r["repeat"])
            passes[key] = bool(r["answer"]) and score_numbers(r["answer"], item)
            lenient[key] = bool(r["answer"]) and score_numbers_percent_ok(
                r["answer"], item
            )
    seen = Counter()
    with open(GRADING_FILE, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            seen[(row["id"], row["version"])] += 1
            key = (row["id"], row["version"], seen[(row["id"], row["version"])])
            assert row["answer"] == answers[key], (
                f"grading row does not match the run file: {key}"
            )
            assert row["pass"].strip() in ("0", "1"), (
                f"pass column is empty or not 0/1: {key}"
            )
            passes[key] = lenient[key] = row["pass"].strip() == "1"
    return passes, lenient


def table(header, rows):
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(
        encoding="utf-8", errors="replace"
    )  # answers contain the rupee sign
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions_by_id = {q["id"]: q for q in questions}
    run = json.loads(RUN_FILE.read_text(encoding="utf-8"))
    results, repeats = run["results"], run["meta"]["repeats"]
    passes, lenient = load_passes(questions_by_id, results)

    def key(r):
        return (r["id"], r["version"], r["repeat"])

    metrics, failures = [], []
    for version in VERSIONS:
        runs = [r for r in results if r["version"] == version]
        n = len(runs)
        nums = [r for r in runs if r["id"] not in GRADING_IDS]
        num_ok = sum(passes[key(r)] for r in nums)
        num_ok2 = sum(lenient[key(r)] for r in nums)
        text_ok = sum(passes[key(r)] for r in runs if r["id"] in GRADING_IDS)
        avg_in = sum(r["input_tokens"] for r in runs) / n
        avg_out = sum(r["output_tokens"] for r in runs) / n
        cost = avg_in * INPUT_PRICE_PER_M / 1e6 + avg_out * OUTPUT_PRICE_PER_M / 1e6
        total = (N_NUMBER + N_TEXT) * repeats
        metrics.append(
            [
                version,
                f"{num_ok}/{N_NUMBER * repeats}",
                f"{num_ok2}/{N_NUMBER * repeats}",
                f"{text_ok}/{N_TEXT * repeats}",
                f"{num_ok + text_ok}/{total}",
                f"{num_ok2 + text_ok}/{total}",
                f"{sum(r['seconds'] for r in runs) / n:.1f}",
                f"{avg_in:.0f}",
                f"{avg_out:.0f}",
                f"${cost:.4f}",
                sum(1 for r in runs if r["error"]),
            ]
        )
        for r in runs:
            if not passes[key(r)]:
                answer = (
                    (r["answer"] or r["error"] or "")
                    .replace("\n", " ")
                    .replace("|", "\\|")
                )
                failures.append(
                    [
                        r["id"],
                        version,
                        r["repeat"],
                        "yes" if lenient[key(r)] else "no",
                        answer[:150],
                    ]
                )

    rates = []
    for item in questions:
        row = [item["id"]]
        for version in VERSIONS:
            runs = [
                r for r in results if r["id"] == item["id"] and r["version"] == version
            ]
            row += [
                f"{sum(passes[key(r)] for r in runs)}/{len(runs)}",
                f"{sum(lenient[key(r)] for r in runs)}/{len(runs)}",
            ]
        rates.append(row)

    out = f"## Results by version (averaged over {repeats} repeats; passes out of {(N_NUMBER + N_TEXT) * repeats})\n\n"
    out += table(
        [
            "version",
            f"numbers strict (of {N_NUMBER * repeats})",
            f"numbers % tolerant (of {N_NUMBER * repeats})",
            f"text/names/refusals (of {N_TEXT * repeats})",
            f"overall strict (of {(N_NUMBER + N_TEXT) * repeats})",
            f"overall % tolerant (of {(N_NUMBER + N_TEXT) * repeats})",
            "avg seconds",
            "avg input tokens",
            "avg output tokens",
            "avg cost per question",
            "runs with error",
        ],
        metrics,
    )
    out += "\n\n## Pass rate per question (strict / % tolerant, out of the repeats)\n\n"
    out += table(
        ["id"] + [f"{v} {kind}" for v in VERSIONS for kind in ("strict", "tolerant")],
        rates,
    )
    out += "\n\n## Failed runs (strict scoring)\n\n"
    out += (
        table(
            [
                "id",
                "version",
                "repeat",
                "passes % tolerant",
                "answer (first 150 chars)",
            ],
            failures,
        )
        if failures
        else "No failures."
    )
    print(out)
    SUMMARY_FILE.write_text(out + "\n", encoding="utf-8")
    print(f"\nSaved {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
