"""End-to-end eval runner (shell only): runs the questions through the agent versions and saves the raw results.

Run from the project root, e.g.:
    python -m evals.run_evals --ids q02 q18 --versions raw graph_full
Scoring is NOT here: score_numbers is a stub for you to write.
"""

import argparse
import csv
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler

from research_agent.config import RECURSION_LIMIT
from research_agent.graph.build import build_graph
from research_agent.graph.numbers import extract_numbers
from research_agent.raw_agent.loop import run_agent

EVALS_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVALS_DIR / "questions.json"
RESULTS_DIR = EVALS_DIR / "results"
VERSIONS = ["raw", "graph_no_critic", "graph_full"]
GRADING_IDS = {f"q{n:02d}" for n in range(12, 21)}  # q12-q20 are graded by hand


def _close(x: float, target: float, tol: dict) -> bool:
    if tol["type"] == "relative":
        return abs(x - target) <= tol["value"] * abs(target)
    return abs(x - target) <= tol["value"]


def _matches_any(numbers: list[float], values: list[float], tol: dict) -> bool:
    return any(_close(n, v, tol) for n in numbers for v in values)


def score_numbers(answer: str, item: dict) -> bool:
    numbers = extract_numbers(answer)
    tol = item["tolerance"]
    groups = item.get("required_groups")
    if groups:  # comparisons: every company needs a correct value
        return all(_matches_any(numbers, g, tol) for g in groups)
    return _matches_any(numbers, item["accepted_values"], tol)


def run_raw(question):
    """The hand-written loop. It returns the text plus the tool results and its own token totals."""
    answer, details = run_agent(question, return_details=True)
    return {
        "answer": answer,
        "evidence": details["tool_results"],
        "input_tokens": details["input_tokens"],
        "output_tokens": details["output_tokens"],
    }


def run_graph(graph, question, skip_critic, usage):
    """The LangGraph version. A new thread_id per run; auto_approve skips the human pause."""
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": RECURSION_LIMIT,
        "callbacks": [
            usage
        ],  # collects usage_metadata from every model call inside the graph
    }
    start = {
        "question": question,
        "evidence": [],
        "auto_approve": True,
        "skip_critic": skip_critic,
    }
    result = graph.invoke(start, config)
    return {
        "answer": result.get("final_report"),
        "evidence": result.get("evidence", []),
        "critic_passed": result.get("critic_passed"),
        "revision_count": result.get("revision_count"),
    }


def run_one(version, question, graph):
    """Run one question on one version. Never raises: an error is recorded and the eval goes on."""
    usage = UsageMetadataCallbackHandler()
    record = {
        "answer": None,
        "evidence": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None,
    }
    start = time.perf_counter()
    try:
        if version == "raw":
            record.update(run_raw(question))
        else:
            record.update(
                run_graph(
                    graph,
                    question,
                    skip_critic=(version == "graph_no_critic"),
                    usage=usage,
                )
            )
    except Exception as e:  # noqa: BLE001 - record any failure and keep going
        record["error"] = f"{type(e).__name__}: {e}"
    record["seconds"] = round(time.perf_counter() - start, 1)
    if version != "raw":
        # usage_metadata is {model_name: {input_tokens, output_tokens, total_tokens, ...}}
        record["input_tokens"] = sum(
            u.get("input_tokens", 0) for u in usage.usage_metadata.values()
        )
        record["output_tokens"] = sum(
            u.get("output_tokens", 0) for u in usage.usage_metadata.values()
        )
    return record


def expected_text(item):
    points = "; ".join(item.get("expected_points", []))
    return f"{points} | {item.get('notes', '')}" if points else item.get("notes", "")


def write_grading_csv(path, results, questions_by_id):
    rows = [r for r in results if r["id"] in GRADING_IDS]
    with open(
        path, "w", newline="", encoding="utf-8-sig"
    ) as f:  # utf-8-sig so Excel shows the rupee sign
        writer = csv.writer(f)
        writer.writerow(["id", "version", "question", "answer", "expected", "pass"])
        for r in rows:
            writer.writerow(
                [
                    r["id"],
                    r["version"],
                    r["question"],
                    r["answer"] or r["error"] or "",
                    expected_text(questions_by_id[r["id"]]),
                    "",
                ]
            )
    return len(rows)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(message)s"
    )  # shows the raw loop's round logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.stdout.reconfigure(
        encoding="utf-8", errors="replace"
    )  # the agent prints rupee signs; Windows console is cp1252
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", choices=VERSIONS, default=VERSIONS)
    parser.add_argument(
        "--ids", nargs="+", help="question ids to run, e.g. q02 q18 (default: all)"
    )
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions_by_id = {q["id"]: q for q in questions}
    if args.ids:
        unknown = [i for i in args.ids if i not in questions_by_id]
        if unknown:
            sys.exit(f"Unknown question ids: {unknown}")
        questions = [questions_by_id[i] for i in args.ids]

    graph = build_graph() if any(v != "raw" for v in args.versions) else None
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    run_path = RESULTS_DIR / f"run_{stamp}.json"
    grading_path = RESULTS_DIR / f"grading_{stamp}.csv"

    results = []
    for item in questions:
        for version in args.versions:
            for repeat in range(1, args.repeats + 1):
                record = run_one(version, item["question"], graph)
                results.append(
                    {
                        "id": item["id"],
                        "category": item["category"],
                        "question": item["question"],
                        "version": version,
                        "repeat": repeat,
                        **record,
                    }
                )
                tag = f" ERROR {record['error']}" if record["error"] else ""
                print(f"{item['id']} {version} {record['seconds']}s{tag}")
                meta = {
                    "versions": args.versions,
                    "ids": [q["id"] for q in questions],
                    "repeats": args.repeats,
                }
                # save after every run so a crash does not lose the earlier ones
                run_path.write_text(
                    json.dumps(
                        {"meta": meta, "results": results}, indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )

    n = write_grading_csv(grading_path, results, questions_by_id)
    print(f"Saved {run_path}")
    print(f"Saved {grading_path} ({n} rows)")


if __name__ == "__main__":
    main()
