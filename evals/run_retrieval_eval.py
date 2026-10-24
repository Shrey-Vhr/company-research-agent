"""Retrieval eval: run search() on each question and check whether a gold page is in the top 5.

Run from the project root: python -m evals.run_retrieval_eval [--collection annual_reports_1600]
"""

import argparse
import json
from datetime import date
from pathlib import Path

from research_agent.rag.search import search

EVALS_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVALS_DIR / "retrieval_questions.json"
RESULTS_DIR = EVALS_DIR / "results"
K = 5


def is_hit(result_pages, answer_pages):
    """Return True if the search found the answer. Written by hand, not generated."""
    for page in result_pages:
        if page in answer_pages:
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="annual_reports_1600")
    args = parser.parse_args()

    items = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    print(f"Collection: {args.collection}")

    rows = []
    for item in items:
        results = search(item["question"], company=item["company"], k=K, collection_name=args.collection)
        result_pages = [r["pdf_page"] for r in results]
        hit = is_hit(result_pages, item["answer_pages"])
        print(f'{item["id"]}  {item["company"]:<6} {"HIT " if hit else "MISS"}  pages {result_pages}')
        rows.append({
            "id": item["id"],
            "company": item["company"],
            "question": item["question"],
            "answer_pages": item["answer_pages"],
            "result_pages": result_pages,
            "scores": [r["score"] for r in results],
            "hit": hit,
        })

    hits = sum(row["hit"] for row in rows)
    print(f"\nTotal hit rate: {hits}/{len(rows)} = {hits / len(rows):.0%}")

    by_company = {}
    for row in rows:
        company_hits, company_total = by_company.get(row["company"], (0, 0))
        by_company[row["company"]] = (company_hits + row["hit"], company_total + 1)
    for company, (company_hits, company_total) in by_company.items():
        print(f"  {company:<6} {company_hits}/{company_total} = {company_hits / company_total:.0%}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"retrieval_{args.collection}_{date.today().isoformat()}.json"
    summary = {
        "date": date.today().isoformat(),
        "collection": args.collection,
        "k": K,
        "hits": hits,
        "total": len(rows),
        "hit_rate": round(hits / len(rows), 3),
        "by_company": {c: {"hits": h, "total": t} for c, (h, t) in by_company.items()},
        "results": rows,
    }
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out_file}")
