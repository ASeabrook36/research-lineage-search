import json
from pathlib import Path
from statistics import median

from research.collect import ROOT, write_json
from research.search import search


def evaluate():
    cases = json.loads((Path(__file__).resolve().parent.parent / "data" / "evaluation.json").read_text())
    results = []
    for case in cases:
        response = search(case["query"], limit=5)
        hits = response["results"]
        expected = {t.casefold().rstrip(".") for t in case.get("expected_titles", [])}
        relevant = [
            hit["title"] for hit in hits
            if hit["title"].casefold().rstrip(".") in expected
            or any(term in hit["title"].casefold() for term in case.get("expected_title_terms", []))
        ]
        results.append({
            **case, "hit_at_5": bool(relevant), "relevant_titles": relevant,
            "retrieved": [{"id": hit["id"], "title": hit["title"], "score": hit["score"]} for hit in hits],
            "elapsed_ms": response["elapsed_ms"],
        })
    rate = sum(r["hit_at_5"] for r in results) / len(results)
    report = {
        "cases": len(results), "hit_at_5": rate,
        "median_latency_ms": median(r["elapsed_ms"] for r in results),
        "target_hit_at_5": 0.8, "meets_target": rate >= 0.8,
        "scope": "Small hand-authored topic relevance set, not a historical-influence or exhaustive recall benchmark.",
        "results": results,
    }
    write_json(ROOT / "evaluation-report.json", report)
    if rate < 0.8:
        raise RuntimeError(f"Retrieval relevance target missed: {rate:.0%}. Inspect evaluation-report.json")
    return report
