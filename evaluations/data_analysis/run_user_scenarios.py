from __future__ import annotations

import json
import os
from pathlib import Path

from agents.data_analysis_agent.pipeline import run_pipeline
from evaluations.data_analysis.user_scenarios import USER_SCENARIOS


def main() -> None:
    rows: list[dict[str, object]] = []
    retrieval_mode = os.getenv("RAG_RETRIEVAL_MODE", "plain")

    for scenario in USER_SCENARIOS:
        scenario_id = str(scenario["scenario_id"])
        name = str(scenario["name"])
        structured_case = scenario["structured_case"]

        result = run_pipeline(structured_case)  # type: ignore[arg-type]

        rag = result.get("rag_result", {})
        score = result.get("success_probability", {})
        trace = result.get("scoring_trace", {})

        row = {
            "scenario_id": scenario_id,
            "name": name,
            "retrieval_mode": retrieval_mode,
            "rag_items": len(rag.get("items", [])),
            "candidate_count": rag.get("stats", {}).get("candidate_count", 0),
            "returned_count": rag.get("stats", {}).get("returned_count", 0),
            "success_score": score.get("score", 0),
            "success_band": score.get("band", "LOW"),
            "precedent_score": trace.get("precedent_score", 0),
            "case_adjustment": trace.get("case_adjustment", 0),
            "total_score": trace.get("total_score", 0),
            "guardrails": trace.get("guardrails_applied", []),
            "eval_bucket": _eval_bucket(int(trace.get("total_score", 0))),
        }
        rows.append(row)

    print("scenario_id | rag_items | success_score | band | total_score | guardrails")
    print("-" * 84)
    for r in rows:
        print(
            f"{r['scenario_id']:>10} | {r['rag_items']:>9} | {r['success_score']:>13} | "
            f"{r['success_band']:<6} | {r['total_score']:>11} | {','.join(r['guardrails'])}"
        )

    out_path = Path("evaluations/data_analysis/user_scenarios_results.json")
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    summary = _build_summary(rows)
    summary_path = Path("evaluations/data_analysis/user_scenarios_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")


def _build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    scores = [int(r.get("total_score", 0)) for r in rows]
    bands: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    eval_buckets: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH_LIKE": 0}
    guardrail_counts: dict[str, int] = {}

    for r in rows:
        band = str(r.get("success_band", "LOW"))
        bands[band] = bands.get(band, 0) + 1
        bucket = str(r.get("eval_bucket", "LOW"))
        eval_buckets[bucket] = eval_buckets.get(bucket, 0) + 1
        for guardrail in r.get("guardrails", []):  # type: ignore[assignment]
            key = str(guardrail)
            guardrail_counts[key] = guardrail_counts.get(key, 0) + 1

    mean_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    return {
        "scenario_count": len(rows),
        "score_min": min(scores) if scores else 0,
        "score_max": max(scores) if scores else 0,
        "score_mean": mean_score,
        "band_distribution": bands,
        "eval_bucket_distribution": eval_buckets,
        "guardrail_distribution": guardrail_counts,
    }


def _eval_bucket(score: int) -> str:
    if score <= 35:
        return "LOW"
    if score <= 49:
        return "MEDIUM"
    return "HIGH_LIKE"


if __name__ == "__main__":
    main()
