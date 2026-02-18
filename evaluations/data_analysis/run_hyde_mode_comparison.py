from __future__ import annotations

import json
from pathlib import Path

from agents.data_analysis_agent.pipeline import run_pipeline
from core.schemas.rag_contract import RetrievalMode
from evaluations.data_analysis.user_scenarios import USER_SCENARIOS


MODES: list[RetrievalMode] = ["plain", "hyde", "reverse_hyde", "hybrid_hyde"]


def main() -> None:
    rows: list[dict[str, object]] = []

    for mode in MODES:
        for scenario in USER_SCENARIOS:
            scenario_id = str(scenario["scenario_id"])
            name = str(scenario["name"])
            structured_case = scenario["structured_case"]
            result = run_pipeline(structured_case, retrieval_mode=mode)  # type: ignore[arg-type]

            rag = result.get("rag_result", {})
            score = result.get("success_probability", {})
            trace = result.get("scoring_trace", {})
            guardrails = [str(x) for x in trace.get("guardrails_applied", [])]

            rows.append(
                {
                    "mode": mode,
                    "scenario_id": scenario_id,
                    "name": name,
                    "rag_items": len(rag.get("items", [])),
                    "candidate_count": int(rag.get("stats", {}).get("candidate_count", 0)),
                    "returned_count": int(rag.get("stats", {}).get("returned_count", 0)),
                    "success_score": int(score.get("score", 0)),
                    "success_band": str(score.get("band", "LOW")),
                    "precedent_score": int(trace.get("precedent_score", 0)),
                    "case_adjustment": int(trace.get("case_adjustment", 0)),
                    "total_score": int(trace.get("total_score", 0)),
                    "guardrails": guardrails,
                    "guardrail_count": len(guardrails),
                    "query_variants": list(rag.get("query_variants", [])),
                }
            )

    summary = _build_summary(rows)
    _write_outputs(rows, summary)
    _print_table(summary)


def _build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_mode: dict[str, dict[str, object]] = {}
    for mode in MODES:
        mode_rows = [r for r in rows if r.get("mode") == mode]
        scores = [int(r.get("total_score", 0)) for r in mode_rows]
        rag_counts = [int(r.get("rag_items", 0)) for r in mode_rows]
        guardrail_counts = [int(r.get("guardrail_count", 0)) for r in mode_rows]

        bands = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for r in mode_rows:
            band = str(r.get("success_band", "LOW"))
            if band in bands:
                bands[band] += 1

        by_mode[mode] = {
            "scenario_count": len(mode_rows),
            "score_min": min(scores) if scores else 0,
            "score_max": max(scores) if scores else 0,
            "score_mean": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "avg_rag_items": round(sum(rag_counts) / len(rag_counts), 2) if rag_counts else 0.0,
            "avg_guardrail_count": round(sum(guardrail_counts) / len(guardrail_counts), 2) if guardrail_counts else 0.0,
            "band_distribution": bands,
        }

    return {"modes": MODES, "by_mode": by_mode}


def _write_outputs(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    out_dir = Path("evaluations/data_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "hyde_mode_comparison.json"
    json_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_path = out_dir / "hyde_mode_comparison.md"
    md_path.write_text(_render_markdown(summary), encoding="utf-8")


def _render_markdown(summary: dict[str, object]) -> str:
    by_mode = summary.get("by_mode", {})
    lines = [
        "# HyDE Mode Comparison",
        "",
        "| mode | scenarios | score_min | score_max | score_mean | avg_rag_items | avg_guardrail_count | LOW | MEDIUM | HIGH |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = by_mode.get(mode, {})
        bands = row.get("band_distribution", {})
        lines.append(
            f"| {mode} | {row.get('scenario_count', 0)} | {row.get('score_min', 0)} | {row.get('score_max', 0)} | "
            f"{row.get('score_mean', 0.0)} | {row.get('avg_rag_items', 0.0)} | {row.get('avg_guardrail_count', 0.0)} | "
            f"{bands.get('LOW', 0)} | {bands.get('MEDIUM', 0)} | {bands.get('HIGH', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _print_table(summary: dict[str, object]) -> None:
    by_mode = summary.get("by_mode", {})
    print("mode | score_mean | range | avg_rag_items | avg_guardrail_count | band(LOW/MEDIUM/HIGH)")
    print("-" * 95)
    for mode in MODES:
        row = by_mode.get(mode, {})
        bands = row.get("band_distribution", {})
        print(
            f"{mode:<12} | {row.get('score_mean', 0.0):>10} | "
            f"{row.get('score_min', 0):>3}-{row.get('score_max', 0):<3} | "
            f"{row.get('avg_rag_items', 0.0):>13} | {row.get('avg_guardrail_count', 0.0):>18} | "
            f"{bands.get('LOW', 0)}/{bands.get('MEDIUM', 0)}/{bands.get('HIGH', 0)}"
        )


if __name__ == "__main__":
    main()
