from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluations.onboarding.common import load_dataset, normalize_run_summary, prepare_runtime, run_onboarding_once
from evaluations.onboarding.metrics import (
    confidence_match_rate,
    decision_exact_match_rate,
    required_docs_exact_match_rate,
    retrieval_topk_jaccard_mean,
    sufficiency_flip_rate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="온보딩 재현성 평가")
    parser.add_argument("--dataset", default="evaluations/onboarding/dataset.jsonl")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", default="evaluations/onboarding")
    parser.add_argument("--min-decision-match-rate", type=float, default=0.90)
    parser.add_argument("--min-required-docs-match-rate", type=float, default=0.95)
    parser.add_argument("--max-sufficiency-flip-rate", type=float, default=0.05)
    return parser.parse_args()


def _round(value: float) -> float:
    return round(float(value), 4)


def _case_drift_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("scenario_id", "unknown")), []).append(row)

    result: list[dict[str, Any]] = []
    for scenario_id, runs in sorted(grouped.items()):
        signatures = sorted({str(run.get("decision_signature", "")).strip() for run in runs if str(run.get("decision_signature", "")).strip()})
        required_sets = sorted({tuple(run.get("required_document_ids") or []) for run in runs})
        suff_values = sorted({run.get("evidence_sufficient") for run in runs if isinstance(run.get("evidence_sufficient"), bool)})
        conf_values = sorted({str(run.get("final_plan_confidence", "")).strip().lower() for run in runs if str(run.get("final_plan_confidence", "")).strip()})
        result.append(
            {
                "scenario_id": scenario_id,
                "run_count": len(runs),
                "signature_variant_count": len(signatures),
                "required_docs_variant_count": len(required_sets),
                "sufficiency_variants": suff_values,
                "confidence_variants": conf_values,
                "signature_variants_preview": signatures[:5],
            }
        )
    return result


def main() -> None:
    args = _parse_args()
    dataset = load_dataset(args.dataset)
    if not dataset:
        raise SystemExit("dataset이 비어 있습니다.")
    repeats = max(int(args.repeats), 1)

    runtime = prepare_runtime()
    run_rows: list[dict[str, Any]] = []

    for case in dataset:
        scenario_id = str(case.get("scenario_id", "unknown")).strip() or "unknown"
        for repeat_idx in range(repeats):
            state = run_onboarding_once(case, runtime, repeat_index=repeat_idx)
            run_rows.append(normalize_run_summary(state, scenario_id=scenario_id, repeat_index=repeat_idx))

    metrics = {
        "decision_exact_match_rate": _round(decision_exact_match_rate(run_rows)),
        "required_docs_exact_match_rate": _round(required_docs_exact_match_rate(run_rows)),
        "sufficiency_flip_rate": _round(sufficiency_flip_rate(run_rows)),
        "confidence_match_rate": _round(confidence_match_rate(run_rows)),
        "retrieval_topk_jaccard_mean": _round(retrieval_topk_jaccard_mean(run_rows)),
    }
    thresholds = {
        "min_decision_exact_match_rate": float(args.min_decision_match_rate),
        "min_required_docs_exact_match_rate": float(args.min_required_docs_match_rate),
        "max_sufficiency_flip_rate": float(args.max_sufficiency_flip_rate),
    }

    hard_gate_passed = (
        metrics["decision_exact_match_rate"] >= thresholds["min_decision_exact_match_rate"]
        and metrics["required_docs_exact_match_rate"] >= thresholds["min_required_docs_exact_match_rate"]
        and metrics["sufficiency_flip_rate"] <= thresholds["max_sufficiency_flip_rate"]
    )
    summary = {
        "dataset_path": args.dataset,
        "scenario_count": len(dataset),
        "repeats": repeats,
        "run_count": len(run_rows),
        "metrics": metrics,
        "thresholds": thresholds,
        "hard_gate_passed": hard_gate_passed,
    }
    drift_report = {
        "summary": summary,
        "by_scenario": _case_drift_report(run_rows),
        "runs": run_rows,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    drift_path = out_dir / "drift_report.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    drift_path.write_text(json.dumps(drift_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not hard_gate_passed:
        raise SystemExit("재현성 hard gate 실패")


if __name__ == "__main__":
    main()

