from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.onboarding_agent.document_catalog import DOCUMENT_BY_ID
from evaluations.onboarding.common import load_dataset, normalize_run_summary, prepare_runtime, run_onboarding_once


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="온보딩 스모크 평가(PR 게이트)")
    parser.add_argument("--dataset", default="evaluations/onboarding/dataset.jsonl")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--output", default="evaluations/onboarding/smoke_summary.json")
    parser.add_argument("--baseline", default="evaluations/onboarding/smoke_baseline.json")
    parser.add_argument("--max-baseline-mismatch-rate", type=float, default=0.10)
    return parser.parse_args()


def _load_baseline(path: str) -> dict[str, str]:
    baseline_path = Path(path)
    if not baseline_path.exists():
        return {}
    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("signatures"), dict):
        raw = payload.get("signatures", {})
    elif isinstance(payload, dict):
        raw = payload
    else:
        raw = {}
    result: dict[str, str] = {}
    for key, value in raw.items():
        scenario_id = str(key).strip()
        signature = str(value).strip()
        if scenario_id and signature:
            result[scenario_id] = signature
    return result


def main() -> None:
    args = _parse_args()
    dataset = load_dataset(args.dataset)
    if not dataset:
        raise SystemExit("dataset이 비어 있습니다.")

    target_cases = dataset[: max(int(args.max_cases), 1)]
    runtime = prepare_runtime()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for case in target_cases:
        scenario_id = str(case.get("scenario_id", "unknown")).strip() or "unknown"
        state = run_onboarding_once(case, runtime, repeat_index=0)
        row = normalize_run_summary(state, scenario_id=scenario_id, repeat_index=0)
        rows.append(row)

        required_ids = list(row.get("required_document_ids") or [])
        if not required_ids:
            failures.append(f"{scenario_id}: required_document_ids 비어 있음")
        if len(required_ids) > 3:
            failures.append(f"{scenario_id}: required_document_ids가 3개 초과")
        invalid_ids = [doc_id for doc_id in required_ids if doc_id not in DOCUMENT_BY_ID]
        if invalid_ids:
            failures.append(f"{scenario_id}: invalid required_document_ids={invalid_ids}")

        if not isinstance(row.get("evidence_sufficient"), bool):
            failures.append(f"{scenario_id}: evidence_sufficient 타입 오류")
        if not str(row.get("decision_signature", "")).strip():
            failures.append(f"{scenario_id}: decision_signature 누락")
        if "inconsistency:sufficiency_without_evidence" in list(row.get("quality_flags") or []):
            failures.append(f"{scenario_id}: sufficiency 근거 불일치 플래그 감지")

    baseline = _load_baseline(args.baseline)
    comparable = [row for row in rows if str(row.get("scenario_id")) in baseline]
    baseline_mismatch_count = 0
    for row in comparable:
        scenario_id = str(row.get("scenario_id"))
        if str(row.get("decision_signature", "")).strip() != baseline.get(scenario_id, ""):
            baseline_mismatch_count += 1
    baseline_mismatch_rate = (
        baseline_mismatch_count / len(comparable)
        if comparable
        else 0.0
    )
    if comparable and baseline_mismatch_rate > float(args.max_baseline_mismatch_rate):
        failures.append(
            "baseline mismatch rate 초과: "
            f"{baseline_mismatch_rate:.4f} > {float(args.max_baseline_mismatch_rate):.4f}"
        )

    summary = {
        "dataset_path": args.dataset,
        "case_count": len(rows),
        "baseline_case_count": len(comparable),
        "baseline_mismatch_rate": round(float(baseline_mismatch_rate), 4),
        "max_baseline_mismatch_rate": float(args.max_baseline_mismatch_rate),
        "passed": len(failures) == 0,
        "failures": failures,
        "rows": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("스모크 평가 실패")


if __name__ == "__main__":
    main()

