from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage
from langchain_upstage import ChatUpstage
from pydantic import BaseModel, Field

from evaluations.onboarding.common import load_dataset, normalize_run_summary, prepare_runtime, run_onboarding_once


class QualJudgeResponse(BaseModel):
    groundedness: int = Field(ge=1, le=5)
    consistency: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    groundedness_violations: int = Field(ge=0, le=10)
    rationale: str = Field(default="")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="온보딩 정성평가(LLM judge)")
    parser.add_argument("--dataset", default="evaluations/onboarding/dataset.jsonl")
    parser.add_argument("--rubric", default="evaluations/onboarding/rubric.yaml")
    parser.add_argument("--output", default="evaluations/onboarding/qual_summary.json")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--min-qual-score-avg", type=float, default=4.0)
    parser.add_argument("--max-groundedness-violations", type=int, default=0)
    return parser.parse_args()


def _build_judge_client() -> ChatUpstage:
    model = str(os.getenv("ONBOARDING_EVAL_JUDGE_MODEL", "solar-pro2")).strip() or "solar-pro2"
    try:
        timeout_sec = float(str(os.getenv("ONBOARDING_EVAL_JUDGE_TIMEOUT", "45")).strip())
    except ValueError:
        timeout_sec = 45.0
    return ChatUpstage(model=model, timeout=max(timeout_sec, 5.0), max_retries=1)


def _load_rubric(path: str) -> dict[str, Any]:
    rubric_path = Path(path)
    if not rubric_path.exists():
        return {}
    try:
        payload = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _trim_issue_query_plan(raw_items: list[Any], *, max_items: int = 4) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for item in raw_items[:max_items]:
        if not isinstance(item, dict):
            continue
        intent = str(item.get("intent", "")).strip()
        query_seed = str(item.get("query_seed", "")).strip()
        must_keywords = [str(keyword).strip() for keyword in (item.get("must_keywords") or []) if str(keyword).strip()]
        priority = item.get("priority")
        trimmed.append(
            {
                "intent": intent,
                "query_seed": query_seed[:220],
                "must_keywords": must_keywords[:5],
                "priority": priority,
            }
        )
    return trimmed


def _build_prompt(rubric: dict[str, Any], onboarding_state: dict[str, Any], row: dict[str, Any]) -> str:
    rubric_text = yaml.safe_dump(rubric, allow_unicode=True, sort_keys=False) if rubric else "(rubric 없음)"
    decision_summary = onboarding_state.get("decision_summary", {})
    decision_explanation = str(onboarding_state.get("decision_explanation", "")).strip()
    trace_preview = list(onboarding_state.get("decision_trace") or [])[-4:]
    issue_hypotheses = [str(item).strip() for item in (row.get("issue_hypotheses") or []) if str(item).strip()][:6]
    issue_query_plan = _trim_issue_query_plan(list(row.get("issue_query_plan") or []), max_items=4)
    issue_query_seeds = [str(item).strip() for item in (row.get("issue_query_seeds") or []) if str(item).strip()][:4]
    final_plan_text = str(row.get("final_plan_text", "")).strip()
    final_plan_preview = str(row.get("final_plan_preview", "")).strip()
    issue_planning_output = {
        "issue_hypotheses": issue_hypotheses,
        "issue_query_plan": issue_query_plan,
        "issue_query_seeds": issue_query_seeds,
    }
    final_planning_output = {
        "final_plan_preview": final_plan_preview[:600],
        "final_plan_text": final_plan_text[:1800],
        "required_document_ids": row.get("required_document_ids", []),
        "final_plan_confidence": row.get("final_plan_confidence"),
    }
    return f"""당신은 보험 온보딩 판단품질 평가자입니다.
아래 출력물을 rubric 기준으로 정성평가하세요.

[rubric]
{rubric_text}

[핵심 판단 요약(결과)]
- required_document_ids: {row.get("required_document_ids", [])}
- evidence_sufficient: {row.get("evidence_sufficient")}
- final_plan_confidence: {row.get("final_plan_confidence")}
- quality_flags: {row.get("quality_flags", [])}

[issue_planning_output]
{json.dumps(issue_planning_output, ensure_ascii=False)}

[final_planning_output]
{json.dumps(final_planning_output, ensure_ascii=False)}

[decision_summary]
{json.dumps(decision_summary, ensure_ascii=False)}

[decision_explanation]
{decision_explanation[:1800]}

[trace_preview]
{json.dumps(trace_preview, ensure_ascii=False)}

채점 규칙:
1) groundedness: 문서/약관 근거에 기반한 판단인지 (1~5)
   - issue_planning 쿼리/가설이 근거 탐색에 실제로 기여하는지 포함
2) consistency: 단계 간 모순이 없는지 (1~5)
   - issue_planning → final_planning → decision_summary 흐름 일관성 포함
3) actionability: 사용자 입장에서 실행 가능한지 (1~5)
   - final_planning의 계획/요청서류가 실행 가능한지 포함
4) groundedness_violations: 근거 불충분/허위추론 의심 건수(0~10)
5) rationale: 한두 문장 근거
"""


def _fallback_score(row: dict[str, Any]) -> QualJudgeResponse:
    flags = list(row.get("quality_flags") or [])
    has_inconsistency = "inconsistency:sufficiency_without_evidence" in flags
    groundedness = 2 if has_inconsistency else 4
    consistency = 3 if any(flag.startswith("fallback:") for flag in flags) else 4
    actionability = 4 if row.get("required_document_ids") else 2
    return QualJudgeResponse(
        groundedness=groundedness,
        consistency=consistency,
        actionability=actionability,
        groundedness_violations=1 if has_inconsistency else 0,
        rationale="LLM judge 실패로 rule-based fallback 적용",
    )


def main() -> None:
    args = _parse_args()
    dataset = load_dataset(args.dataset)
    if not dataset:
        raise SystemExit("dataset이 비어 있습니다.")
    cases = dataset[: max(int(args.max_cases), 1)]
    runtime = prepare_runtime()
    rubric = _load_rubric(args.rubric)
    judge_client = _build_judge_client()
    judge_structured = judge_client.with_structured_output(QualJudgeResponse)

    rows: list[dict[str, Any]] = []
    score_values: list[float] = []
    groundedness_violation_count = 0

    for case in cases:
        scenario_id = str(case.get("scenario_id", "unknown")).strip() or "unknown"
        onboarding_state = run_onboarding_once(case, runtime, repeat_index=0)
        run_row = normalize_run_summary(onboarding_state, scenario_id=scenario_id, repeat_index=0)

        prompt = _build_prompt(rubric, onboarding_state, run_row)
        try:
            judged: QualJudgeResponse = judge_structured.invoke([HumanMessage(content=prompt)])
        except Exception:
            judged = _fallback_score(run_row)

        avg_score = (judged.groundedness + judged.consistency + judged.actionability) / 3.0
        score_values.append(avg_score)
        groundedness_violation_count += int(judged.groundedness_violations)
        rows.append(
            {
                **run_row,
                "qual_scores": {
                    "groundedness": judged.groundedness,
                    "consistency": judged.consistency,
                    "actionability": judged.actionability,
                    "avg": round(avg_score, 4),
                },
                "groundedness_violations": int(judged.groundedness_violations),
                "judge_rationale": str(judged.rationale).strip(),
            }
        )

    qual_score_avg = round(sum(score_values) / len(score_values), 4) if score_values else 0.0
    summary = {
        "dataset_path": args.dataset,
        "case_count": len(rows),
        "qual_score_avg": qual_score_avg,
        "groundedness_violations": groundedness_violation_count,
        "thresholds": {
            "min_qual_score_avg": float(args.min_qual_score_avg),
            "max_groundedness_violations": int(args.max_groundedness_violations),
        },
        "passed": (
            qual_score_avg >= float(args.min_qual_score_avg)
            and groundedness_violation_count <= int(args.max_groundedness_violations)
        ),
        "rows": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit("정성평가 gate 실패")


if __name__ == "__main__":
    main()
