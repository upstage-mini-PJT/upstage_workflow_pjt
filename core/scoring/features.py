from __future__ import annotations

from typing import Any

from core.schemas.case_context import StructuredCase
from tools.data_analysis_tools.caselaw.types import RankedCaseLawDoc


WIN_HINTS = ("승소", "인용", "취소", "일부 인용", "조정 성립")
LOSE_HINTS = ("기각", "패소", "각하", "불수용")


def extract_features(
    structured_case: StructuredCase,
    ranked_cases: list[RankedCaseLawDoc],
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    features.update(_extract_precedent_features(ranked_cases))
    features.update(_extract_evidence_features(structured_case))
    features.update(_extract_case_shape_features(structured_case))
    return features


def _extract_precedent_features(ranked_cases: list[RankedCaseLawDoc]) -> dict[str, Any]:
    if not ranked_cases:
        return {
            "has_similar_precedent": False,
            "precedent_count": 0,
            "precedent_win_rate": 0.0,
            "top_relevance": 0.0,
            "avg_relevance": 0.0,
        }

    wins = 0
    judged = 0
    for row in ranked_cases:
        result_text = str(row.get("result", ""))
        if any(h in result_text for h in WIN_HINTS):
            wins += 1
            judged += 1
        elif any(h in result_text for h in LOSE_HINTS):
            judged += 1

    avg_rel = sum(float(r.get("relevance_score", 0.0)) for r in ranked_cases) / len(ranked_cases)
    return {
        "has_similar_precedent": any(float(x.get("relevance_score", 0.0)) >= 0.35 for x in ranked_cases),
        "precedent_count": len(ranked_cases),
        "precedent_win_rate": (wins / judged) if judged > 0 else 0.5,
        "top_relevance": float(ranked_cases[0].get("relevance_score", 0.0)),
        "avg_relevance": float(avg_rel),
    }


def _extract_evidence_features(structured_case: StructuredCase) -> dict[str, Any]:
    evidence_rows = structured_case.get("evidence_summary", [])
    timeline = structured_case.get("timeline", [])

    text_blob = " ".join(
        [
            str(x.get("title", "")) + " " + str(x.get("summary", "")) + " " + str(x.get("source", ""))
            for x in evidence_rows
        ]
        + [
            str(x.get("description", "")) + " " + str(x.get("source", ""))
            for x in timeline
        ]
    )

    has_medical_records = any(k in text_blob for k in ("진료기록", "의무기록", "EMR", "검사결과"))
    has_doctor_note = any(k in text_blob for k in ("소견서", "진단서", "의사소견"))

    score = 0.0
    if has_medical_records:
        score += 0.4
    if has_doctor_note:
        score += 0.3
    if len(evidence_rows) >= 2:
        score += 0.2
    if len(timeline) >= 1:
        score += 0.1

    return {
        "has_medical_records": has_medical_records,
        "has_doctor_note": has_doctor_note,
        "evidence_completeness": min(1.0, score),
    }


def _extract_case_shape_features(structured_case: StructuredCase) -> dict[str, Any]:
    timeline = structured_case.get("timeline", [])
    dates = [str(x.get("date", "")) for x in timeline if x.get("date")]
    timeline_consistency = 1.0 if dates == sorted(dates) else 0.7

    return {
        "open_question_count": len(structured_case.get("open_questions", [])),
        "denial_reason_count": len(structured_case.get("denial_reasons", [])),
        "policy_clause_count": len(structured_case.get("policy_clauses", [])),
        "timeline_consistency": timeline_consistency,
    }
