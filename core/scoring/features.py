from __future__ import annotations

from core.schemas.case_context import StructuredCase
from tools.data_analysis_tools.caselaw.types import RankedCaseLawDoc


def extract_features(
    structured_case: StructuredCase,
    ranked_cases: list[RankedCaseLawDoc],
) -> dict[str, float]:
    denial_reason_count = float(len(structured_case.get("denial_reasons", [])))
    policy_clause_count = float(len(structured_case.get("policy_clauses", [])))
    evidence_count = float(len(structured_case.get("evidence_summary", [])))
    open_question_count = float(len(structured_case.get("open_questions", [])))
    top_relevance = float(ranked_cases[0]["relevance_score"]) if ranked_cases else 0.0
    avg_relevance = (
        float(sum(doc.get("relevance_score", 0.0) for doc in ranked_cases) / len(ranked_cases))
        if ranked_cases
        else 0.0
    )

    return {
        "denial_reason_count": denial_reason_count,
        "policy_clause_count": policy_clause_count,
        "evidence_count": evidence_count,
        "open_question_count": open_question_count,
        "retrieved_case_count": float(len(ranked_cases)),
        "top_relevance": top_relevance,
        "avg_relevance": avg_relevance,
    }
