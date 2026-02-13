from __future__ import annotations

from core.schemas.case_context import StructuredCase
from tools.data_analysis_tools.caselaw.types import RetrievalQuery


def build_queries(structured_case: StructuredCase) -> list[RetrievalQuery]:
    reasons = structured_case.get("denial_reasons", [])
    clauses = structured_case.get("policy_clauses", [])
    summary = structured_case.get("denial_summary", "")

    queries: list[RetrievalQuery] = []
    for reason in reasons:
        queries.append(
            RetrievalQuery(
                query=f"보험금 부지급 {reason} 유사 판례",
                reason=f"denial_reason:{reason}",
                source="caselaw",
                top_k=5,
            )
        )
        queries.append(
            RetrievalQuery(
                query=f"보험 분쟁 사례 {reason} 분쟁조정",
                reason=f"denial_reason:{reason}",
                source="dispute_case",
                top_k=5,
            )
        )

    for clause in clauses:
        queries.append(
            RetrievalQuery(
                query=f"보험 약관 해석 {clause} 분쟁 사례",
                reason=f"policy_clause:{clause}",
                source="caselaw",
                top_k=3,
            )
        )

    if not queries and summary:
        queries.append(
            RetrievalQuery(
                query=f"보험금 부지급 {summary} 유사 사례",
                reason="fallback:denial_summary",
                source="caselaw",
                top_k=5,
            )
        )

    deduped: list[RetrievalQuery] = []
    seen: set[tuple[str, str]] = set()
    for item in queries:
        key = (item.get("source", "caselaw"), item.get("query", "").strip())
        if key[1] and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped
