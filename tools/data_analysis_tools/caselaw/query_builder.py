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
                top_k=5,
            )
        )
    for clause in clauses:
        queries.append(
            RetrievalQuery(
                query=f"보험 약관 해석 {clause} 분쟁 사례",
                reason=f"policy_clause:{clause}",
                top_k=5,
            )
        )
    if not queries and summary:
        queries.append(
            RetrievalQuery(
                query=f"보험금 부지급 {summary} 분쟁 사례",
                reason="fallback:denial_summary",
                top_k=5,
            )
        )

    deduped: list[RetrievalQuery] = []
    seen: set[str] = set()
    for item in queries:
        query = item.get("query", "").strip()
        if query and query not in seen:
            seen.add(query)
            deduped.append(item)
    return deduped
