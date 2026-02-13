from __future__ import annotations

from typing import Any, TypedDict


class TimelineEvent(TypedDict, total=False):
    date: str
    description: str
    actor: str
    source: str


class EvidenceSummaryItem(TypedDict, total=False):
    title: str
    summary: str
    document_type: str
    source: str


class StructuredCase(TypedDict, total=False):
    user_info: dict[str, Any]
    denial_summary: str
    denial_reasons: list[str]
    policy_clauses: list[str]
    timeline: list[TimelineEvent]
    evidence_summary: list[EvidenceSummaryItem]
    open_questions: list[str]


def normalize_structured_case(raw: dict[str, Any] | StructuredCase) -> StructuredCase:
    return StructuredCase(
        user_info=dict(raw.get("user_info", {})),
        denial_summary=str(raw.get("denial_summary", "")),
        denial_reasons=[str(x) for x in raw.get("denial_reasons", [])],
        policy_clauses=[str(x) for x in raw.get("policy_clauses", [])],
        timeline=[dict(x) for x in raw.get("timeline", [])],
        evidence_summary=[dict(x) for x in raw.get("evidence_summary", [])],
        open_questions=[str(x) for x in raw.get("open_questions", [])],
    )
