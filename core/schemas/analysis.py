from __future__ import annotations

from typing import Literal, TypedDict

from core.schemas.provenance import Provenance


class IssueNode(TypedDict, total=False):
    issue_id: str
    title: str
    description: str
    related_denial_reasons: list[str]
    related_policy_clauses: list[str]


class IssueTree(TypedDict, total=False):
    root_title: str
    nodes: list[IssueNode]


class GapAnalysisItem(TypedDict, total=False):
    issue_id: str
    missing_evidence: str
    impact: Literal["low", "medium", "high"]
    rationale: str


class RecommendedAction(TypedDict, total=False):
    action_id: str
    title: str
    detail: str
    priority: Literal["low", "medium", "high"]
    linked_issue_id: str


class SuccessProbability(TypedDict, total=False):
    band: Literal["LOW", "MEDIUM", "HIGH"]
    positive_drivers: list[str]
    negative_drivers: list[str]
    assumptions: list[str]


class EvidencePackItem(TypedDict, total=False):
    evidence_id: str
    issue_id: str
    evidence_title: str
    summary: str
    relevance_score: float
    provenance: list[Provenance]


class EvidenceGuideItem(TypedDict, total=False):
    ref_token: str
    source_label: Literal["판례", "분쟁사례", "웹자료"]
    title: str
    why_relevant: str


class UserGuidance(TypedDict, total=False):
    plain_summary: str
    next_steps: list[str]
    evidence_guide: list[EvidenceGuideItem]
    disclaimer: str


class RAGItemProvenance(TypedDict, total=False):
    index_name: str
    retrieved_at: str
    retrieval_method: Literal["vector", "keyword", "hybrid"]


class RAGRetrievalItem(TypedDict, total=False):
    source_type: Literal["CASELAW", "DISPUTE", "WEB"]
    doc_id: str
    chunk_id: str
    title: str
    snippet: str
    score: float
    rerank_score: float
    url: str | None
    published_at: str | None
    provenance: RAGItemProvenance


class RAGRetrievalStats(TypedDict, total=False):
    candidate_count: int
    returned_count: int
    latency_ms: int


class RAGRetrievalResult(TypedDict, total=False):
    query_id: str
    query: str
    filters: dict
    items: list[RAGRetrievalItem]
    stats: RAGRetrievalStats


class ScoringTrace(TypedDict, total=False):
    precedent_score: int
    case_adjustment: int
    case_adjustment_source: Literal["llm", "heuristic", "zero"]
    total_score: int
    guardrails_applied: list[str]
    cited_case_ids: list[str]


class AnalysisResult(TypedDict, total=False):
    issue_tree: IssueTree
    gap_analysis: list[GapAnalysisItem]
    recommended_actions: list[RecommendedAction]
    success_probability: SuccessProbability
    evidence_pack: list[EvidencePackItem]
    user_guidance: UserGuidance
