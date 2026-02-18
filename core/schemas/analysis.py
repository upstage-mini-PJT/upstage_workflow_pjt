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


class AnalysisResult(TypedDict, total=False):
    issue_tree: IssueTree
    gap_analysis: list[GapAnalysisItem]
    recommended_actions: list[RecommendedAction]
    success_probability: SuccessProbability
    evidence_pack: list[EvidencePackItem]
