from __future__ import annotations

from dataclasses import dataclass

from core.schemas.rag_contract import RAGRetrievalResult, ScoringTrace, TreeNode
from core.schemas.rag_conventions import (
    clamp_case_adjustment,
    clamp_total_score,
)


@dataclass(frozen=True)
class ComparativeScoringInput:
    rag_result: RAGRetrievalResult
    case_adjustment: int = 0
    rationale: str = ""
    cited_case_ids: list[str] | None = None


def compute_comparative_scoring(payload: ComparativeScoringInput) -> ScoringTrace:
    guardrails_applied: list[str] = []

    precedent_score = _compute_precedent_score(payload.rag_result)
    adjustment = clamp_case_adjustment(payload.case_adjustment)
    if adjustment != payload.case_adjustment:
        guardrails_applied.append("case_adjustment_clamped")

    cited_case_ids = payload.cited_case_ids or []
    rationale = payload.rationale.strip()

    if adjustment != 0 and (not rationale or not cited_case_ids):
        adjustment = 0
        guardrails_applied.append("adjustment_invalidated_missing_rationale_or_citations")

    valid_doc_ids = {
        str(item.get("doc_id", "")).strip()
        for item in payload.rag_result.get("items", [])
        if str(item.get("doc_id", "")).strip()
    }

    if adjustment != 0:
        invalid_ids = [cid for cid in cited_case_ids if cid not in valid_doc_ids]
        if invalid_ids:
            adjustment = 0
            guardrails_applied.append("adjustment_invalidated_citation_mismatch")

    total_score = clamp_total_score(precedent_score + adjustment)
    if total_score != precedent_score + adjustment:
        guardrails_applied.append("total_score_clamped")

    return ScoringTrace(
        precedent_score=precedent_score,
        case_adjustment=adjustment,
        total_score=total_score,
        guardrails_applied=guardrails_applied,
        cited_case_ids=cited_case_ids,
        rationale=rationale,
        tree=_build_scoring_tree(
            precedent_score=precedent_score,
            case_adjustment=adjustment,
            total_score=total_score,
            guardrails_applied=guardrails_applied,
        ),
    )


def _compute_precedent_score(rag_result: RAGRetrievalResult) -> int:
    items = rag_result.get("items", [])
    if not items:
        return 0

    weighted_scores: list[float] = []
    for item in items:
        rerank_score = float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
        weighted_scores.append(max(0.0, min(1.0, rerank_score)))

    average = sum(weighted_scores) / len(weighted_scores)
    return round(average * 100)


def _build_scoring_tree(
    precedent_score: int,
    case_adjustment: int,
    total_score: int,
    guardrails_applied: list[str],
) -> TreeNode:
    root_id = "scoring:root"
    return TreeNode(
        node_id=root_id,
        node_type="ROOT",
        parent_id=None,
        title="Comparative Scoring",
        payload={"guardrails_applied": guardrails_applied},
        children=[
            TreeNode(
                node_id="scoring:precedent",
                node_type="SCORE_STEP",
                parent_id=root_id,
                title="Precedent Score",
                payload={"value": precedent_score},
                children=[],
            ),
            TreeNode(
                node_id="scoring:adjustment",
                node_type="SCORE_STEP",
                parent_id=root_id,
                title="Case Adjustment",
                payload={"value": case_adjustment},
                children=[],
            ),
            TreeNode(
                node_id="scoring:total",
                node_type="SCORE_STEP",
                parent_id=root_id,
                title="Total Score",
                payload={"value": total_score},
                children=[],
            ),
        ],
    )
