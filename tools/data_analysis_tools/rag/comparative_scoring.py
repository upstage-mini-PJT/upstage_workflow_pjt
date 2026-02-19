from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from core.schemas.rag_contract import RAGRetrievalResult, ScoringTrace, TreeNode
from core.schemas.rag_conventions import (
    clamp_case_adjustment,
    clamp_total_score,
)


@dataclass(frozen=True)
class ComparativeScoringInput:
    rag_result: RAGRetrievalResult
    case_adjustment: int = 0
    case_adjustment_source: str = "zero"
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

    if adjustment != 0 and not cited_case_ids:
        adjustment = 0
        guardrails_applied.append("adjustment_invalidated_missing_citations")

    valid_doc_ids = {
        str(item.get("doc_id", "")).strip()
        for item in payload.rag_result.get("items", [])
        if str(item.get("doc_id", "")).strip()
    }

    if adjustment != 0:
        invalid_ids = [cid for cid in cited_case_ids if cid not in valid_doc_ids]
        if invalid_ids:
            cited_case_ids = [cid for cid in cited_case_ids if cid in valid_doc_ids]
            guardrails_applied.append("adjustment_invalidated_citation_mismatch")
            if not cited_case_ids:
                adjustment = 0

    total_score = clamp_total_score(precedent_score + adjustment)
    if total_score != precedent_score + adjustment:
        guardrails_applied.append("total_score_clamped")

    return ScoringTrace(
        precedent_score=precedent_score,
        case_adjustment=adjustment,
        case_adjustment_source=str(payload.case_adjustment_source or "zero"),
        total_score=total_score,
        guardrails_applied=guardrails_applied,
        cited_case_ids=cited_case_ids,
        rationale=rationale,
        tree=_build_scoring_tree(
            precedent_score=precedent_score,
            case_adjustment=adjustment,
            total_score=total_score,
            guardrails_applied=guardrails_applied,
            retrieval_context={
                "retrieval_mode": payload.rag_result.get("retrieval_mode", "plain"),
                "query_variants": payload.rag_result.get("query_variants", ["plain"]),
                "retrieved_items": len(payload.rag_result.get("items", [])),
            },
        ),
    )


def _compute_precedent_score(rag_result: RAGRetrievalResult) -> int:
    items = rag_result.get("items", [])
    if not items:
        return 0

    scores = [
        max(0.0, min(1.0, float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)))
        for item in items
    ]
    weighted = _weighted_topk_score(scores)
    calibrated = weighted**0.75

    base = calibrated * 80.0
    top_boost = min(12.0, max(0.0, scores[0] - 0.5) * 40.0)
    consistency = _consistency_bonus(scores)
    coverage = min(8.0, len(scores) * 1.6)
    diversity = _source_diversity_bonus(items)
    low_quality_penalty = _low_quality_penalty(scores[0])

    raw_score = base + top_boost + consistency + coverage + diversity - low_quality_penalty
    return max(0, min(100, round(raw_score)))


def _weighted_topk_score(scores: list[float]) -> float:
    top1 = scores[0] if len(scores) >= 1 else 0.0
    top2 = scores[1] if len(scores) >= 2 else top1
    top3 = scores[2] if len(scores) >= 3 else top2
    tail = scores[3:8] if len(scores) >= 4 else scores
    tail_avg = sum(tail) / len(tail) if tail else 0.0
    return (0.45 * top1) + (0.25 * top2) + (0.15 * top3) + (0.15 * tail_avg)


def _consistency_bonus(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 2.0
    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)
    std = sqrt(variance)
    return max(0.0, 5.0 - (std * 15.0))


def _source_diversity_bonus(items: list[dict]) -> float:
    source_types = {str(item.get("source_type", "")) for item in items if item.get("source_type")}
    # 1 source -> 0, 2 sources -> 2.5, 3+ sources -> 5
    return min(5.0, max(0.0, (len(source_types) - 1) * 2.5))


def _low_quality_penalty(top_score: float) -> float:
    if top_score >= 0.4:
        return 0.0
    return min(10.0, (0.4 - top_score) * 35.0)


def _build_scoring_tree(
    precedent_score: int,
    case_adjustment: int,
    total_score: int,
    guardrails_applied: list[str],
    retrieval_context: dict | None = None,
) -> TreeNode:
    root_id = "scoring:root"
    return TreeNode(
        node_id=root_id,
        node_type="ROOT",
        parent_id=None,
        title="Comparative Scoring",
        payload={
            "guardrails_applied": guardrails_applied,
            "retrieval_context": retrieval_context or {},
        },
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
