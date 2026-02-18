from core.schemas.rag_contract import RAGRetrievalResult
from tools.data_analysis_tools.rag.comparative_scoring import (
    ComparativeScoringInput,
    compute_comparative_scoring,
)


def _sample_rag() -> RAGRetrievalResult:
    return RAGRetrievalResult(
        query_id="q1",
        query="보험",
        filters={},
        items=[
            {"doc_id": "CASELAW:1", "score": 0.7, "rerank_score": 0.8},
            {"doc_id": "CASELAW:2", "score": 0.5, "rerank_score": 0.4},
        ],
        tree={"node_id": "root", "node_type": "ROOT", "children": []},
        stats={"candidate_count": 2, "returned_count": 2, "latency_ms": 1},
    )


def test_case_adjustment_is_clamped() -> None:
    trace = compute_comparative_scoring(
        ComparativeScoringInput(
            rag_result=_sample_rag(),
            case_adjustment=99,
            rationale="충분한 근거",
            cited_case_ids=["CASELAW:1"],
        )
    )
    assert trace["case_adjustment"] == 15
    assert "case_adjustment_clamped" in trace["guardrails_applied"]


def test_missing_rationale_and_citations_invalidates_adjustment() -> None:
    trace = compute_comparative_scoring(
        ComparativeScoringInput(
            rag_result=_sample_rag(),
            case_adjustment=10,
            rationale="",
            cited_case_ids=[],
        )
    )
    assert trace["case_adjustment"] == 0
    assert "adjustment_invalidated_missing_rationale_or_citations" in trace["guardrails_applied"]


def test_citation_mismatch_is_filtered_when_rationale_exists() -> None:
    trace = compute_comparative_scoring(
        ComparativeScoringInput(
            rag_result=_sample_rag(),
            case_adjustment=8,
            rationale="관련 판례 존재",
            cited_case_ids=["CASELAW:999"],
        )
    )
    assert trace["case_adjustment"] == 8
    assert "citation_mismatch_filtered" in trace["guardrails_applied"]
    assert "adjustment_applied_with_rationale_only" in trace["guardrails_applied"]
