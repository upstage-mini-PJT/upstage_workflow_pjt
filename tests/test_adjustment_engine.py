import os

from tools.data_analysis_tools.rag.adjustment_engine import (
    compute_adjustment_with_fallback,
)


def _sample_ranked_cases():
    return [
        {"doc_id": "DISPUTE:FSS-1", "title": "사례1", "relevance_score": 0.82, "result": "조정 성립"},
        {"doc_id": "DISPUTE:FSS-2", "title": "사례2", "relevance_score": 0.65, "result": "일부 인용"},
    ]


def test_llm_success_used(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_LLM_ADJUSTMENT", "true")
    monkeypatch.setenv("RAG_ADJUSTMENT_FALLBACK", "heuristic")
    monkeypatch.setenv(
        "RAG_LLM_ADJUSTMENT_MOCK_JSON",
        '{"case_adjustment": 7, "rationale": "LLM 판단", "cited_case_ids": ["DISPUTE:FSS-1"]}',
    )
    decision = compute_adjustment_with_fallback(
        structured_case={"denial_summary": "부지급", "denial_reasons": ["면책"]},
        dispute_ranked_cases=_sample_ranked_cases(),
        precedent_score=55,
        features={"evidence_completeness": 0.8, "top_relevance": 0.7, "open_question_count": 0},
        valid_doc_ids={"DISPUTE:FSS-1", "DISPUTE:FSS-2"},
    )
    assert decision.source == "llm"
    assert decision.case_adjustment == 7


def test_llm_fail_heuristic_fallback(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_LLM_ADJUSTMENT", "true")
    monkeypatch.setenv("RAG_ADJUSTMENT_FALLBACK", "heuristic")
    monkeypatch.delenv("RAG_LLM_ADJUSTMENT_MOCK_JSON", raising=False)
    decision = compute_adjustment_with_fallback(
        structured_case={"denial_summary": "부지급", "denial_reasons": ["면책"]},
        dispute_ranked_cases=_sample_ranked_cases(),
        precedent_score=60,
        features={"evidence_completeness": 0.7, "top_relevance": 0.7, "open_question_count": 0},
        valid_doc_ids={"DISPUTE:FSS-1", "DISPUTE:FSS-2"},
    )
    assert decision.source == "heuristic"
    assert "llm_failed_heuristic_fallback" in decision.notes
    assert decision.case_adjustment != 0


def test_llm_fail_zero_fallback(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_LLM_ADJUSTMENT", "true")
    monkeypatch.setenv("RAG_ADJUSTMENT_FALLBACK", "zero")
    monkeypatch.delenv("RAG_LLM_ADJUSTMENT_MOCK_JSON", raising=False)
    decision = compute_adjustment_with_fallback(
        structured_case={"denial_summary": "부지급", "denial_reasons": ["면책"]},
        dispute_ranked_cases=_sample_ranked_cases(),
        precedent_score=60,
        features={"evidence_completeness": 0.7, "top_relevance": 0.7, "open_question_count": 0},
        valid_doc_ids={"DISPUTE:FSS-1", "DISPUTE:FSS-2"},
    )
    assert decision.case_adjustment == 0
    assert "fallback_zero_adjustment_applied" in decision.notes


def test_citation_mismatch_invalidates_adjustment(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_LLM_ADJUSTMENT", "true")
    monkeypatch.setenv(
        "RAG_LLM_ADJUSTMENT_MOCK_JSON",
        '{"case_adjustment": 8, "rationale": "근거 있음", "cited_case_ids": ["DISPUTE:UNKNOWN"]}',
    )
    decision = compute_adjustment_with_fallback(
        structured_case={"denial_summary": "부지급", "denial_reasons": ["면책"]},
        dispute_ranked_cases=_sample_ranked_cases(),
        precedent_score=60,
        features={"evidence_completeness": 0.7, "top_relevance": 0.7, "open_question_count": 0},
        valid_doc_ids={"DISPUTE:FSS-1", "DISPUTE:FSS-2"},
    )
    assert decision.case_adjustment == 0
    assert "adjustment_invalidated_citation_mismatch" in decision.notes


def teardown_module() -> None:
    os.environ.pop("RAG_LLM_ADJUSTMENT_MOCK_JSON", None)
