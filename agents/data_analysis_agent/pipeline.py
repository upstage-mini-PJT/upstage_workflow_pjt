from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
import yaml

from agents.data_analysis_agent.adapters import ranked_case_to_evidence_item
from core.schemas.analysis import (
    AnalysisResult,
    GapAnalysisItem,
    IssueNode,
    IssueTree,
    RecommendedAction,
    SuccessProbability,
    ScoringTrace,
    RAGRetrievalResult,
)
from core.schemas.case_context import StructuredCase, normalize_structured_case
from core.scoring.features import extract_features
from core.scoring.rubric import score_success
from tools.data_analysis_tools.caselaw.client import retrieve_cases
from tools.data_analysis_tools.caselaw.normalizer import normalize_cases
from tools.data_analysis_tools.caselaw.query_builder import build_queries
from tools.data_analysis_tools.caselaw.ranker import rank_cases
from tools.data_analysis_tools.caselaw.types import CaseLawDoc, RankedCaseLawDoc, RetrievalQuery
from tools.data_analysis_tools.rag import (
    ChunkingConfig,
    ChromaVectorIndexStore,
    ComparativeScoringInput,
    HashingEmbedder,
    InMemoryVectorIndexStore,
    RetrieveRequest,
    VectorRetriever,
    chunk_ingestion_docs,
    compute_comparative_scoring,
    embed_chunks,
    normalize_ingestion_docs,
    rerank_retrieval_result,
)


class DataAnalysisState(TypedDict, total=False):
    structured_case: StructuredCase
    queries: list[RetrievalQuery]
    raw_cases: list[dict[str, Any]]
    normalized_cases: list[CaseLawDoc]
    ranked_cases: list[RankedCaseLawDoc]
    rag_result: RAGRetrievalResult
    issue_tree: IssueTree
    gap_analysis: list[GapAnalysisItem]
    recommended_actions: list[RecommendedAction]
    features: dict[str, Any]
    precedent_probability: SuccessProbability
    case_adjustment: int
    adjustment_notes: list[str]
    success_probability: SuccessProbability
    scoring_trace: ScoringTrace
    evidence_pack: list[dict[str, Any]]
    analysis_result: AnalysisResult


def _build_query_node(state: DataAnalysisState) -> DataAnalysisState:
    return {"queries": build_queries(state["structured_case"])}


def _retrieve_node(state: DataAnalysisState) -> DataAnalysisState:
    return {"raw_cases": retrieve_cases(state.get("queries", []), limit=20)}


def _normalize_rank_node(state: DataAnalysisState) -> DataAnalysisState:
    normalized = normalize_cases(state.get("raw_cases", []))
    ranked = rank_cases(normalized, state["structured_case"], top_k=6)

    merged_query = " | ".join([q.get("query", "") for q in state.get("queries", [])])
    ingestion_inputs: list[dict[str, Any]] = []
    for doc in ranked:
        ingestion_inputs.append(
            {
                "source_type": doc.get("source_type", "CASELAW"),
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title", ""),
                "body": doc.get("summary") or doc.get("holding") or "",
                "published_at": doc.get("published_at"),
                "tags": doc.get("keywords", []),
                "url": doc.get("url"),
                "meta": {"result": doc.get("result", ""), "source": doc.get("source", "caselaw")},
            }
        )

    ingestion_docs = normalize_ingestion_docs(ingestion_inputs)
    embedder = HashingEmbedder(embedding_dim=256)
    chunks = chunk_ingestion_docs(
        ingestion_docs,
        ChunkingConfig(
            chunk_size_tokens=300,
            chunk_overlap_tokens=60,
            embedding_model=embedder.model_name,
            embedding_dim=embedder.embedding_dim,
        ),
    )
    vectors = embed_chunks(chunks, embedder)
    index_store = _build_vector_index_store()
    index_store.upsert(chunks, vectors)

    retriever = VectorRetriever(index_store=index_store, embedder=embedder)
    retrieved = retriever.retrieve(
        RetrieveRequest(
            query=merged_query or "보험금 부지급",
            top_k=6,
            query_id=f"Q-{state['structured_case'].get('case_id', 'unknown')}",
            filters={},
        )
    )
    rag_result = RAGRetrievalResult(**rerank_retrieval_result(retrieved))
    return {"normalized_cases": normalized, "ranked_cases": ranked, "rag_result": rag_result}


def _build_vector_index_store() -> Any:
    backend = os.getenv("RAG_VECTOR_BACKEND", "inmemory").strip().lower()
    index_name = os.getenv("RAG_INDEX_NAME", "step3_rag_index").strip() or "step3_rag_index"
    if backend == "chroma":
        persist_dir = os.getenv("RAG_CHROMA_DIR", ".chroma_db").strip() or ".chroma_db"
        return ChromaVectorIndexStore(index_name=index_name, persist_directory=persist_dir)
    return InMemoryVectorIndexStore(index_name=index_name)


def _issue_analysis_node(state: DataAnalysisState) -> DataAnalysisState:
    structured_case = state["structured_case"]
    reasons = structured_case.get("denial_reasons", [])
    clauses = structured_case.get("policy_clauses", [])

    mapping_path = Path("data/data_analysis_data/rubrics/issue_mapping.yml")
    issue_mapping: dict[str, str] = {}
    if mapping_path.exists():
        try:
            loaded = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
            issue_mapping = loaded.get("issue_mapping", {}) if isinstance(loaded, dict) else {}
        except Exception:
            issue_mapping = {}

    nodes: list[IssueNode] = []
    if reasons:
        for idx, reason in enumerate(reasons, start=1):
            nodes.append(
                IssueNode(
                    issue_id=f"ISSUE-{idx}",
                    title=issue_mapping.get(reason, f"부지급 사유 쟁점: {reason}"),
                    description="부지급 사유의 약관/사실 정합성 및 유사사례 적합성 검토",
                    related_denial_reasons=[reason],
                    related_policy_clauses=clauses,
                )
            )
    else:
        nodes.append(
            IssueNode(
                issue_id="ISSUE-1",
                title="부지급 사유 해석 쟁점",
                description="부지급 사유와 약관 해석의 충돌 여부 점검",
                related_denial_reasons=[],
                related_policy_clauses=clauses,
            )
        )

    return {"issue_tree": IssueTree(root_title="보험금 부지급 재심의 쟁점", nodes=nodes)}


def _gap_analysis_node(state: DataAnalysisState) -> DataAnalysisState:
    structured_case = state["structured_case"]
    gaps: list[GapAnalysisItem] = []
    open_questions = structured_case.get("open_questions", [])

    if open_questions:
        for idx, q in enumerate(open_questions, start=1):
            gaps.append(
                GapAnalysisItem(
                    issue_id=f"ISSUE-{idx}",
                    missing_evidence=q,
                    impact="medium",
                    rationale="미해결 질문 해소가 반박 논리 완결성에 직접 영향",
                )
            )
    else:
        evidence_text = " ".join(
            str(x.get("title", "")) + " " + str(x.get("summary", ""))
            for x in structured_case.get("evidence_summary", [])
        )
        if "진단서" not in evidence_text and "소견서" not in evidence_text:
            gaps.append(
                GapAnalysisItem(
                    issue_id="ISSUE-1",
                    missing_evidence="치료 필요성 입증 소견서",
                    impact="high",
                    rationale="의학적 필요성 부지급 사유를 직접 반박하기 위한 핵심 증빙",
                )
            )
        gaps.append(
            GapAnalysisItem(
                issue_id="ISSUE-1",
                missing_evidence="진료기록/입퇴원기록 보강",
                impact="medium",
                rationale="사실관계 타임라인 일치성과 치료 경과를 강화",
            )
        )

    return {"gap_analysis": gaps}


def _build_precedent_only_features(features: dict[str, Any]) -> dict[str, Any]:
    precedent_only = dict(features)
    neutral_defaults = {
        "evidence_completeness": 0.4,
        "has_medical_records": None,
        "has_doctor_note": None,
        "open_question_count": 1,
        "timeline_consistency": 0.7,
        "denial_reason_count": max(1, int(features.get("denial_reason_count", 1))),
        "policy_clause_count": max(1, int(features.get("policy_clause_count", 1))),
    }
    precedent_only.update(neutral_defaults)
    return precedent_only


def _score_precedent_node(state: DataAnalysisState) -> DataAnalysisState:
    rubric_path = Path("data/data_analysis_data/rubrics/success_probability_v1.yml")
    features = extract_features(state["structured_case"], state.get("ranked_cases", []))
    precedent_features = _build_precedent_only_features(features)
    precedent_probability = score_success(precedent_features, str(rubric_path))
    return {
        "features": features,
        "precedent_probability": precedent_probability,
    }


def _heuristic_case_adjustment(payload: dict[str, Any]) -> dict[str, Any]:
    top_cases = payload.get("top_cases", [])
    if not top_cases:
        return {"adjustment": 0, "rationale": "사례 없음", "cited_case_ids": []}

    adjustment = 0
    top_rel = float(top_cases[0].get("relevance_score", 0.0) or 0.0)
    if top_rel >= 0.8:
        adjustment += 6
    elif top_rel >= 0.6:
        adjustment += 3
    elif top_rel < 0.3:
        adjustment -= 3

    win_like = 0
    lose_like = 0
    reliable_cases = [row for row in top_cases if float(row.get("relevance_score", 0.0) or 0.0) >= 0.25]
    for row in reliable_cases:
        result = str(row.get("result", "") or "")
        if any(k in result for k in ["승소", "인용", "조정 성립"]):
            win_like += 1
        if any(k in result for k in ["패소", "기각", "각하", "불수용"]):
            lose_like += 1

    if win_like >= 2:
        adjustment += 4
    elif lose_like >= 2:
        adjustment -= 4

    cited = [str(row.get("doc_id", "")) for row in top_cases[:2] if row.get("doc_id")]
    return {
        "adjustment": adjustment,
        "rationale": "상위 사례 관련도/판정 경향 기반 보정",
        "cited_case_ids": cited,
    }


def _apply_adjustment_guardrails(
    raw_adjustment: Any,
    rationale: str,
    cited_case_ids: list[str],
    ranked_cases: list[RankedCaseLawDoc],
    precedent_score: int,
) -> tuple[int, list[str], list[str]]:
    notes: list[str] = []

    try:
        adjustment = int(round(float(raw_adjustment)))
    except Exception:
        adjustment = 0
        notes.append("사례 보정치 파싱 실패: 0으로 대체")

    valid_ids = {str(c.get("doc_id", "")) for c in ranked_cases}

    if not rationale or len(rationale.strip()) < 8:
        adjustment = 0
        notes.append("사례 보정치 무효화: rationale 부족")

    if not cited_case_ids or any(cid not in valid_ids for cid in cited_case_ids):
        adjustment = 0
        notes.append("사례 보정치 무효화: 유효 provenance 부재")

    if adjustment > 15:
        adjustment = 15
        notes.append("사례 보정치 상한 적용(+15)")
    if adjustment < -15:
        adjustment = -15
        notes.append("사례 보정치 하한 적용(-15)")

    if precedent_score < 20 and adjustment > 8:
        adjustment = 8
        notes.append("저신뢰 1차 점수 보호: +8로 제한")
    if precedent_score > 85 and adjustment < -8:
        adjustment = -8
        notes.append("고신뢰 1차 점수 보호: -8로 제한")
    if precedent_score < 50 and adjustment > 1:
        adjustment = 1
        notes.append("중저신뢰 1차 점수 보호: 양의 보정 +1로 제한")

    valid_cited = [cid for cid in cited_case_ids if cid in valid_ids]
    return adjustment, notes, valid_cited


def _score_to_band(score: int) -> str:
    if score <= 39:
        return "LOW"
    if score <= 69:
        return "MEDIUM"
    return "HIGH"


def _compute_case_adjustment(
    structured_case: StructuredCase,
    ranked_cases: list[RankedCaseLawDoc],
    precedent_score: int,
    features: dict[str, Any],
) -> tuple[int, list[str], list[str], str]:
    if not ranked_cases:
        return 0, ["사례 기반 보정 미적용: 검색된 사례 없음"], [], "사례 없음"

    payload = {
        "case": {
            "denial_summary": structured_case.get("denial_summary", ""),
            "denial_reasons": structured_case.get("denial_reasons", []),
        },
        "top_cases": [
            {
                "doc_id": c.get("doc_id", ""),
                "title": c.get("title", ""),
                "relevance_score": c.get("relevance_score", 0.0),
                "result": c.get("result", ""),
            }
            for c in ranked_cases[:3]
        ],
        "precedent_score": precedent_score,
    }

    raw = _heuristic_case_adjustment(payload)
    raw_adjustment = int(raw.get("adjustment", 0))
    adjustment_notes: list[str] = []

    if float(features.get("evidence_completeness", 0.0)) >= 0.9 and float(features.get("top_relevance", 0.0)) >= 0.3:
        raw_adjustment += 4
        adjustment_notes.append("증빙 완결성과 근거 관련도 반영: +4")
    if float(features.get("evidence_completeness", 0.0)) < 0.3 and int(features.get("open_question_count", 0)) >= 2:
        raw_adjustment -= 10
        adjustment_notes.append("증빙 부족/미해결 쟁점 다수 반영: -10")

    adjustment, notes, cited_ids = _apply_adjustment_guardrails(
        raw_adjustment=raw_adjustment,
        rationale=str(raw.get("rationale", "")),
        cited_case_ids=[str(x) for x in raw.get("cited_case_ids", [])],
        ranked_cases=ranked_cases,
        precedent_score=precedent_score,
    )
    rationale_parts = [str(raw.get("rationale", "")).strip()]
    if adjustment_notes:
        rationale_parts.append(" / ".join(adjustment_notes))
    rationale_text = " | ".join([p for p in rationale_parts if p]) or "사례 기반 휴리스틱 보정"
    return adjustment, adjustment_notes + notes, cited_ids, rationale_text


def _score_adjustment_node(state: DataAnalysisState) -> DataAnalysisState:
    precedent = state.get("precedent_probability", SuccessProbability(score=0, band="LOW"))
    precedent_score = int(precedent.get("score", 0))

    adjustment, notes, cited_ids, rationale_text = _compute_case_adjustment(
        state["structured_case"],
        state.get("ranked_cases", []),
        precedent_score,
        state.get("features", {}),
    )

    scoring_trace = compute_comparative_scoring(
        ComparativeScoringInput(
            rag_result=state.get("rag_result", RAGRetrievalResult(query_id="Q-unknown", query="", filters={}, items=[], stats={"candidate_count": 0, "returned_count": 0, "latency_ms": 0})),
            case_adjustment=adjustment,
            rationale=rationale_text,
            cited_case_ids=cited_ids,
        )
    )
    total_score = int(scoring_trace.get("total_score", max(0, min(100, precedent_score + adjustment))))
    total_band = _score_to_band(total_score)

    positive_drivers = list(precedent.get("positive_drivers", []))
    negative_drivers = list(precedent.get("negative_drivers", []))
    if adjustment > 0:
        positive_drivers.append(f"사례 기반 보정 +{adjustment}점")
    elif adjustment < 0:
        negative_drivers.append(f"사례 기반 보정 {adjustment}점")

    assumptions = list(precedent.get("assumptions", [])) + notes + [
        f"precedent_score={precedent_score}",
        f"case_adjustment={adjustment}",
        f"total_score={total_score}",
    ]

    return {
        "case_adjustment": adjustment,
        "adjustment_notes": notes,
        "scoring_trace": ScoringTrace(**scoring_trace),
        "success_probability": SuccessProbability(
            score=total_score,
            band=total_band,
            positive_drivers=positive_drivers,
            negative_drivers=negative_drivers,
            assumptions=assumptions,
        ),
    }


def _strategy_node(state: DataAnalysisState) -> DataAnalysisState:
    actions: list[RecommendedAction] = []
    for idx, gap in enumerate(state.get("gap_analysis", []), start=1):
        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{idx}",
                title="증빙 보강 및 사유별 반박 정리",
                detail=f"{gap.get('missing_evidence', '')} 자료를 보강하고 쟁점별 반박 논리를 정리",
                priority="high" if gap.get("impact") == "high" else "medium",
                linked_issue_id=gap.get("issue_id", "ISSUE-1"),
            )
        )

    band = state.get("success_probability", {}).get("band", "LOW")
    if band != "HIGH":
        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{len(actions)+1}",
                title="재심의 제출 전략 강화",
                detail="핵심 쟁점별 1페이지 요약서를 첨부해 심사자의 판단 부담을 낮춤",
                priority="medium",
                linked_issue_id="ISSUE-1",
            )
        )

    return {"recommended_actions": actions}


def _package_evidence_node(state: DataAnalysisState) -> DataAnalysisState:
    issue_id = "ISSUE-1"
    nodes = state.get("issue_tree", {}).get("nodes", [])
    if nodes:
        issue_id = str(nodes[0].get("issue_id", "ISSUE-1"))

    evidence = [ranked_case_to_evidence_item(doc, issue_id) for doc in state.get("ranked_cases", [])]
    return {"evidence_pack": evidence}


def _finalize_node(state: DataAnalysisState) -> DataAnalysisState:
    result = AnalysisResult(
        issue_tree=state.get("issue_tree", IssueTree(root_title="보험금 부지급 재심의 쟁점", nodes=[])),
        gap_analysis=state.get("gap_analysis", []),
        recommended_actions=state.get("recommended_actions", []),
        success_probability=state.get(
            "success_probability",
            SuccessProbability(
                score=0,
                band="LOW",
                positive_drivers=[],
                negative_drivers=[],
                assumptions=["insufficient input"],
            ),
        ),
        evidence_pack=state.get("evidence_pack", []),
        rag_result=state.get("rag_result", RAGRetrievalResult(query_id="Q-unknown", query="", filters={}, items=[], stats={"candidate_count": 0, "returned_count": 0, "latency_ms": 0})),
        scoring_trace=state.get("scoring_trace", ScoringTrace(precedent_score=0, case_adjustment=0, total_score=0, guardrails_applied=["missing score trace"], cited_case_ids=[])),
    )
    return {"analysis_result": result}


def build_graph() -> Any:
    graph = StateGraph(DataAnalysisState)
    graph.add_node("build_query", _build_query_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("normalize_rank", _normalize_rank_node)
    graph.add_node("issue_analysis", _issue_analysis_node)
    graph.add_node("gap_analysis", _gap_analysis_node)
    graph.add_node("score_precedent", _score_precedent_node)
    graph.add_node("score_adjustment", _score_adjustment_node)
    graph.add_node("strategy", _strategy_node)
    graph.add_node("package_evidence", _package_evidence_node)
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "build_query")
    graph.add_edge("build_query", "retrieve")
    graph.add_edge("retrieve", "normalize_rank")
    graph.add_edge("normalize_rank", "issue_analysis")
    graph.add_edge("issue_analysis", "gap_analysis")
    graph.add_edge("gap_analysis", "score_precedent")
    graph.add_edge("score_precedent", "score_adjustment")
    graph.add_edge("score_adjustment", "strategy")
    graph.add_edge("strategy", "package_evidence")
    graph.add_edge("package_evidence", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_pipeline(structured_case: StructuredCase | dict[str, Any]) -> AnalysisResult:
    initial_state = DataAnalysisState(structured_case=normalize_structured_case(structured_case))
    app = build_graph()
    result_state: DataAnalysisState = app.invoke(initial_state)
    return result_state["analysis_result"]
