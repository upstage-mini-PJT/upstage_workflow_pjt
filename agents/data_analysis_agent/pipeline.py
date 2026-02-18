from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict, cast

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
from core.schemas.rag_contract import RetrievalMode
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
from tools.data_analysis_tools.rag.adjustment_engine import compute_adjustment_with_fallback


class DataAnalysisState(TypedDict, total=False):
    structured_case: StructuredCase
    retrieval_mode: RetrievalMode
    queries: list[RetrievalQuery]
    precedent_queries: list[RetrievalQuery]
    dispute_queries: list[RetrievalQuery]
    raw_cases: list[dict[str, Any]]
    raw_precedent_cases: list[dict[str, Any]]
    raw_dispute_cases: list[dict[str, Any]]
    normalized_cases: list[CaseLawDoc]
    normalized_precedent_cases: list[CaseLawDoc]
    normalized_dispute_cases: list[CaseLawDoc]
    ranked_cases: list[RankedCaseLawDoc]
    precedent_ranked_cases: list[RankedCaseLawDoc]
    dispute_ranked_cases: list[RankedCaseLawDoc]
    precedent_rag_result: RAGRetrievalResult
    dispute_rag_result: RAGRetrievalResult
    rag_result: RAGRetrievalResult
    issue_tree: IssueTree
    gap_analysis: list[GapAnalysisItem]
    recommended_actions: list[RecommendedAction]
    features: dict[str, Any]
    precedent_probability: SuccessProbability
    case_adjustment: int
    case_adjustment_source: str
    adjustment_notes: list[str]
    success_probability: SuccessProbability
    scoring_trace: ScoringTrace
    evidence_pack: list[dict[str, Any]]
    analysis_result: AnalysisResult


def _build_query_node(state: DataAnalysisState) -> DataAnalysisState:
    queries = build_queries(state["structured_case"])
    precedent_queries = [q for q in queries if str(q.get("source", "")).upper() == "CASELAW"]
    dispute_queries = [q for q in queries if str(q.get("source", "")).upper() == "DISPUTE"]
    return {
        "queries": queries,
        "precedent_queries": precedent_queries,
        "dispute_queries": dispute_queries,
    }


def _retrieve_node(state: DataAnalysisState) -> DataAnalysisState:
    raw_precedent = retrieve_cases(state.get("precedent_queries", []), limit=20)
    raw_dispute = retrieve_cases(state.get("dispute_queries", []), limit=20)
    return {
        "raw_precedent_cases": raw_precedent,
        "raw_dispute_cases": raw_dispute,
        "raw_cases": raw_precedent + raw_dispute,
    }


def _normalize_rank_node(state: DataAnalysisState) -> DataAnalysisState:
    normalized_precedent = normalize_cases(state.get("raw_precedent_cases", []))
    normalized_dispute = normalize_cases(state.get("raw_dispute_cases", []))

    precedent_ranked = rank_cases(normalized_precedent, state["structured_case"], top_k=6)
    dispute_ranked = rank_cases(normalized_dispute, state["structured_case"], top_k=6)

    precedent_query = " | ".join([q.get("query", "") for q in state.get("precedent_queries", [])]) or "보험금 부지급 판례"
    dispute_query = " | ".join([q.get("query", "") for q in state.get("dispute_queries", [])]) or "보험 분쟁 사례"

    precedent_rag = _build_rag_result_from_ranked(
        ranked=precedent_ranked,
        query=precedent_query,
        query_id=f"Q-{state['structured_case'].get('case_id', 'unknown')}-precedent",
        retrieval_mode=state.get("retrieval_mode", "plain"),
        source_filter="CASELAW",
        structured_case=state.get("structured_case", {}),
    )
    dispute_rag = _build_rag_result_from_ranked(
        ranked=dispute_ranked,
        query=dispute_query,
        query_id=f"Q-{state['structured_case'].get('case_id', 'unknown')}-dispute",
        retrieval_mode=state.get("retrieval_mode", "plain"),
        source_filter="DISPUTE",
        structured_case=state.get("structured_case", {}),
    )
    merged_rag = _merge_rag_results(precedent_rag, dispute_rag, f"Q-{state['structured_case'].get('case_id', 'unknown')}")

    return {
        "normalized_precedent_cases": normalized_precedent,
        "normalized_dispute_cases": normalized_dispute,
        "normalized_cases": normalized_precedent + normalized_dispute,
        "precedent_ranked_cases": precedent_ranked,
        "dispute_ranked_cases": dispute_ranked,
        "ranked_cases": precedent_ranked + dispute_ranked,
        "precedent_rag_result": precedent_rag,
        "dispute_rag_result": dispute_rag,
        "rag_result": merged_rag,
    }


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
    features = extract_features(state["structured_case"], state.get("precedent_ranked_cases", []))
    precedent_features = _build_precedent_only_features(features)
    precedent_probability = score_success(precedent_features, str(rubric_path))
    return {
        "features": features,
        "precedent_probability": precedent_probability,
    }


def _score_to_band(score: int) -> str:
    if score <= 39:
        return "LOW"
    if score <= 69:
        return "MEDIUM"
    return "HIGH"


def _score_adjustment_node(state: DataAnalysisState) -> DataAnalysisState:
    precedent = state.get("precedent_probability", SuccessProbability(score=0, band="LOW"))
    precedent_score = int(precedent.get("score", 0))
    dispute_rag = state.get("dispute_rag_result", RAGRetrievalResult(query_id="Q-dispute", query="", filters={}, items=[], stats={"candidate_count": 0, "returned_count": 0, "latency_ms": 0}))
    valid_doc_ids = {str(item.get("doc_id", "")) for item in dispute_rag.get("items", []) if item.get("doc_id")}

    decision = compute_adjustment_with_fallback(
        structured_case=state["structured_case"],
        dispute_ranked_cases=state.get("dispute_ranked_cases", []),
        precedent_score=precedent_score,
        features=state.get("features", {}),
        valid_doc_ids=valid_doc_ids,
    )

    scoring_trace = compute_comparative_scoring(
        ComparativeScoringInput(
            rag_result=state.get("precedent_rag_result", RAGRetrievalResult(query_id="Q-precedent", query="", filters={}, items=[], stats={"candidate_count": 0, "returned_count": 0, "latency_ms": 0})),
            case_adjustment=decision.case_adjustment,
            case_adjustment_source=decision.source,
            rationale=decision.rationale,
            cited_case_ids=decision.cited_case_ids,
        )
    )
    total_score = int(scoring_trace.get("total_score", max(0, min(100, precedent_score + decision.case_adjustment))))
    total_band = _score_to_band(total_score)

    positive_drivers = list(precedent.get("positive_drivers", []))
    negative_drivers = list(precedent.get("negative_drivers", []))
    if decision.case_adjustment > 0:
        positive_drivers.append(f"사례 기반 보정 +{decision.case_adjustment}점")
    elif decision.case_adjustment < 0:
        negative_drivers.append(f"사례 기반 보정 {decision.case_adjustment}점")

    assumptions = list(precedent.get("assumptions", [])) + list(decision.notes) + [
        f"precedent_score={precedent_score}",
        f"case_adjustment={decision.case_adjustment}",
        f"case_adjustment_source={decision.source}",
        f"total_score={total_score}",
    ]

    return {
        "case_adjustment": decision.case_adjustment,
        "case_adjustment_source": decision.source,
        "adjustment_notes": decision.notes,
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
        scoring_trace=state.get("scoring_trace", ScoringTrace(precedent_score=0, case_adjustment=0, case_adjustment_source="zero", total_score=0, guardrails_applied=["missing score trace"], cited_case_ids=[])),
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


def _build_rag_result_from_ranked(
    *,
    ranked: list[RankedCaseLawDoc],
    query: str,
    query_id: str,
    retrieval_mode: RetrievalMode,
    source_filter: str,
    structured_case: StructuredCase,
) -> RAGRetrievalResult:
    ingestion_inputs: list[dict[str, Any]] = []
    for doc in ranked:
        ingestion_inputs.append(
            {
                "source_type": doc.get("source_type", source_filter),
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title", ""),
                "body": doc.get("summary") or doc.get("holding") or "",
                "published_at": doc.get("published_at"),
                "tags": doc.get("keywords", []),
                "url": doc.get("url"),
                "meta": {"result": doc.get("result", ""), "source": doc.get("source", "")},
            }
        )
    if not ingestion_inputs:
        return RAGRetrievalResult(
            query_id=query_id,
            query=query,
            retrieval_mode=retrieval_mode,
            query_variants=["plain"] if retrieval_mode == "plain" else [str(retrieval_mode)],
            filters={"source_type": source_filter},
            items=[],
            stats={"candidate_count": 0, "returned_count": 0, "latency_ms": 0},
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
            query=query,
            top_k=6,
            query_id=query_id,
            filters={"source_type": source_filter},
            retrieval_mode=retrieval_mode,
            structured_case=structured_case,
        )
    )
    return RAGRetrievalResult(**rerank_retrieval_result(retrieved))


def _merge_rag_results(precedent: RAGRetrievalResult, dispute: RAGRetrievalResult, query_id: str) -> RAGRetrievalResult:
    items = list(precedent.get("items", [])) + list(dispute.get("items", []))
    items.sort(key=lambda x: float(x.get("rerank_score", x.get("score", 0.0))), reverse=True)
    merged_items = items[:6]
    return RAGRetrievalResult(
        query_id=query_id,
        query=f"{precedent.get('query','')} || {dispute.get('query','')}",
        retrieval_mode=precedent.get("retrieval_mode", "plain"),
        query_variants=list(
            dict.fromkeys(list(precedent.get("query_variants", [])) + list(dispute.get("query_variants", [])))
        ),
        filters={},
        items=merged_items,
        stats={
            "candidate_count": int(precedent.get("stats", {}).get("candidate_count", 0))
            + int(dispute.get("stats", {}).get("candidate_count", 0)),
            "returned_count": len(merged_items),
            "latency_ms": int(precedent.get("stats", {}).get("latency_ms", 0))
            + int(dispute.get("stats", {}).get("latency_ms", 0)),
        },
    )


def _resolve_retrieval_mode(mode: str | None) -> RetrievalMode:
    raw = (mode or os.getenv("RAG_RETRIEVAL_MODE", "plain")).strip().lower()
    if raw in {"plain", "hyde", "reverse_hyde", "hybrid_hyde"}:
        return cast(RetrievalMode, raw)
    return "plain"


def run_pipeline(
    structured_case: StructuredCase | dict[str, Any],
    retrieval_mode: RetrievalMode | None = None,
) -> AnalysisResult:
    initial_state = DataAnalysisState(
        structured_case=normalize_structured_case(structured_case),
        retrieval_mode=_resolve_retrieval_mode(retrieval_mode),
    )
    app = build_graph()
    result_state: DataAnalysisState = app.invoke(initial_state)
    return result_state["analysis_result"]
