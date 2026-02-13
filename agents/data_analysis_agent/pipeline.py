from __future__ import annotations

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
)
from core.schemas.case_context import StructuredCase, normalize_structured_case
from core.scoring.features import extract_features
from core.scoring.rubric import score_success
from tools.data_analysis_tools.caselaw.client import retrieve_cases
from tools.data_analysis_tools.caselaw.normalizer import normalize_cases
from tools.data_analysis_tools.caselaw.query_builder import build_queries
from tools.data_analysis_tools.caselaw.ranker import rank_cases
from tools.data_analysis_tools.caselaw.types import CaseLawDoc, RankedCaseLawDoc, RetrievalQuery


class DataAnalysisState(TypedDict, total=False):
    structured_case: StructuredCase
    queries: list[RetrievalQuery]
    raw_cases: list[dict[str, Any]]
    normalized_cases: list[CaseLawDoc]
    ranked_cases: list[RankedCaseLawDoc]
    issue_tree: IssueTree
    gap_analysis: list[GapAnalysisItem]
    recommended_actions: list[RecommendedAction]
    success_probability: dict[str, Any]
    evidence_pack: list[dict[str, Any]]
    analysis_result: AnalysisResult


def _build_query_node(state: DataAnalysisState) -> DataAnalysisState:
    structured_case = state["structured_case"]
    return {"queries": build_queries(structured_case)}


def _retrieve_node(state: DataAnalysisState) -> DataAnalysisState:
    queries = state.get("queries", [])
    return {"raw_cases": retrieve_cases(queries, limit=20)}


def _normalize_rank_node(state: DataAnalysisState) -> DataAnalysisState:
    normalized = normalize_cases(state.get("raw_cases", []))
    ranked = rank_cases(normalized, state["structured_case"], top_k=5)
    return {"normalized_cases": normalized, "ranked_cases": ranked}


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
                    description="보험사 부지급 사유의 약관/사실관계 정합성 검토",
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
    gaps: list[GapAnalysisItem] = []
    open_questions = state["structured_case"].get("open_questions", [])
    if open_questions:
        for idx, question in enumerate(open_questions, start=1):
            gaps.append(
                GapAnalysisItem(
                    issue_id=f"ISSUE-{idx}",
                    missing_evidence=question,
                    impact="medium",
                    rationale="open question 해소 여부가 재심의 논리 보강에 필요",
                )
            )
    else:
        gaps.append(
            GapAnalysisItem(
                issue_id="ISSUE-1",
                missing_evidence="치료 필요성/면책사유 배제를 입증하는 보강 진료기록",
                impact="high",
                rationale="핵심 판단 포인트를 직접 보강하는 증빙이 부족할 수 있음",
            )
        )
    return {"gap_analysis": gaps}


def _strategy_node(state: DataAnalysisState) -> DataAnalysisState:
    actions: list[RecommendedAction] = []
    for idx, gap in enumerate(state.get("gap_analysis", []), start=1):
        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{idx}",
                title="증빙 보강 및 사유별 반박 정리",
                detail=f"{gap.get('missing_evidence', '')} 관련 자료를 확보해 반박 근거를 보강",
                priority="high" if gap.get("impact") == "high" else "medium",
                linked_issue_id=gap.get("issue_id", "ISSUE-1"),
            )
        )
    return {"recommended_actions": actions}


def _score_node(state: DataAnalysisState) -> DataAnalysisState:
    rubric_path = Path("data/data_analysis_data/rubrics/success_probability_v1.yml")
    features = extract_features(state["structured_case"], state.get("ranked_cases", []))
    scored = score_success(features, str(rubric_path))
    return {"success_probability": scored}


def _evidence_node(state: DataAnalysisState) -> DataAnalysisState:
    issue_id = "ISSUE-1"
    evidence = [ranked_case_to_evidence_item(doc, issue_id) for doc in state.get("ranked_cases", [])]
    return {"evidence_pack": evidence}


def _finalize_node(state: DataAnalysisState) -> DataAnalysisState:
    result = AnalysisResult(
        issue_tree=state.get("issue_tree", IssueTree(root_title="보험금 부지급 재심의 쟁점", nodes=[])),
        gap_analysis=state.get("gap_analysis", []),
        recommended_actions=state.get("recommended_actions", []),
        success_probability=state.get(
            "success_probability",
            {
                "score": 0,
                "band": "LOW",
                "positive_drivers": [],
                "negative_drivers": [],
                "assumptions": ["insufficient input"],
            },
        ),
        evidence_pack=state.get("evidence_pack", []),
    )
    return {"analysis_result": result}


def build_graph() -> Any:
    graph = StateGraph(DataAnalysisState)
    graph.add_node("build_query", _build_query_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("normalize_rank", _normalize_rank_node)
    graph.add_node("issue_analysis", _issue_analysis_node)
    graph.add_node("gap_analysis", _gap_analysis_node)
    graph.add_node("strategy", _strategy_node)
    graph.add_node("score", _score_node)
    graph.add_node("evidence", _evidence_node)
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "build_query")
    graph.add_edge("build_query", "retrieve")
    graph.add_edge("retrieve", "normalize_rank")
    graph.add_edge("normalize_rank", "issue_analysis")
    graph.add_edge("issue_analysis", "gap_analysis")
    graph.add_edge("gap_analysis", "strategy")
    graph.add_edge("strategy", "score")
    graph.add_edge("score", "evidence")
    graph.add_edge("evidence", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_pipeline(structured_case: StructuredCase | dict[str, Any]) -> AnalysisResult:
    initial_state = DataAnalysisState(structured_case=normalize_structured_case(structured_case))
    app = build_graph()
    result_state: DataAnalysisState = app.invoke(initial_state)
    return result_state["analysis_result"]
