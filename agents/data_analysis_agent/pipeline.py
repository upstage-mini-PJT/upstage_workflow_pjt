from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
import yaml

from agents.data_analysis_agent.adapters import ranked_case_to_evidence_item
from core.schemas.analysis import (
    AnalysisResult,
    GapAnalysisItem,
    IssueNode,
    IssueTree,
    RecommendedAction,
    SuccessProbability,
)
from core.schemas.case_context import StructuredCase, normalize_structured_case
from core.scoring.features import extract_features
from core.scoring.rubric import ScoreEstimate, score_success
from tools.data_analysis_tools.caselaw.client import retrieve_cases
from tools.data_analysis_tools.caselaw.normalizer import normalize_cases
from tools.data_analysis_tools.caselaw.query_builder import build_queries
from tools.data_analysis_tools.caselaw.ranker import rank_cases
from tools.data_analysis_tools.caselaw.types import CaseLawDoc, RankedCaseLawDoc, RetrievalQuery


class DataAnalysisState(TypedDict, total=False):
    structured_case: StructuredCase
    analysis_options: dict[str, Any]
    rag_result: dict[str, Any]
    queries: list[RetrievalQuery]
    raw_cases: list[dict[str, Any]]
    normalized_cases: list[CaseLawDoc]
    ranked_cases: list[RankedCaseLawDoc]
    issue_tree: IssueTree
    gap_analysis: list[GapAnalysisItem]
    recommended_actions: list[RecommendedAction]
    features: dict[str, Any]
    precedent_probability: ScoreEstimate
    case_adjustment: int
    adjustment_notes: list[str]
    internal_scoring: dict[str, Any]
    scoring_trace: dict[str, Any]
    success_probability_public: SuccessProbability
    strategy_context: dict[str, Any]
    synthesized_summary: dict[str, Any]
    quality_flags: list[str]
    llm_enrichment_trace: dict[str, Any]
    llm_raw_output: str
    evidence_pack: list[dict[str, Any]]
    analysis_result: AnalysisResult


def _build_query_node(state: DataAnalysisState) -> DataAnalysisState:
    return {"queries": build_queries(state["structured_case"])}


def _retrieve_node(state: DataAnalysisState) -> DataAnalysisState:
    return {"raw_cases": retrieve_cases(state.get("queries", []), limit=20)}


def _normalize_rank_node(state: DataAnalysisState) -> DataAnalysisState:
    normalized = normalize_cases(state.get("raw_cases", []))
    ranked = rank_cases(normalized, state["structured_case"], top_k=6)
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

    valid_cited_ids = [cid for cid in cited_case_ids if cid in valid_ids]
    if not valid_cited_ids:
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

    return adjustment, notes, valid_cited_ids


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
) -> tuple[int, list[str], list[str]]:
    if not ranked_cases:
        return 0, ["사례 기반 보정 미적용: 검색된 사례 없음"], []

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

    adjustment, notes, valid_cited_ids = _apply_adjustment_guardrails(
        raw_adjustment=raw_adjustment,
        rationale=str(raw.get("rationale", "")),
        cited_case_ids=[str(x) for x in raw.get("cited_case_ids", [])],
        ranked_cases=ranked_cases,
        precedent_score=precedent_score,
    )
    return adjustment, adjustment_notes + notes, valid_cited_ids


def _score_adjustment_node(state: DataAnalysisState) -> DataAnalysisState:
    precedent = state.get("precedent_probability", ScoreEstimate(score=0, band="LOW"))
    precedent_score = int(precedent.get("score", 0))

    adjustment, notes, valid_cited_ids = _compute_case_adjustment(
        state["structured_case"],
        state.get("ranked_cases", []),
        precedent_score,
        state.get("features", {}),
    )

    return {
        "case_adjustment": adjustment,
        "adjustment_notes": notes,
        "internal_scoring": {
            "precedent_score": precedent_score,
            "case_adjustment": adjustment,
            "total_score": max(0, min(100, precedent_score + adjustment)),
        },
        "scoring_trace": {
            "precedent_score": precedent_score,
            "case_adjustment": adjustment,
            "total_score": max(0, min(100, precedent_score + adjustment)),
            "guardrails_applied": notes,
            "cited_case_ids": valid_cited_ids,
        },
    }


def _estimate_success_probability_node(state: DataAnalysisState) -> DataAnalysisState:
    precedent = state.get("precedent_probability", ScoreEstimate(score=0, band="LOW", positive_drivers=[], negative_drivers=[], assumptions=[]))
    internal_scoring = state.get("internal_scoring", {})

    precedent_score = int(internal_scoring.get("precedent_score", precedent.get("score", 0) or 0))
    case_adjustment = int(internal_scoring.get("case_adjustment", state.get("case_adjustment", 0) or 0))
    total_score = int(internal_scoring.get("total_score", max(0, min(100, precedent_score + case_adjustment))))
    band = _score_to_band(total_score)

    positive_drivers = list(precedent.get("positive_drivers", []))
    negative_drivers = list(precedent.get("negative_drivers", []))
    if case_adjustment > 0:
        positive_drivers.append(f"사례 기반 보정 +{case_adjustment}점")
    elif case_adjustment < 0:
        negative_drivers.append(f"사례 기반 보정 {case_adjustment}점")

    assumptions = list(precedent.get("assumptions", [])) + list(state.get("adjustment_notes", []))

    return {
        "success_probability_public": SuccessProbability(
            band=band,
            positive_drivers=positive_drivers,
            negative_drivers=negative_drivers,
            assumptions=assumptions,
        ),
        "scoring_trace": {
            **state.get("scoring_trace", {}),
            "precedent_score": precedent_score,
            "case_adjustment": case_adjustment,
            "total_score": total_score,
        },
    }


def _normalize_rag_result(rag_result: dict[str, Any] | None) -> dict[str, Any]:
    if not rag_result or not isinstance(rag_result, dict):
        return {"items": [], "stats": {}}

    items = []
    for raw in rag_result.get("items", []):
        source_type = str(raw.get("source_type", "CASELAW")).upper()
        if source_type not in {"CASELAW", "DISPUTE", "WEB"}:
            source_type = "CASELAW"

        item = {
            "source_type": source_type,
            "doc_id": str(raw.get("doc_id", "")),
            "chunk_id": str(raw.get("chunk_id", "")),
            "title": str(raw.get("title", "")),
            "snippet": str(raw.get("snippet", "")),
            "score": float(raw.get("score", 0.0) or 0.0),
            "rerank_score": float(raw.get("rerank_score", 0.0) or 0.0),
            "url": raw.get("url"),
            "published_at": raw.get("published_at"),
            "provenance": dict(raw.get("provenance", {})),
        }
        items.append(item)

    return {
        "query_id": str(rag_result.get("query_id", "")),
        "query": str(rag_result.get("query", "")),
        "filters": dict(rag_result.get("filters", {})),
        "items": items,
        "stats": dict(rag_result.get("stats", {})),
    }


def _select_strategy_evidence(rag_items: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    if not rag_items:
        return []

    source_weight = {"CASELAW": 1.0, "DISPUTE": 0.85, "WEB": 0.5}

    ranked = []
    for item in rag_items:
        rerank = float(item.get("rerank_score", 0.0) or 0.0)
        score = float(item.get("score", 0.0) or 0.0)
        source_type = str(item.get("source_type", "CASELAW"))
        weighted = (rerank * 0.75 + score * 0.25) * source_weight.get(source_type, 0.7)

        copied = dict(item)
        copied["weighted_rank"] = round(weighted, 6)
        ranked.append(copied)

    ranked.sort(key=lambda x: x.get("weighted_rank", 0.0), reverse=True)
    return ranked[:top_n]


def _group_evidence_by_issue(issues: list[IssueNode], evidence: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = {}
    if not issues:
        return mapping

    for issue in issues:
        issue_id = str(issue.get("issue_id", "ISSUE-1"))
        issue_terms = " ".join(
            [
                str(issue.get("title", "")),
                str(issue.get("description", "")),
                " ".join(issue.get("related_denial_reasons", [])),
                " ".join(issue.get("related_policy_clauses", [])),
            ]
        )

        matched: list[dict[str, Any]] = []
        for item in evidence:
            blob = " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("snippet", "")),
                    str(item.get("doc_id", "")),
                ]
            )
            if any(term and term in blob for term in issue_terms.split()):
                matched.append(item)

        if not matched:
            matched = evidence[:2]
        mapping[issue_id] = matched[:2]

    return mapping


def _build_strategy_context_node(state: DataAnalysisState) -> DataAnalysisState:
    issue_nodes = state.get("issue_tree", {}).get("nodes", [])
    rag_result = _normalize_rag_result(state.get("rag_result", {}))

    if rag_result.get("items"):
        top_evidence = _select_strategy_evidence(rag_result.get("items", []), top_n=5)
    else:
        fallback_items: list[dict[str, Any]] = []
        for row in state.get("ranked_cases", []):
            fallback_items.append(
                {
                    "source_type": "CASELAW" if row.get("source_type", "caselaw") == "caselaw" else "DISPUTE",
                    "doc_id": row.get("doc_id", ""),
                    "chunk_id": "",
                    "title": row.get("title", ""),
                    "snippet": row.get("summary", ""),
                    "score": float(row.get("relevance_score", 0.0)),
                    "rerank_score": float(row.get("relevance_score", 0.0)),
                    "url": None,
                    "published_at": None,
                    "provenance": {},
                }
            )
        top_evidence = _select_strategy_evidence(fallback_items, top_n=5)

    issue_evidence_map = _group_evidence_by_issue(issue_nodes, top_evidence)

    coverage = {}
    risk_flags: list[str] = []
    for issue in issue_nodes:
        issue_id = str(issue.get("issue_id", "ISSUE-1"))
        count = len(issue_evidence_map.get(issue_id, []))
        coverage[issue_id] = count
        if count == 0:
            risk_flags.append(f"{issue_id}:근거없음")
        elif count == 1:
            risk_flags.append(f"{issue_id}:근거부족")

    if not top_evidence:
        risk_flags.append("global:검색근거없음")

    return {
        "strategy_context": {
            "top_evidence": top_evidence,
            "issue_evidence_map": issue_evidence_map,
            "coverage": coverage,
            "risk_flags": risk_flags,
        }
    }


def _strategy_node(state: DataAnalysisState) -> DataAnalysisState:
    actions: list[RecommendedAction] = []
    strategy_context = state.get("strategy_context", {})
    issue_evidence_map = strategy_context.get("issue_evidence_map", {})
    risk_flags = strategy_context.get("risk_flags", [])
    band = state.get("success_probability_public", {}).get("band", "LOW")
    risk_level = str(state.get("analysis_options", {}).get("risk_level", "BALANCED")).upper()

    for idx, gap in enumerate(state.get("gap_analysis", []), start=1):
        issue_id = gap.get("issue_id", "ISSUE-1")
        evidence = issue_evidence_map.get(issue_id, [])[:2]
        evidence_hint = ""
        if evidence:
            refs = []
            for ev in evidence:
                doc_id = str(ev.get("doc_id", ""))
                source = str(ev.get("source_type", "CASELAW"))
                if not doc_id:
                    continue
                refs.append(doc_id if ":" in doc_id else f"{source}:{doc_id}")
            if refs:
                evidence_hint = f" 근거 참고({', '.join(refs)})."

        priority = "high" if gap.get("impact") == "high" else "medium"
        if f"{issue_id}:근거부족" in risk_flags:
            priority = "high"

        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{idx}",
                title="증빙 보강 및 사유별 반박 정리",
                detail=f"{gap.get('missing_evidence', '')} 자료를 보강하고 쟁점별 반박 논리를 정리.{evidence_hint}",
                priority=priority,
                linked_issue_id=issue_id,
            )
        )

    if band == "HIGH":
        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{len(actions)+1}",
                title="재심의 제출 완성도 점검",
                detail="제출 문서 완성도 및 첨부 누락 여부를 최종 점검 후 즉시 제출",
                priority="medium" if risk_level != "AGGRESSIVE" else "high",
                linked_issue_id="ISSUE-1",
            )
        )
    elif band == "MEDIUM":
        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{len(actions)+1}",
                title="쟁점별 반박서 + 추가 증빙 병행",
                detail="핵심 쟁점별 1페이지 반박서 작성과 보강 증빙 제출을 병행",
                priority="high" if risk_level in {"BALANCED", "AGGRESSIVE"} else "medium",
                linked_issue_id="ISSUE-1",
            )
        )
    else:
        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{len(actions)+1}",
                title="사전 질의/정리 후 재심의",
                detail="증빙 우선 보강 후 사전 질의로 쟁점을 정리하고 재심의 제출",
                priority="high",
                linked_issue_id="ISSUE-1",
            )
        )

    return {"recommended_actions": actions}


def _package_evidence_node(state: DataAnalysisState) -> DataAnalysisState:
    nodes = state.get("issue_tree", {}).get("nodes", [])
    issue_id = str(nodes[0].get("issue_id", "ISSUE-1")) if nodes else "ISSUE-1"

    strategy_context = state.get("strategy_context", {})
    top_evidence = strategy_context.get("top_evidence", [])

    if top_evidence:
        packaged: list[dict[str, Any]] = []
        for item in top_evidence:
            source_type = str(item.get("source_type", "CASELAW")).upper()
            if source_type == "DISPUTE":
                mapped_source = "dispute_case"
            elif source_type == "WEB":
                mapped_source = "web"
            else:
                mapped_source = "caselaw"
            provenance = {
                "source_type": mapped_source,
                "source_id": str(item.get("doc_id", "")),
                "title": str(item.get("title", "")),
                "snippet": str(item.get("snippet", "")),
                "retrieved_at": str(item.get("provenance", {}).get("retrieved_at", "")),
                "metadata": {
                    "url": item.get("url"),
                    "index_name": item.get("provenance", {}).get("index_name", ""),
                    "retrieval_method": item.get("provenance", {}).get("retrieval_method", ""),
                },
            }
            packaged.append(
                {
                    "evidence_id": str(item.get("doc_id", "")),
                    "issue_id": issue_id,
                    "evidence_title": str(item.get("title", "")),
                    "summary": str(item.get("snippet", "")),
                    "relevance_score": float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
                    "provenance": [provenance],
                }
            )
        return {"evidence_pack": packaged}

    evidence = [ranked_case_to_evidence_item(doc, issue_id) for doc in state.get("ranked_cases", [])]
    return {"evidence_pack": evidence}


def _priority_rank(priority: str) -> int:
    order = {"high": 0, "medium": 1, "low": 2}
    return order.get(str(priority).lower(), 1)


def _source_type_to_public(source_type: str) -> str:
    st = str(source_type).strip().lower()
    if st in {"caselaw", "case_law"}:
        return "CASELAW"
    if st in {"dispute_case", "dispute"}:
        return "DISPUTE"
    return "WEB"


def _ref_from_evidence(evidence_item: dict[str, Any]) -> str:
    provenance_list = evidence_item.get("provenance", [])
    if provenance_list and isinstance(provenance_list, list):
        first = provenance_list[0] if provenance_list else {}
        if isinstance(first, dict):
            source_type = _source_type_to_public(str(first.get("source_type", "caselaw")))
            source_id = str(first.get("source_id", "")).strip()
            if source_id:
                return f"{source_type}:{source_id}"

    evidence_id = str(evidence_item.get("evidence_id", "")).strip()
    if evidence_id:
        return f"CASELAW:{evidence_id}"
    return "CASELAW:UNKNOWN"


def _build_issue_evidence_map_from_pack(evidence_pack: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    issue_map: dict[str, list[dict[str, Any]]] = {}
    for item in evidence_pack:
        issue_id = str(item.get("issue_id", "ISSUE-1"))
        issue_map.setdefault(issue_id, []).append(item)
    for issue_id in issue_map:
        issue_map[issue_id] = issue_map[issue_id][:2]
    return issue_map


def _synthesize_analysis_node(state: DataAnalysisState) -> DataAnalysisState:
    issue_tree = state.get("issue_tree", {})
    issue_nodes = issue_tree.get("nodes", [])
    gap_analysis = state.get("gap_analysis", [])
    actions = state.get("recommended_actions", [])
    evidence_pack = state.get("evidence_pack", [])
    strategy_context = state.get("strategy_context", {})

    sorted_actions = sorted(actions, key=lambda a: _priority_rank(str(a.get("priority", "medium"))))
    high_impact_gaps = [g for g in gap_analysis if str(g.get("impact", "")).lower() == "high"]
    risk_flags = list(strategy_context.get("risk_flags", []))

    quality_flags: list[str] = []
    if not issue_nodes:
        quality_flags.append("missing_issue_tree")
    if not sorted_actions:
        quality_flags.append("missing_actions")
    if not evidence_pack:
        quality_flags.append("missing_evidence_pack")
    if not state.get("success_probability_public", {}).get("band"):
        quality_flags.append("missing_band")
    if len(evidence_pack) < max(1, len(issue_nodes)):
        quality_flags.append("low_evidence_coverage")

    key_issues = [
        {
            "issue_id": str(node.get("issue_id", "ISSUE-1")),
            "title": str(node.get("title", "")),
        }
        for node in issue_nodes[:3]
    ]

    summary = {
        "band": state.get("success_probability_public", {}).get("band", "LOW"),
        "key_issues": key_issues,
        "high_impact_gap_count": len(high_impact_gaps),
        "evidence_count": len(evidence_pack),
        "risk_flags": risk_flags,
        "priority_action_ids": [str(a.get("action_id", "")) for a in sorted_actions[:3]],
    }

    return {
        "recommended_actions": sorted_actions,
        "synthesized_summary": summary,
        "quality_flags": quality_flags,
    }


def _build_action_evidence_candidates(
    actions: list[RecommendedAction],
    evidence_pack: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_issue = _build_issue_evidence_map_from_pack(evidence_pack)
    candidates: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        action_id = str(action.get("action_id", ""))
        issue_id = str(action.get("linked_issue_id", "ISSUE-1"))
        issue_evidence = by_issue.get(issue_id, [])
        if not issue_evidence:
            issue_evidence = evidence_pack[:2]
        candidates[action_id] = issue_evidence[:2]
    return candidates


def _build_llm_messages(
    actions: list[RecommendedAction],
    action_evidence: dict[str, list[dict[str, Any]]],
    band: str,
    simplified: bool = False,
) -> tuple[SystemMessage, HumanMessage]:
    if simplified:
        system_prompt = (
            "You rewrite Korean action details for insurance re-review. "
            "Do not change action_id, priority, linked_issue_id. "
            "Return JSON only: {\"actions\":[{\"action_id\":\"...\",\"enriched_detail\":\"...\"}]}. "
            "Each enriched_detail must include at least one evidence reference like (CASELAW:123). "
            "No legal certainty claims."
        )
    else:
        system_prompt = (
            "You are a legal-insurance analysis writing assistant. "
            "Narration-only task: do not alter decisions. "
            "Never change band, priority, linked_issue_id. "
            "Ground strictly on provided evidence only. "
            "Output JSON only with schema "
            "{\"actions\":[{\"action_id\":\"ACTION-1\",\"enriched_detail\":\"...\"}]}. "
            "Each enriched_detail must include at least one provenance token in format "
            "(CASELAW:doc_id) or (DISPUTE:doc_id) or (WEB:doc_id). "
            "Use concise Korean imperative style and avoid definitive legal advice."
        )

    payload_actions = []
    for action in actions:
        action_id = str(action.get("action_id", ""))
        evidence_rows = []
        for ev in action_evidence.get(action_id, []):
            evidence_rows.append(
                {
                    "ref": _ref_from_evidence(ev),
                    "title": str(ev.get("evidence_title", "")),
                    "snippet": str(ev.get("summary", "")),
                }
            )
        payload_actions.append(
            {
                "action_id": action_id,
                "title": str(action.get("title", "")),
                "detail": str(action.get("detail", "")),
                "priority": str(action.get("priority", "")),
                "linked_issue_id": str(action.get("linked_issue_id", "")),
                "evidence_candidates": evidence_rows,
            }
        )

    human_payload = {
        "band": band,
        "actions": payload_actions,
        "instructions": {
            "keep_action_count": len(actions),
            "do_not_change_ids_or_priority": True,
            "korean_style": "간결한 실행 지시형",
        },
    }

    return SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(human_payload, ensure_ascii=False))


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        loaded = json.loads(stripped)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        snippet = stripped[start : end + 1]
        try:
            loaded = json.loads(snippet)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    return {}


def _has_reference_token(text: str) -> bool:
    return bool(re.search(r"\((CASELAW|DISPUTE|WEB):[^)]+\)", text))


def _enforce_reference(detail: str, evidence_candidates: list[dict[str, Any]]) -> str:
    text = str(detail).strip()
    if _has_reference_token(text):
        return text
    ref = ""
    if evidence_candidates:
        ref = _ref_from_evidence(evidence_candidates[0])
    if not ref:
        ref = "CASELAW:UNKNOWN"
    suffix = f" ({ref})"
    return (text + suffix).strip()


def _trigger_hitl_retry(state: DataAnalysisState, reason: str, attempt: int, latency_ms: int) -> bool:
    handler = state.get("analysis_options", {}).get("hitl_retry_handler")
    if callable(handler):
        try:
            return bool(
                handler(
                    {
                        "reason": reason,
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                    }
                )
            )
        except Exception:
            return True
    return True


def _stream_chat_completion(llm: ChatOpenAI, messages: list[Any], timeout_s: float) -> tuple[str, int]:
    started = time.monotonic()
    chunks: list[str] = []
    for chunk in llm.stream(messages):
        elapsed = time.monotonic() - started
        if elapsed > timeout_s:
            raise TimeoutError("llm_stream_timeout")
        piece = getattr(chunk, "content", "")
        if isinstance(piece, list):
            piece = "".join(str(p) for p in piece)
        if piece:
            chunks.append(str(piece))
    latency_ms = int((time.monotonic() - started) * 1000)
    return "".join(chunks), latency_ms


def _llm_action_enrichment_node(state: DataAnalysisState) -> DataAnalysisState:
    actions = list(state.get("recommended_actions", []))
    default_model = "solar-pro3"
    default_base_url = "https://api.upstage.ai/v1/solar"
    if not actions:
        return {
            "llm_enrichment_trace": {
                "model": default_model,
                "latency_ms": 0,
                "attempt": 0,
                "streaming_used": False,
                "hitl_triggered": False,
                "fallback_used": True,
            },
            "llm_raw_output": "",
        }

    if os.getenv("STEP3_DISABLE_LLM_ENRICHMENT", "0") == "1":
        return {
            "llm_enrichment_trace": {
                "model": default_model,
                "latency_ms": 0,
                "attempt": 0,
                "streaming_used": False,
                "hitl_triggered": False,
                "fallback_used": True,
            },
            "llm_raw_output": "",
        }

    options = state.get("analysis_options", {})
    enabled = bool(options.get("llm_enrichment_enabled", True))
    if not enabled:
        return {
            "llm_enrichment_trace": {
                "model": default_model,
                "latency_ms": 0,
                "attempt": 0,
                "streaming_used": False,
                "hitl_triggered": False,
                "fallback_used": True,
            },
            "llm_raw_output": "",
        }

    api_key = os.getenv("UPSTAGE_API_KEY", "")
    if not api_key:
        return {
            "llm_enrichment_trace": {
                "model": default_model,
                "latency_ms": 0,
                "attempt": 0,
                "streaming_used": False,
                "hitl_triggered": False,
                "fallback_used": True,
            },
            "llm_raw_output": "",
        }

    model_name = str(options.get("llm_model", default_model))
    base_url = str(options.get("llm_base_url", os.getenv("UPSTAGE_BASE_URL", default_base_url))).strip() or default_base_url
    timeout_s = float(options.get("llm_timeout_s", 10))

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": 0.2,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url

    llm = ChatOpenAI(**llm_kwargs)
    action_evidence = _build_action_evidence_candidates(actions, state.get("evidence_pack", []))
    band = str(state.get("success_probability_public", {}).get("band", "LOW"))

    hitl_triggered = False
    raw_text = ""
    latency_ms = 0
    fallback_used = False
    final_attempt = 0

    attempts = [False, False, True]
    for idx, simplified in enumerate(attempts, start=1):
        final_attempt = idx
        system_msg, human_msg = _build_llm_messages(actions, action_evidence, band, simplified=simplified)
        try:
            raw_text, latency_ms = _stream_chat_completion(llm, [system_msg, human_msg], timeout_s=timeout_s)
            parsed = _extract_json_object(raw_text)
            rows = parsed.get("actions", []) if isinstance(parsed, dict) else []
            if not isinstance(rows, list):
                rows = []
            enriched_by_id = {
                str(row.get("action_id", "")): str(row.get("enriched_detail", "")).strip()
                for row in rows
                if isinstance(row, dict)
            }

            if not enriched_by_id and idx == 1:
                continue
            if not enriched_by_id:
                fallback_used = True
                break

            new_actions: list[RecommendedAction] = []
            for action in actions:
                action_id = str(action.get("action_id", ""))
                candidate = enriched_by_id.get(action_id, str(action.get("detail", "")))
                candidate = _enforce_reference(candidate, action_evidence.get(action_id, []))
                patched = dict(action)
                patched["detail"] = candidate
                new_actions.append(RecommendedAction(**patched))

            return {
                "recommended_actions": new_actions,
                "llm_raw_output": raw_text,
                "llm_enrichment_trace": {
                    "model": model_name,
                    "latency_ms": latency_ms,
                    "attempt": idx,
                    "streaming_used": True,
                    "hitl_triggered": hitl_triggered,
                    "fallback_used": False,
                },
            }
        except TimeoutError:
            if not hitl_triggered:
                hitl_triggered = _trigger_hitl_retry(
                    state=state,
                    reason="timeout",
                    attempt=idx,
                    latency_ms=int(timeout_s * 1000),
                )
            continue
        except Exception:
            if idx == len(attempts):
                fallback_used = True
                break
            continue

    fallback_actions: list[RecommendedAction] = []
    for action in actions:
        action_id = str(action.get("action_id", ""))
        detail = _enforce_reference(str(action.get("detail", "")), action_evidence.get(action_id, []))
        patched = dict(action)
        patched["detail"] = detail
        fallback_actions.append(RecommendedAction(**patched))

    return {
        "recommended_actions": fallback_actions,
        "llm_raw_output": raw_text,
        "llm_enrichment_trace": {
            "model": model_name,
            "latency_ms": latency_ms,
            "attempt": final_attempt,
            "streaming_used": True,
            "hitl_triggered": hitl_triggered,
            "fallback_used": True,
        },
    }


def _finalize_node(state: DataAnalysisState) -> DataAnalysisState:
    result = AnalysisResult(
        issue_tree=state.get("issue_tree", IssueTree(root_title="보험금 부지급 재심의 쟁점", nodes=[])),
        gap_analysis=state.get("gap_analysis", []),
        recommended_actions=state.get("recommended_actions", []),
        success_probability=state.get(
            "success_probability_public",
            SuccessProbability(
                band="LOW",
                positive_drivers=[],
                negative_drivers=[],
                assumptions=["insufficient input"],
            ),
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
    graph.add_node("score_precedent", _score_precedent_node)
    graph.add_node("score_adjustment", _score_adjustment_node)
    graph.add_node("estimate_success_probability", _estimate_success_probability_node)
    graph.add_node("build_strategy_context", _build_strategy_context_node)
    graph.add_node("strategy", _strategy_node)
    graph.add_node("package_evidence", _package_evidence_node)
    graph.add_node("synthesize_analysis", _synthesize_analysis_node)
    graph.add_node("llm_action_enrichment", _llm_action_enrichment_node)
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "build_query")
    graph.add_edge("build_query", "retrieve")
    graph.add_edge("retrieve", "normalize_rank")
    graph.add_edge("normalize_rank", "issue_analysis")
    graph.add_edge("issue_analysis", "gap_analysis")
    graph.add_edge("gap_analysis", "score_precedent")
    graph.add_edge("score_precedent", "score_adjustment")
    graph.add_edge("score_adjustment", "estimate_success_probability")
    graph.add_edge("estimate_success_probability", "build_strategy_context")
    graph.add_edge("build_strategy_context", "strategy")
    graph.add_edge("strategy", "package_evidence")
    graph.add_edge("package_evidence", "synthesize_analysis")
    graph.add_edge("synthesize_analysis", "llm_action_enrichment")
    graph.add_edge("llm_action_enrichment", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_pipeline(
    structured_case: StructuredCase | dict[str, Any],
    rag_result: dict[str, Any] | None = None,
    analysis_options: dict[str, Any] | None = None,
) -> AnalysisResult:
    initial_state = DataAnalysisState(
        structured_case=normalize_structured_case(structured_case),
        rag_result=rag_result or {},
        analysis_options=analysis_options or {"risk_level": "BALANCED", "output_style": "USER_READABLE"},
    )
    app = build_graph()
    result_state: DataAnalysisState = app.invoke(initial_state)
    return result_state["analysis_result"]
