from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
import yaml

from agents.data_analysis_agent.adapters import ranked_case_to_evidence_item
from core.schemas.analysis import (
    AnalysisResult,
    GapAnalysisItem,
    IssueBriefItem,
    IssueNode,
    IssueTree,
    RecommendedAction,
    SuccessProbability,
    UserGuidance,
)
from core.schemas.case_context import StructuredCase, normalize_structured_case
from core.schemas.rag_contract import RAGRetrievalResult, RetrievalMode, ScoringTrace
from core.scoring.features import extract_features
from core.scoring.rubric import ScoreEstimate, score_success
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
    analysis_options: dict[str, Any]
    rag_result: dict[str, Any]
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

    rag_result_internal: RAGRetrievalResult
    precedent_rag_result: RAGRetrievalResult
    dispute_rag_result: RAGRetrievalResult
    domain_filter_trace: dict[str, Any]

    issue_tree: IssueTree
    gap_analysis: list[GapAnalysisItem]
    recommended_actions: list[RecommendedAction]

    features: dict[str, Any]
    precedent_probability: ScoreEstimate
    case_adjustment: int
    case_adjustment_source: str
    adjustment_notes: list[str]

    internal_scoring: dict[str, Any]
    scoring_trace: ScoringTrace
    success_probability_public: SuccessProbability

    strategy_context: dict[str, Any]
    synthesized_summary: dict[str, Any]
    quality_flags: list[str]
    llm_enrichment_trace: dict[str, Any]
    llm_raw_output: str
    user_guidance: UserGuidance

    evidence_pack: list[dict[str, Any]]
    analysis_result: AnalysisResult


def _empty_rag_result(query_id: str = "Q-unknown", query: str = "") -> RAGRetrievalResult:
    return RAGRetrievalResult(
        query_id=query_id,
        query=query,
        retrieval_mode="plain",
        query_variants=["plain"],
        filters={},
        items=[],
        stats={"candidate_count": 0, "returned_count": 0, "latency_ms": 0},
    )


def _resolve_retrieval_mode(options: dict[str, Any] | None) -> RetrievalMode:
    raw = str((options or {}).get("retrieval_mode", os.getenv("RAG_RETRIEVAL_MODE", "plain"))).strip().lower()
    if raw in {"plain", "hyde", "reverse_hyde", "hybrid_hyde"}:
        return cast(RetrievalMode, raw)
    return "plain"


def _normalize_external_rag_result(rag_result: dict[str, Any] | None, retrieval_mode: RetrievalMode) -> RAGRetrievalResult:
    if not rag_result or not isinstance(rag_result, dict):
        return _empty_rag_result()

    items: list[dict[str, Any]] = []
    for raw in rag_result.get("items", []):
        if not isinstance(raw, dict):
            continue
        source_type = str(raw.get("source_type", "CASELAW")).upper()
        if source_type not in {"CASELAW", "DISPUTE", "WEB"}:
            source_type = "CASELAW"
        items.append(
            {
                "source_type": source_type,
                "doc_id": str(raw.get("doc_id", "")),
                "chunk_id": str(raw.get("chunk_id", "")),
                "title": str(raw.get("title", "")),
                "snippet": str(raw.get("snippet", "")),
                "score": float(raw.get("score", 0.0) or 0.0),
                "rerank_score": float(raw.get("rerank_score", raw.get("score", 0.0)) or 0.0),
                "url": raw.get("url"),
                "published_at": raw.get("published_at"),
                "provenance": dict(raw.get("provenance", {})),
            }
        )

    return RAGRetrievalResult(
        query_id=str(rag_result.get("query_id", "Q-external")),
        query=str(rag_result.get("query", "")),
        retrieval_mode=cast(RetrievalMode, str(rag_result.get("retrieval_mode", retrieval_mode))),
        query_variants=list(rag_result.get("query_variants", ["external"])),
        filters=dict(rag_result.get("filters", {})),
        items=items,
        stats={
            "candidate_count": int(rag_result.get("stats", {}).get("candidate_count", len(items))),
            "returned_count": int(rag_result.get("stats", {}).get("returned_count", len(items))),
            "latency_ms": int(rag_result.get("stats", {}).get("latency_ms", 0)),
        },
    )


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


def _build_vector_index_store() -> Any:
    backend = os.getenv("RAG_VECTOR_BACKEND", "inmemory").strip().lower()
    index_name = os.getenv("RAG_INDEX_NAME", "step3_rag_index").strip() or "step3_rag_index"
    if backend == "chroma":
        persist_dir = os.getenv("RAG_CHROMA_DIR", ".chroma_db").strip() or ".chroma_db"
        return ChromaVectorIndexStore(index_name=index_name, persist_directory=persist_dir)
    return InMemoryVectorIndexStore(index_name=index_name)


def _strip_known_source_prefix(doc_id: str) -> str:
    sid = str(doc_id).strip()
    for prefix in ("CASELAW:", "DISPUTE:", "WEB:"):
        if sid.startswith(prefix):
            return sid[len(prefix) :]
    return sid


def _load_domain_filter_rules() -> dict[str, Any]:
    defaults = {
        "allow_keywords": [
            "실손",
            "의료",
            "입원",
            "보험금",
            "약관",
            "청구",
            "부지급",
            "비급여",
            "급여",
        ],
        "deny_keywords": [
            "자동차",
            "차량",
            "운전자",
            "대물",
            "대인",
            "교통사고",
            "자동차시세",
        ],
        "min_evidence_count": 5,
    }

    cfg_path = Path("data/data_analysis_data/rubrics/domain_filter.yml")
    if not cfg_path.exists():
        return defaults

    try:
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return defaults
    if not isinstance(loaded, dict):
        return defaults

    allow = [str(x).strip().lower() for x in loaded.get("allow_keywords", []) if str(x).strip()]
    deny = [str(x).strip().lower() for x in loaded.get("deny_keywords", []) if str(x).strip()]
    min_count = int(loaded.get("min_evidence_count", defaults["min_evidence_count"]) or defaults["min_evidence_count"])

    return {
        "allow_keywords": allow or defaults["allow_keywords"],
        "deny_keywords": deny or defaults["deny_keywords"],
        "min_evidence_count": max(1, min_count),
    }


def _collect_case_domain_keywords(structured_case: StructuredCase) -> list[str]:
    tokens: list[str] = []
    for reason in structured_case.get("denial_reasons", []):
        tokens.extend(str(reason).lower().split())
    for clause in structured_case.get("policy_clauses", []):
        tokens.extend(str(clause).lower().split())
    for ev in structured_case.get("evidence_summary", []):
        if not isinstance(ev, dict):
            continue
        tokens.extend(str(ev.get("title", "")).lower().split())
        tokens.extend(str(ev.get("summary", "")).lower().split())
    compacted = [tok for tok in tokens if tok and len(tok) >= 2]
    # keep deterministic order while removing duplicates
    return list(dict.fromkeys(compacted))


def _sort_rag_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = list(items)
    copied.sort(key=lambda x: float(x.get("rerank_score", x.get("score", 0.0)) or 0.0), reverse=True)
    return copied


def _apply_domain_filter(
    items: list[dict[str, Any]],
    structured_case: StructuredCase,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not items:
        return [], {"mode": "empty", "input_count": 0, "output_count": 0}

    rules = _load_domain_filter_rules()
    allow_keywords = set(str(k).strip().lower() for k in rules.get("allow_keywords", []))
    deny_keywords = set(str(k).strip().lower() for k in rules.get("deny_keywords", []))
    min_count = int(rules.get("min_evidence_count", 5))
    case_keywords = set(_collect_case_domain_keywords(structured_case))
    allow_keywords |= case_keywords

    def _is_target_source(row: dict[str, Any]) -> bool:
        return str(row.get("source_type", "")).upper() in {"CASELAW", "DISPUTE"}

    def _text_blob(row: dict[str, Any]) -> str:
        return " ".join(
            [
                str(row.get("title", "")),
                str(row.get("snippet", "")),
                str(row.get("doc_id", "")),
            ]
        ).lower()

    sorted_items = _sort_rag_items(items)
    non_target_rows: list[dict[str, Any]] = []
    strict_target_rows: list[dict[str, Any]] = []
    relaxed_target_rows: list[dict[str, Any]] = []
    soft_deny_rows: list[dict[str, Any]] = []
    hard_deny_rows: list[dict[str, Any]] = []

    for row in sorted_items:
        if not _is_target_source(row):
            non_target_rows.append(row)
            continue

        blob = _text_blob(row)
        has_allow = any(k and k in blob for k in allow_keywords)
        has_deny = any(k and k in blob for k in deny_keywords)
        if has_allow and not has_deny:
            strict_target_rows.append(row)
        elif not has_deny:
            relaxed_target_rows.append(row)
        elif has_allow and has_deny:
            soft_deny_rows.append(row)
        else:
            hard_deny_rows.append(row)

    strict_selected = list(non_target_rows) + list(strict_target_rows)
    relaxed_selected = list(non_target_rows) + list(strict_target_rows) + list(relaxed_target_rows)

    removed_doc_ids = [str(x.get("doc_id", "")) for x in soft_deny_rows + hard_deny_rows if str(x.get("doc_id", ""))]
    strict_count = len(strict_selected)
    relaxed_count = len(relaxed_selected)

    if strict_count >= min_count:
        return strict_selected, {
            "mode": "strict",
            "input_count": len(items),
            "output_count": strict_count,
            "strict_count": strict_count,
            "relaxed_count": relaxed_count,
            "soft_deny_included_count": 0,
            "soft_deny_doc_ids": [],
            "removed_doc_ids": removed_doc_ids[:10],
        }

    if relaxed_count >= min_count:
        return relaxed_selected, {
            "mode": "relaxed",
            "input_count": len(items),
            "output_count": relaxed_count,
            "strict_count": strict_count,
            "relaxed_count": relaxed_count,
            "soft_deny_included_count": 0,
            "soft_deny_doc_ids": [],
            "removed_doc_ids": [str(x.get("doc_id", "")) for x in hard_deny_rows if str(x.get("doc_id", ""))][:10],
        }

    def _row_key(row: dict[str, Any]) -> tuple[str, str]:
        return (str(row.get("doc_id", "")), str(row.get("chunk_id", "")))

    selected = list(relaxed_selected)
    seen_keys = {_row_key(row) for row in selected}

    max_soft_deny = max(1, min_count // 5)
    soft_deny_doc_ids: list[str] = []
    for row in soft_deny_rows:
        if len(selected) >= min_count or len(soft_deny_doc_ids) >= max_soft_deny:
            break
        key = _row_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(row)
        doc_id = str(row.get("doc_id", ""))
        if doc_id:
            soft_deny_doc_ids.append(doc_id)

    if len(selected) >= min_count:
        return selected, {
            "mode": "fallback_soft_deny",
            "input_count": len(items),
            "output_count": len(selected),
            "strict_count": strict_count,
            "relaxed_count": relaxed_count,
            "soft_deny_included_count": len(soft_deny_doc_ids),
            "soft_deny_doc_ids": soft_deny_doc_ids[:10],
            "removed_doc_ids": [str(x.get("doc_id", "")) for x in hard_deny_rows if str(x.get("doc_id", ""))][:10],
        }

    # Hard fallback: keep availability by filling remaining slots with high-scoring rows.
    fallback_target = max(1, min_count)
    for row in (soft_deny_rows + hard_deny_rows):
        if len(selected) >= fallback_target:
            break
        key = _row_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(row)

    return selected, {
        "mode": "fallback_hard",
        "input_count": len(items),
        "output_count": len(selected),
        "strict_count": strict_count,
        "relaxed_count": relaxed_count,
        "soft_deny_included_count": len(soft_deny_doc_ids),
        "soft_deny_doc_ids": soft_deny_doc_ids[:10],
        "removed_doc_ids": [],
    }


def _normalize_rank_node(state: DataAnalysisState) -> DataAnalysisState:
    normalized_precedent = normalize_cases(state.get("raw_precedent_cases", []))
    normalized_dispute = normalize_cases(state.get("raw_dispute_cases", []))

    precedent_ranked = rank_cases(normalized_precedent, state["structured_case"], top_k=6)
    dispute_ranked = rank_cases(normalized_dispute, state["structured_case"], top_k=6)

    precedent_query = " | ".join([q.get("query", "") for q in state.get("precedent_queries", [])]) or "보험금 부지급 판례"
    dispute_query = " | ".join([q.get("query", "") for q in state.get("dispute_queries", [])]) or "보험 분쟁 사례"

    retrieval_mode = state.get("retrieval_mode", "plain")
    precedent_rag = _build_rag_result_from_ranked(
        ranked=precedent_ranked,
        query=precedent_query,
        query_id=f"Q-{state['structured_case'].get('case_id', 'unknown')}-precedent",
        retrieval_mode=retrieval_mode,
        source_filter="CASELAW",
        structured_case=state.get("structured_case", {}),
    )
    dispute_rag = _build_rag_result_from_ranked(
        ranked=dispute_ranked,
        query=dispute_query,
        query_id=f"Q-{state['structured_case'].get('case_id', 'unknown')}-dispute",
        retrieval_mode=retrieval_mode,
        source_filter="DISPUTE",
        structured_case=state.get("structured_case", {}),
    )
    merged_rag = _merge_rag_results(precedent_rag, dispute_rag, f"Q-{state['structured_case'].get('case_id', 'unknown')}")

    external_rag = _normalize_external_rag_result(state.get("rag_result"), retrieval_mode)
    rag_internal = external_rag if external_rag.get("items") else merged_rag
    filtered_items, domain_trace = _apply_domain_filter(
        list(rag_internal.get("items", [])),
        state.get("structured_case", StructuredCase()),
    )
    filtered_rag = dict(rag_internal)
    filtered_rag["items"] = filtered_items
    filtered_rag["stats"] = {
        "candidate_count": int(rag_internal.get("stats", {}).get("candidate_count", len(filtered_items))),
        "returned_count": len(filtered_items),
        "latency_ms": int(rag_internal.get("stats", {}).get("latency_ms", 0)),
    }

    return {
        "normalized_precedent_cases": normalized_precedent,
        "normalized_dispute_cases": normalized_dispute,
        "normalized_cases": normalized_precedent + normalized_dispute,
        "precedent_ranked_cases": precedent_ranked,
        "dispute_ranked_cases": dispute_ranked,
        "ranked_cases": precedent_ranked + dispute_ranked,
        "precedent_rag_result": precedent_rag,
        "dispute_rag_result": dispute_rag,
        "rag_result_internal": RAGRetrievalResult(**filtered_rag),
        "domain_filter_trace": domain_trace,
    }


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


def _select_trace_rag(state: DataAnalysisState) -> RAGRetrievalResult:
    rag_internal = state.get("rag_result_internal", _empty_rag_result("Q-internal"))
    if rag_internal.get("items"):
        case_only = [
            item
            for item in rag_internal.get("items", [])
            if str(item.get("source_type", "")).upper() == "CASELAW"
        ]
        if case_only:
            copied = dict(rag_internal)
            copied["items"] = case_only
            copied["stats"] = {
                "candidate_count": int(len(case_only)),
                "returned_count": int(len(case_only)),
                "latency_ms": int(rag_internal.get("stats", {}).get("latency_ms", 0)),
            }
            return RAGRetrievalResult(**copied)
    return state.get("precedent_rag_result", _empty_rag_result("Q-precedent"))


def _score_adjustment_node(state: DataAnalysisState) -> DataAnalysisState:
    precedent = state.get(
        "precedent_probability",
        ScoreEstimate(score=0, band="LOW", positive_drivers=[], negative_drivers=[], assumptions=[]),
    )
    precedent_score = int(precedent.get("score", 0) or 0)

    dispute_rag = state.get("dispute_rag_result", _empty_rag_result("Q-dispute"))
    valid_doc_ids = {str(item.get("doc_id", "")) for item in dispute_rag.get("items", []) if item.get("doc_id")}

    decision = compute_adjustment_with_fallback(
        structured_case=state["structured_case"],
        dispute_ranked_cases=state.get("dispute_ranked_cases", []),
        precedent_score=precedent_score,
        features=state.get("features", {}),
        valid_doc_ids=valid_doc_ids,
    )

    trace_rag = _select_trace_rag(state)
    scoring_trace = compute_comparative_scoring(
        ComparativeScoringInput(
            rag_result=trace_rag,
            case_adjustment=decision.case_adjustment,
            case_adjustment_source=decision.source,
            rationale=decision.rationale,
            cited_case_ids=decision.cited_case_ids,
        )
    )

    trace_total_score = int(scoring_trace.get("total_score", max(0, min(100, precedent_score + decision.case_adjustment))))

    return {
        "case_adjustment": decision.case_adjustment,
        "case_adjustment_source": decision.source,
        "adjustment_notes": decision.notes,
        "internal_scoring": {
            "precedent_score": precedent_score,
            "case_adjustment": decision.case_adjustment,
            # SSOT: comparative scoring trace total_score
            "total_score": trace_total_score,
        },
        "scoring_trace": ScoringTrace(**scoring_trace),
    }


def _merge_unique(primary: list[str], secondary: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for text in primary + secondary:
        key = str(text).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(key)
    return merged


def _estimate_success_probability_node(state: DataAnalysisState) -> DataAnalysisState:
    precedent = state.get(
        "precedent_probability",
        ScoreEstimate(score=0, band="LOW", positive_drivers=[], negative_drivers=[], assumptions=[]),
    )
    internal_scoring = state.get("internal_scoring", {})

    precedent_score = int(internal_scoring.get("precedent_score", precedent.get("score", 0) or 0))
    case_adjustment = int(internal_scoring.get("case_adjustment", state.get("case_adjustment", 0) or 0))
    scoring_trace = state.get("scoring_trace", {})
    trace_total = scoring_trace.get("total_score")
    total_score = int(trace_total if trace_total is not None else internal_scoring.get("total_score", max(0, min(100, precedent_score + case_adjustment))))
    band = _score_to_band(total_score)

    rubric_positive = list(precedent.get("positive_drivers", []))
    rubric_negative = list(precedent.get("negative_drivers", []))

    comparative_positive: list[str] = []
    comparative_negative: list[str] = []
    if total_score >= 70:
        comparative_positive.append("RAG 비교 점수 높음")
    elif total_score <= 39:
        comparative_negative.append("RAG 비교 점수 낮음")

    if case_adjustment > 0:
        comparative_positive.append(f"사례 기반 보정 +{case_adjustment}점")
    elif case_adjustment < 0:
        comparative_negative.append(f"사례 기반 보정 {case_adjustment}점")

    positive_drivers = _merge_unique(comparative_positive, rubric_positive)
    negative_drivers = _merge_unique(comparative_negative, rubric_negative)

    assumptions = list(precedent.get("assumptions", [])) + list(state.get("adjustment_notes", []))
    assumptions.append(f"case_adjustment_source={state.get('case_adjustment_source', 'zero')}")

    return {
        "success_probability_public": SuccessProbability(
            band=cast(Any, band),
            positive_drivers=positive_drivers,
            negative_drivers=negative_drivers,
            assumptions=assumptions,
        ),
        "scoring_trace": {
            **scoring_trace,
            "precedent_score": precedent_score,
            "case_adjustment": case_adjustment,
            "total_score": total_score,
        },
    }


def _select_strategy_evidence(rag_items: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    if not rag_items:
        return []

    source_weight = {"CASELAW": 1.0, "DISPUTE": 0.85, "WEB": 0.5}
    ranked = []
    for item in rag_items:
        rerank = float(item.get("rerank_score", 0.0) or 0.0)
        score = float(item.get("score", 0.0) or 0.0)
        source_type = str(item.get("source_type", "CASELAW")).upper()
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
        terms = " ".join(
            [
                str(issue.get("title", "")),
                str(issue.get("description", "")),
                " ".join(issue.get("related_denial_reasons", [])),
                " ".join(issue.get("related_policy_clauses", [])),
            ]
        )

        matched: list[dict[str, Any]] = []
        for item in evidence:
            blob = " ".join([str(item.get("title", "")), str(item.get("snippet", "")), str(item.get("doc_id", ""))])
            if any(term and term in blob for term in terms.split()):
                matched.append(item)

        if not matched:
            matched = evidence[:2]
        mapping[issue_id] = matched[:2]

    return mapping


def _build_strategy_context_node(state: DataAnalysisState) -> DataAnalysisState:
    issue_nodes = state.get("issue_tree", {}).get("nodes", [])

    rag_items = list(state.get("rag_result_internal", {}).get("items", []))
    if rag_items:
        top_evidence = _select_strategy_evidence(rag_items, top_n=5)
    else:
        fallback_items: list[dict[str, Any]] = []
        for row in state.get("ranked_cases", []):
            source = str(row.get("source_type", row.get("source", "caselaw"))).lower()
            normalized_source = "DISPUTE" if source in {"dispute", "dispute_case"} else "CASELAW"
            fallback_items.append(
                {
                    "source_type": normalized_source,
                    "doc_id": row.get("doc_id", ""),
                    "chunk_id": "",
                    "title": row.get("title", ""),
                    "snippet": row.get("summary", ""),
                    "score": float(row.get("relevance_score", 0.0) or 0.0),
                    "rerank_score": float(row.get("relevance_score", 0.0) or 0.0),
                    "url": row.get("url"),
                    "published_at": row.get("published_at"),
                    "provenance": {},
                }
            )
        top_evidence = _select_strategy_evidence(fallback_items, top_n=5)

    issue_evidence_map = _group_evidence_by_issue(issue_nodes, top_evidence)

    coverage: dict[str, int] = {}
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
            human_refs = []
            for ev in evidence:
                source = str(ev.get("source_type", "CASELAW")).upper()
                doc_id = str(ev.get("doc_id", "")).strip()
                title = str(ev.get("title", "")).strip() or "제목 정보 없음"
                if not doc_id:
                    continue
                ref_token = _canonical_doc_ref(source, doc_id)
                human_refs.append(f"{_source_label_ko(source)}({ref_token}, {title})")
            if human_refs:
                evidence_hint = f" 근거: {', '.join(human_refs[:2])}."

        priority = "high" if gap.get("impact") == "high" else "medium"
        if f"{issue_id}:근거부족" in risk_flags:
            priority = "high"

        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{idx}",
                title="증빙 보강 및 사유별 반박 정리",
                detail=(
                    f"먼저 {gap.get('missing_evidence', '')}를 준비해 제출 자료를 보강하세요. "
                    f"이 자료가 있으면 보험사 거절 사유에 대한 반박 근거를 더 명확히 제시할 수 있습니다.{evidence_hint}"
                ),
                priority=cast(Any, priority),
                linked_issue_id=issue_id,
            )
        )

    if band == "HIGH":
        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{len(actions)+1}",
                title="재심의 제출 완성도 점검",
                detail="제출 전 체크리스트로 누락 문서와 금액/기간 오기를 점검한 뒤 바로 접수하세요. 제출 지연을 줄이면 절차상 불이익 가능성을 낮출 수 있습니다.",
                priority=cast(Any, "medium" if risk_level != "AGGRESSIVE" else "high"),
                linked_issue_id="ISSUE-1",
            )
        )
    elif band == "MEDIUM":
        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{len(actions)+1}",
                title="쟁점별 반박서 + 추가 증빙 병행",
                detail="핵심 쟁점마다 1페이지 반박서를 작성하고, 해당 쟁점을 뒷받침하는 증빙을 함께 제출하세요. 설명문과 증빙을 짝지어 내면 심사자가 판단하기 쉬워집니다.",
                priority=cast(Any, "high" if risk_level in {"BALANCED", "AGGRESSIVE"} else "medium"),
                linked_issue_id="ISSUE-1",
            )
        )
    else:
        actions.append(
            RecommendedAction(
                action_id=f"ACTION-{len(actions)+1}",
                title="사전 질의/정리 후 재심의",
                detail="즉시 제출보다 먼저 부족한 증빙을 채우고, 사전 질의로 핵심 쟁점을 정리한 뒤 재심의를 진행하세요. 준비도를 높이면 불필요한 반려를 줄일 수 있습니다.",
                priority=cast(Any, "high"),
                linked_issue_id="ISSUE-1",
            )
        )

    return {"recommended_actions": actions}


def _package_evidence_node(state: DataAnalysisState) -> DataAnalysisState:
    strategy_context = state.get("strategy_context", {})
    top_evidence = strategy_context.get("top_evidence", [])

    if top_evidence:
        issue_nodes = state.get("issue_tree", {}).get("nodes", [])
        default_issue_id = str(issue_nodes[0].get("issue_id", "ISSUE-1")) if issue_nodes else "ISSUE-1"

        issue_evidence_map = strategy_context.get("issue_evidence_map", {})
        issue_by_doc: dict[str, str] = {}
        for issue_id, rows in issue_evidence_map.items():
            for row in rows:
                doc_id = str(row.get("doc_id", ""))
                if doc_id and doc_id not in issue_by_doc:
                    issue_by_doc[doc_id] = str(issue_id)

        packaged: list[dict[str, Any]] = []
        for item in top_evidence:
            source_type = str(item.get("source_type", "CASELAW")).upper()
            mapped_source = "dispute_case" if source_type == "DISPUTE" else ("web" if source_type == "WEB" else "caselaw")
            doc_id = str(item.get("doc_id", ""))
            provenance = {
                "source_type": mapped_source,
                "source_id": doc_id,
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
                    "evidence_id": doc_id,
                    "issue_id": issue_by_doc.get(doc_id, default_issue_id),
                    "evidence_title": str(item.get("title", "")),
                    "summary": str(item.get("snippet", "")),
                    "relevance_score": float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
                    "provenance": [provenance],
                }
            )
        return {"evidence_pack": packaged}

    issue_nodes = state.get("issue_tree", {}).get("nodes", [])
    issue_id = str(issue_nodes[0].get("issue_id", "ISSUE-1")) if issue_nodes else "ISSUE-1"
    evidence = [ranked_case_to_evidence_item(doc, issue_id) for doc in state.get("ranked_cases", [])]
    return {"evidence_pack": evidence}


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(priority).lower(), 1)


def _normalize_action_category(action: RecommendedAction) -> str:
    text = " ".join([str(action.get("title", "")), str(action.get("detail", ""))]).lower()
    if "반박" in text or "쟁점별" in text:
        return "반박서"
    if "증빙" in text or "소견서" in text or "확인서" in text or "영수증" in text:
        return "증빙보강"
    return "제출체크"


def _extract_doc_hint_from_detail(detail: str) -> str:
    text = str(detail or "")
    match = re.search(r"먼저\s+(.+?)를 준비", text)
    if match:
        return _trim_for_user(match.group(1))
    return ""


def _canonical_title_by_category(category: str) -> str:
    if category == "반박서":
        return "쟁점별 반박서 작성"
    if category == "증빙보강":
        return "증빙 보강 패키지 정리"
    return "제출 전 최종 체크"


def _default_action_detail_by_category(category: str) -> str:
    if category == "반박서":
        return "핵심 쟁점별 반박 포인트를 1페이지로 정리하고 근거 문서를 함께 매핑하세요."
    if category == "증빙보강":
        return "핵심 증빙 원본/사본을 점검하고 누락 서류를 먼저 보강해 제출 패키지를 완성하세요."
    return "접수 전 체크리스트로 제출 문서와 쟁점 대응표를 최종 점검하세요."


def _make_default_action(category: str, linked_issue_id: str) -> RecommendedAction:
    priority = "high" if category in {"반박서", "증빙보강"} else "medium"
    return RecommendedAction(
        action_id="",
        title=_canonical_title_by_category(category),
        detail=_default_action_detail_by_category(category),
        priority=cast(Any, priority),
        linked_issue_id=linked_issue_id,
    )


def _dedupe_actions(
    actions: list[RecommendedAction],
    *,
    default_issue_id: str = "ISSUE-1",
    min_actions: int = 2,
) -> list[RecommendedAction]:
    if not actions:
        seed = [
            _make_default_action("반박서", default_issue_id),
            _make_default_action("증빙보강", default_issue_id),
        ]
        min_actions = max(1, min_actions)
        actions = seed[:min_actions]

    grouped: dict[str, list[RecommendedAction]] = {"반박서": [], "증빙보강": [], "제출체크": []}
    for action in actions:
        grouped[_normalize_action_category(action)].append(action)

    deduped: list[RecommendedAction] = []
    for category in ["반박서", "증빙보강", "제출체크"]:
        rows = grouped.get(category, [])
        if not rows:
            continue
        rows = sorted(rows, key=lambda x: _priority_rank(str(x.get("priority", "medium"))))
        base = dict(rows[0])
        base["title"] = _canonical_title_by_category(category)
        base["linked_issue_id"] = str(base.get("linked_issue_id", "ISSUE-1"))
        base["priority"] = cast(Any, "high" if any(str(r.get("priority", "medium")) == "high" for r in rows) else "medium")

        details = [str(r.get("detail", "")).strip() for r in rows if str(r.get("detail", "")).strip()]
        merged_detail = details[0] if details else "핵심 쟁점에 맞춰 제출 자료를 정리하세요."
        doc_hints = list(dict.fromkeys([_extract_doc_hint_from_detail(x) for x in details if _extract_doc_hint_from_detail(x)]))
        if doc_hints:
            merged_detail = f"{merged_detail} 우선 준비 자료: {', '.join(doc_hints[:3])}."
        base["detail"] = _trim_for_user(merged_detail)
        deduped.append(RecommendedAction(**base))

    if not deduped:
        deduped = [_make_default_action("제출체크", default_issue_id)]

    min_actions = max(1, int(min_actions or 1))
    existing_categories = {_normalize_action_category(action) for action in deduped}

    preferred_order = ["반박서", "증빙보강", "제출체크"]
    while len(deduped) < min_actions:
        target_category = ""
        for category in preferred_order:
            if category not in existing_categories:
                target_category = category
                break
        if not target_category:
            target_category = "증빙보강"
        deduped.append(_make_default_action(target_category, default_issue_id))
        existing_categories.add(target_category)

    normalized: list[RecommendedAction] = []
    for idx, action in enumerate(deduped, start=1):
        patched = dict(action)
        patched["action_id"] = f"ACTION-{idx}"
        patched["linked_issue_id"] = str(patched.get("linked_issue_id", default_issue_id) or default_issue_id)
        normalized.append(RecommendedAction(**patched))
    return normalized


def _synthesize_analysis_node(state: DataAnalysisState) -> DataAnalysisState:
    issue_nodes = state.get("issue_tree", {}).get("nodes", [])
    gap_analysis = state.get("gap_analysis", [])
    default_issue_id = str(issue_nodes[0].get("issue_id", "ISSUE-1")) if issue_nodes else "ISSUE-1"
    actions = _dedupe_actions(
        state.get("recommended_actions", []),
        default_issue_id=default_issue_id,
        min_actions=2,
    )
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

    summary = {
        "band": state.get("success_probability_public", {}).get("band", "LOW"),
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


def _source_type_to_public(source_type: str) -> str:
    st = str(source_type).strip().lower()
    if st in {"caselaw", "case_law"}:
        return "CASELAW"
    if st in {"dispute_case", "dispute"}:
        return "DISPUTE"
    return "WEB"


def _source_label_ko(source_type: str) -> str:
    st = str(source_type).upper()
    if st == "CASELAW":
        return "판례"
    if st == "DISPUTE":
        return "분쟁사례"
    return "웹자료"


def _canonical_doc_ref(source_type: str, doc_id: str) -> str:
    canonical_source = str(source_type).upper()
    sid = str(doc_id).strip()
    if not sid:
        return f"{canonical_source}:UNKNOWN"

    parts = sid.split(":")
    while len(parts) > 1 and parts[0].upper() in {"CASELAW", "DISPUTE", "WEB"} and parts[1].upper() == parts[0].upper():
        parts = parts[1:]
    sid = ":".join(parts)

    if ":" in sid and sid.split(":", 1)[0].upper() in {"CASELAW", "DISPUTE", "WEB"}:
        return sid
    return f"{canonical_source}:{sid}"


def _ref_from_evidence(evidence_item: dict[str, Any]) -> str:
    provenance_list = evidence_item.get("provenance", [])
    if provenance_list and isinstance(provenance_list, list):
        first = provenance_list[0] if provenance_list else {}
        if isinstance(first, dict):
            source_type = _source_type_to_public(str(first.get("source_type", "caselaw")))
            source_id = str(first.get("source_id", "")).strip()
            if source_id:
                return _canonical_doc_ref(source_type, source_id)

    evidence_id = str(evidence_item.get("evidence_id", "")).strip()
    if evidence_id:
        return _canonical_doc_ref("CASELAW", evidence_id)
    return "CASELAW:UNKNOWN"


def _title_from_evidence(evidence_item: dict[str, Any]) -> str:
    title = str(evidence_item.get("evidence_title", "")).strip()
    if title:
        return title
    provenance_list = evidence_item.get("provenance", [])
    if provenance_list and isinstance(provenance_list, list):
        first = provenance_list[0] if provenance_list else {}
        if isinstance(first, dict):
            ptitle = str(first.get("title", "")).strip()
            if ptitle:
                return ptitle
    return "제목 정보 없음"


def _humanized_ref_from_evidence(evidence_item: dict[str, Any]) -> str:
    ref = _ref_from_evidence(evidence_item)
    source_type = ref.split(":", 1)[0] if ":" in ref else "CASELAW"
    label = _source_label_ko(source_type)
    title = _title_from_evidence(evidence_item)
    return f"근거: {label}({ref}, {title})"


def _build_action_evidence_candidates(
    actions: list[RecommendedAction],
    evidence_pack: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_issue: dict[str, list[dict[str, Any]]] = {}
    for item in evidence_pack:
        issue_id = str(item.get("issue_id", "ISSUE-1"))
        by_issue.setdefault(issue_id, []).append(item)
    for issue_id in by_issue:
        by_issue[issue_id] = by_issue[issue_id][:2]

    candidates: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        action_id = str(action.get("action_id", ""))
        issue_id = str(action.get("linked_issue_id", "ISSUE-1"))
        issue_evidence = by_issue.get(issue_id, [])
        if not issue_evidence:
            issue_evidence = evidence_pack[:2]
        candidates[action_id] = issue_evidence[:2]
    return candidates


def _has_reference_token(text: str) -> bool:
    return bool(re.search(r"(CASELAW|DISPUTE|WEB):[^),\\s]+", text))


def _enforce_reference(detail: str, evidence_candidates: list[dict[str, Any]]) -> str:
    text = str(detail).strip()
    if _has_reference_token(text) and "근거:" in text:
        return text
    if evidence_candidates:
        guide = _humanized_ref_from_evidence(evidence_candidates[0])
    else:
        guide = "근거: 판례(CASELAW:UNKNOWN, 제목 정보 없음)"
    connector = " " if text else ""
    return (text + connector + guide).strip()


def _ensure_reference_on_actions(
    actions: list[RecommendedAction],
    evidence_candidates: dict[str, list[dict[str, Any]]],
) -> list[RecommendedAction]:
    patched_actions: list[RecommendedAction] = []
    for action in actions:
        action_id = str(action.get("action_id", ""))
        patched = dict(action)
        patched["detail"] = _enforce_reference(str(action.get("detail", "")), evidence_candidates.get(action_id, []))
        patched_actions.append(RecommendedAction(**patched))
    return patched_actions


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
            "Each enriched_detail must include at least one evidence reference in Korean humanized format "
            "like '근거: 판례(CASELAW:123, 제목)'. "
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
            "Each enriched_detail must include at least one provenance in Korean humanized format: "
            "'근거: 판례(CASELAW:doc_id, 제목)' or '근거: 분쟁사례(DISPUTE:doc_id, 제목)' or "
            "'근거: 웹자료(WEB:doc_id, 제목)'. "
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


def _trigger_hitl_retry(state: DataAnalysisState, reason: str, attempt: int, latency_ms: int) -> bool:
    handler = state.get("analysis_options", {}).get("hitl_retry_handler")
    if callable(handler):
        try:
            return bool(handler({"reason": reason, "attempt": attempt, "latency_ms": latency_ms}))
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
    action_evidence = _build_action_evidence_candidates(actions, state.get("evidence_pack", []))

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

    options = state.get("analysis_options", {})
    disabled = os.getenv("STEP3_DISABLE_LLM_ENRICHMENT", "0") == "1" or not bool(options.get("llm_enrichment_enabled", True))
    api_key = os.getenv("UPSTAGE_API_KEY", "")

    if disabled or not api_key:
        return {
            "recommended_actions": _ensure_reference_on_actions(actions, action_evidence),
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

    llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.2)
    band = str(state.get("success_probability_public", {}).get("band", "LOW"))

    hitl_triggered = False
    raw_text = ""
    latency_ms = 0
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
            if not enriched_by_id and idx < len(attempts):
                continue

            if enriched_by_id:
                new_actions: list[RecommendedAction] = []
                for action in actions:
                    action_id = str(action.get("action_id", ""))
                    detail = enriched_by_id.get(action_id, str(action.get("detail", "")))
                    patched = dict(action)
                    patched["detail"] = _enforce_reference(detail, action_evidence.get(action_id, []))
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
                hitl_triggered = _trigger_hitl_retry(state, reason="timeout", attempt=idx, latency_ms=int(timeout_s * 1000))
            continue
        except Exception:
            continue

    return {
        "recommended_actions": _ensure_reference_on_actions(actions, action_evidence),
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


def _mask_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return text

    def _mask_token(token: str) -> str:
        token = token.strip()
        if not token:
            return token
        if re.fullmatch(r"[가-힣]{2,4}", token):
            return token[0] + ("*" * (len(token) - 1))
        if re.fullmatch(r"[A-Za-z][A-Za-z'.-]{1,}", token):
            return token[0] + ("*" * (len(token) - 1))
        return token

    parts = re.split(r"(\s+)", text)
    return "".join(_mask_token(part) if idx % 2 == 0 else part for idx, part in enumerate(parts))


def _mask_digits_keep_tail(text: str, *, keep_tail: int = 2) -> str:
    chars = list(str(text or ""))
    digit_positions = [idx for idx, ch in enumerate(chars) if ch.isdigit()]
    if not digit_positions:
        return "".join(chars)
    keep_set = set(digit_positions[-max(0, keep_tail) :]) if keep_tail > 0 else set()
    for idx in digit_positions:
        if idx not in keep_set:
            chars[idx] = "*"
    return "".join(chars)


def _mask_alnum_keep_tail(text: str, *, keep_tail: int = 2) -> str:
    chars = list(str(text or ""))
    positions = [idx for idx, ch in enumerate(chars) if ch.isalnum()]
    if not positions:
        return "".join(chars)
    keep_set = set(positions[-max(0, keep_tail) :]) if keep_tail > 0 else set()
    for idx in positions:
        if idx not in keep_set:
            chars[idx] = "*"
    return "".join(chars)


def _mask_email_in_text(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        local = match.group("local")
        domain = match.group("domain")
        masked_local = local[0] + ("*" * max(len(local) - 1, 1))
        domain_parts = domain.split(".")
        if domain_parts:
            head = domain_parts[0]
            domain_parts[0] = head[0] + ("*" * max(len(head) - 1, 1)) if head else "***"
        return f"{masked_local}@{'.'.join(domain_parts)}"

    return re.sub(
        r"(?P<local>[A-Za-z0-9._%+-]+)@(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        _repl,
        str(text or ""),
    )


def _mask_phone_in_text(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}-****-**{match.group(3)[-2:]}"

    return re.sub(r"\b(01[0-9]|0[2-9][0-9]?)-?(\d{3,4})-?(\d{4})\b", _repl, str(text or ""))


def _mask_pii_text(text: str) -> str:
    masked = str(text or "")
    if not masked:
        return masked

    label_pattern = re.compile(
        r"(?P<label>"
        r"(?:수신|성명|이름|환자명|피보험자|계약자|수익자|작성자|대상자|생년월일|주민등록번호|"
        r"계약번호|증권번호|전화번호|휴대전화|연락처|이메일|주소|계좌번호)\s*[:：]\s*)"
        r"(?P<value>[^\n]+)"
    )

    def _label_repl(match: re.Match[str]) -> str:
        label = match.group("label")
        value = match.group("value").strip()
        normalized_label = label.replace(" ", "")
        if any(key in normalized_label for key in ("성명", "이름", "환자명", "피보험자", "계약자", "수익자", "작성자", "대상자", "수신")):
            masked_value = _mask_name(value)
        elif "주민등록번호" in normalized_label:
            masked_value = re.sub(r"(\d{6})[- ]?(\d{7})", r"\1-*******", value)
        elif any(key in normalized_label for key in ("전화번호", "휴대전화", "연락처")):
            masked_value = _mask_phone_in_text(value)
        elif "이메일" in normalized_label:
            masked_value = _mask_email_in_text(value)
        elif any(key in normalized_label for key in ("계약번호", "증권번호", "계좌번호")):
            masked_value = _mask_alnum_keep_tail(value, keep_tail=2)
        elif "생년월일" in normalized_label:
            masked_value = re.sub(r"\b(\d{4})[-./](\d{2})[-./](\d{2})\b", r"\1-**-**", value)
            masked_value = re.sub(r"\b(\d{4})(\d{2})(\d{2})\b", r"\1****", masked_value)
        elif "주소" in normalized_label:
            core = value[:6]
            masked_value = f"{core}***" if value else value
        else:
            masked_value = value
        return f"{label}{masked_value}"

    masked = label_pattern.sub(_label_repl, masked)
    masked = re.sub(
        r"(계약번호|증권번호|계좌번호)\s*[:：]\s*([A-Za-z0-9-]+)",
        lambda m: f"{m.group(1)}: {_mask_alnum_keep_tail(m.group(2), keep_tail=2)}",
        masked,
    )
    masked = re.sub(r"\b(\d{6})[- ]?([1-4]\d{6})\b", r"\1-*******", masked)
    masked = _mask_phone_in_text(masked)
    masked = _mask_email_in_text(masked)
    masked = re.sub(r"\b([가-힣]{2,4})(?=\s*고객님\b)", lambda m: _mask_name(m.group(1)), masked)
    return masked


def _mask_recommended_actions(actions: list[RecommendedAction]) -> list[RecommendedAction]:
    masked_actions: list[RecommendedAction] = []
    for action in actions:
        patched = dict(action)
        patched["title"] = _mask_pii_text(str(patched.get("title", "")))
        patched["detail"] = _mask_pii_text(str(patched.get("detail", "")))
        masked_actions.append(RecommendedAction(**patched))
    return masked_actions


def _mask_user_guidance(guidance: UserGuidance) -> UserGuidance:
    issue_brief_rows = []
    for row in guidance.get("issue_brief", []):
        issue_brief_rows.append(
            {
                "issue_id": str(row.get("issue_id", "")),
                "title": _mask_pii_text(str(row.get("title", ""))),
                "why_it_matters": _mask_pii_text(str(row.get("why_it_matters", ""))),
                "needed_evidence": _mask_pii_text(str(row.get("needed_evidence", ""))),
            }
        )

    evidence_rows = []
    for row in guidance.get("evidence_guide", []):
        evidence_rows.append(
            {
                "ref_token": str(row.get("ref_token", "")),
                "source_label": str(row.get("source_label", "")),
                "title": _mask_pii_text(str(row.get("title", ""))),
                "why_relevant": _mask_pii_text(str(row.get("why_relevant", ""))),
            }
        )

    return UserGuidance(
        plain_summary=_mask_pii_text(str(guidance.get("plain_summary", ""))),
        issue_brief=cast(Any, issue_brief_rows),
        next_steps=[_mask_pii_text(str(step)) for step in guidance.get("next_steps", [])],
        evidence_guide=cast(Any, evidence_rows),
        disclaimer=_mask_pii_text(str(guidance.get("disclaimer", ""))),
    )


def _simplify_korean_terms(text: str) -> str:
    normalized = str(text or "")
    if not normalized.strip():
        return ""

    # Keep replacements idempotent to avoid nested phrases like
    # "건강보험 미적용 항목(건강보험 미적용 항목(비급여))".
    replacements = [
        (r"(?<!건강보험 미적용 항목\()비급여", "건강보험 미적용 항목(비급여)"),
        (r"(?<!본인 부담금\()자기부담금", "본인 부담금(자기부담금)"),
        (r"(?<!보상 제외\()면책", "보상 제외(면책)"),
        (r"지급거절", "보험금 지급 거절"),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)

    # Normalize duplicated parenthetical expansions that can still appear
    # from upstream text.
    normalized = normalized.replace(
        "건강보험 미적용 항목(건강보험 미적용 항목(비급여))",
        "건강보험 미적용 항목(비급여)",
    )
    normalized = normalized.replace(
        "본인 부담금(본인 부담금(자기부담금))",
        "본인 부담금(자기부담금)",
    )
    normalized = normalized.replace(
        "건강보험 미적용 항목(비급여) 항목",
        "건강보험 미적용 항목(비급여)",
    )
    normalized = normalized.replace(
        "급여/건강보험 미적용 항목(비급여)",
        "급여/비급여",
    )

    return _trim_for_user(normalized)


def _trim_for_user(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    condensed: list[str] = []
    blank_emitted = False
    for line in normalized_lines:
        if not line:
            if not blank_emitted:
                condensed.append("")
            blank_emitted = True
            continue
        condensed.append(line)
        blank_emitted = False
    return "\n".join(condensed).strip()


def _parse_numbered_sections(text: str) -> dict[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return {}

    lines = raw.splitlines()
    header_with_title = re.compile(r"^\s*(\d+)\s*[\)\.\-]\s*(.+?)\s*$")
    header_only = re.compile(r"^\s*(\d+)\s*[\)\.\-]\s*$")

    alias_map = {
        "지금 상황": "situation",
        "현재 상황": "situation",
        "보험사 판단": "insurer_claim",
        "보험사 주장": "insurer_claim",
        "약관 근거": "policy_basis",
        "약관": "policy_basis",
        "문서에서 확인된 사실": "document_facts",
        "문서 근거": "document_facts",
        "한 줄 결론": "conclusion",
        "결론": "conclusion",
        "다음 단계": "next_step_hint",
        "후속 조치": "next_step_hint",
        "쉬운 설명": "plain_explanation",
        "요약 설명": "plain_explanation",
        "안내": "notice",
        "유의사항": "notice",
    }

    def normalize_title(raw_title: str) -> str:
        title = _trim_for_user(raw_title)
        return alias_map.get(title, title)

    sections: dict[str, str] = {}
    current_number: str | None = None
    current_key: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_number, current_key, current_lines
        if current_number:
            body = _trim_for_user("\n".join(current_lines))
            if body:
                sections[current_number] = body
            if current_key:
                sections[current_key] = body
        current_number = None
        current_key = None
        current_lines = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        with_title = header_with_title.match(line)
        only_number = header_only.match(line)

        if with_title:
            flush()
            current_number = str(with_title.group(1))
            current_key = normalize_title(with_title.group(2))
            idx += 1
            continue

        if only_number:
            flush()
            current_number = str(only_number.group(1))
            next_title = ""
            if idx + 1 < len(lines):
                candidate = lines[idx + 1].strip()
                if candidate and not header_with_title.match(candidate) and not header_only.match(candidate):
                    next_title = candidate
                    idx += 1
            current_key = normalize_title(next_title) if next_title else None
            idx += 1
            continue

        if current_number:
            current_lines.append(line)
        idx += 1

    flush()
    return sections


def _extract_step2_context(structured_case: StructuredCase) -> dict[str, Any]:
    user_info = structured_case.get("user_info", {})
    if not isinstance(user_info, dict):
        user_info = {}

    decision_explanation = str(user_info.get("decision_explanation", "")).strip()
    sections = _parse_numbered_sections(decision_explanation)

    def _strip_bullet_prefix(text: str) -> str:
        return re.sub(r"^\s*-\s*", "", str(text or "")).strip()

    situation = (
        sections.get("situation")
        or sections.get("1")
        or str(user_info.get("decision_summary_user_situation", "")).strip()
        or "현재 청구/거절 상황 요약 정보가 부족합니다."
    )
    insurer_claim = (
        sections.get("insurer_claim")
        or sections.get("2")
        or str(user_info.get("decision_summary_insurer_claim", "")).strip()
        or "보험사의 거절 사유 요약 정보가 부족합니다."
    )
    conclusion = (
        sections.get("conclusion")
        or sections.get("5")
        or str(user_info.get("decision_summary_conclusion_reason", "")).strip()
        or "추가 증빙을 통해 재심의 반박 근거를 보강해야 합니다."
    )
    plain_explanation = sections.get("plain_explanation") or sections.get("7") or ""
    next_step_hint = sections.get("next_step_hint") or sections.get("6") or ""

    required_documents_raw = user_info.get("required_documents", [])
    required_documents = [str(x).strip() for x in required_documents_raw if str(x).strip()] if isinstance(required_documents_raw, list) else []

    return {
        "situation": _trim_for_user(_simplify_korean_terms(_strip_bullet_prefix(situation))),
        "insurer_claim": _trim_for_user(_simplify_korean_terms(_strip_bullet_prefix(insurer_claim))),
        "conclusion": _trim_for_user(_simplify_korean_terms(_strip_bullet_prefix(conclusion))),
        "plain_explanation": _trim_for_user(_simplify_korean_terms(_strip_bullet_prefix(plain_explanation))),
        "next_step_hint": _trim_for_user(_simplify_korean_terms(_strip_bullet_prefix(next_step_hint))),
        "required_documents": required_documents,
    }


def _direction_by_band(band: str) -> str:
    if band == "HIGH":
        return "현재 자료를 정리해 빠르게 접수하는 전략이 유리합니다."
    if band == "MEDIUM":
        return "핵심 쟁점별 반박과 증빙 보강을 병행하는 전략이 필요합니다."
    return "즉시 제출보다 증빙을 먼저 보강한 뒤 재심의를 준비하는 것이 안전합니다."


def _build_next_steps(
    actions: list[RecommendedAction],
    required_documents: list[str],
    conclusion: str,
    next_step_hint: str = "",
) -> list[str]:
    if not actions:
        fallback_reason = _trim_for_user(next_step_hint or conclusion) or "핵심 쟁점 기준으로 보강 자료를 먼저 정리하세요."
        return [
            "\n".join(
                [
                    "1) 무엇: 제출 전 최종 체크를 진행하세요.",
                    "   방법: 핵심 증빙과 약관 근거를 1:1로 매칭해 제출 순서를 정리하세요.",
                    f"   이유: {fallback_reason}",
                    "   준비물: 진단서/진료기록/약관 사본",
                    "   완료기준: 제출 체크리스트를 완료하고 누락 서류가 없는지 확인했습니다.",
                ]
            )
        ]

    steps: list[str] = []
    seen_keys: set[str] = set()
    used_reasons: set[str] = set()
    fallback_reason = _trim_for_user(next_step_hint or conclusion) or "심사자가 쟁점을 빠르게 확인할 수 있습니다."
    default_docs = ["진단서/의사 소견서", "진료비 세부산정내역서/영수증", "보험증권/가입내역서"]
    doc_pool = [d for d in required_documents if d] or default_docs

    def _strip_reference_tokens(text: str) -> str:
        cleaned = re.sub(r"\s*근거:\s*.+$", "", str(text or ""), flags=re.DOTALL).strip()
        return _trim_for_user(cleaned)

    def _pick_reason(candidates: list[str], base_reason: str) -> str:
        for candidate in candidates:
            normalized = _trim_for_user(candidate)
            if normalized and normalized not in used_reasons:
                used_reasons.add(normalized)
                return normalized
        normalized_base = _trim_for_user(base_reason)
        if normalized_base and normalized_base not in used_reasons:
            used_reasons.add(normalized_base)
            return normalized_base
        return normalized_base or "심사자가 핵심 쟁점을 빠르게 확인할 수 있습니다."

    def _short_reason_for_action(title: str, base_reason: str) -> str:
        lowered = title.lower()
        if "증빙" in lowered:
            return _pick_reason(
                [
                    "보험사 판단을 반박할 객관 자료가 필요합니다.",
                    "심사자가 쟁점별 사실관계를 누락 없이 확인할 수 있습니다.",
                ],
                base_reason,
            )
        if "반박" in lowered:
            return _pick_reason(
                [
                    "쟁점별 반박 포인트를 심사자가 한 번에 이해할 수 있습니다.",
                    "쟁점별 주장과 근거를 분리해 제출하면 재심의 판단이 쉬워집니다.",
                ],
                base_reason,
            )
        if "제출" in lowered:
            return _pick_reason(
                [
                    "심사 지연을 줄이고 재심의 판단 속도를 높일 수 있습니다.",
                    "제출 패키지 완성도를 높여 불필요한 보완 요청을 줄일 수 있습니다.",
                ],
                base_reason,
            )
        return _pick_reason(["우선순위 높은 쟁점부터 정리하면 대응 품질이 올라갑니다."], base_reason)

    def _completion_criteria(title: str) -> str:
        lowered = title.lower()
        if "증빙" in lowered:
            return "필수 증빙 원본/사본과 핵심 수치 요약표를 모두 준비했습니다."
        if "반박" in lowered:
            return "쟁점별 반박서와 근거 문서 매핑표가 완성되었습니다."
        if "제출" in lowered:
            return "제출 체크리스트를 완료하고 접수 내역을 확인했습니다."
        return "다음 단계로 넘어가기 위한 필수 자료 확인을 마쳤습니다."

    for idx, action in enumerate(actions, start=1):
        title = _trim_for_user(str(action.get("title", ""))) or f"단계 {idx}"
        key = re.sub(r"\s+", "", title).lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)

        detail_raw = _strip_reference_tokens(str(action.get("detail", "")))
        detail = _trim_for_user(_simplify_korean_terms(detail_raw))
        if not detail:
            detail = "핵심 쟁점에 맞춰 제출 자료를 정리하세요."

        reason = _trim_for_user(
            _simplify_korean_terms(_short_reason_for_action(title, fallback_reason))
        )
        doc_hint = doc_pool[min(len(doc_pool) - 1, len(steps))]
        step_text = (
            f"{len(steps)+1}) 무엇: {title}\n"
            f"   방법: {detail}\n"
            f"   이유: {reason}\n"
            f"   준비물: {doc_hint}\n"
            f"   완료기준: {_completion_criteria(title)}"
        )
        steps.append(step_text)
        if len(steps) >= 5:
            break

    return steps


def _action_title_for_issue(actions: list[RecommendedAction], issue_id: str) -> str:
    for action in actions:
        if str(action.get("linked_issue_id", "")) == str(issue_id):
            return str(action.get("title", "")).strip() or "핵심 반박 준비"
    return str(actions[0].get("title", "핵심 반박 준비")).strip() if actions else "핵심 반박 준비"


def _build_issue_brief(
    issue_nodes: list[IssueNode],
    gap_analysis: list[GapAnalysisItem],
    required_documents: list[str],
    max_items: int = 3,
) -> list[IssueBriefItem]:
    if not issue_nodes:
        return []

    gap_map: dict[str, list[GapAnalysisItem]] = {}
    for gap in gap_analysis:
        issue_id = str(gap.get("issue_id", "ISSUE-1"))
        gap_map.setdefault(issue_id, []).append(gap)
    for issue_id, rows in gap_map.items():
        rows.sort(key=lambda x: _priority_rank(str(x.get("impact", "medium"))))
        gap_map[issue_id] = rows

    briefs: list[IssueBriefItem] = []
    default_docs = [d for d in required_documents if d] or ["진단서/의사 소견서", "진료기록/영수증", "약관 사본"]
    for idx, issue in enumerate(issue_nodes[:max_items]):
        issue_id = str(issue.get("issue_id", f"ISSUE-{idx+1}"))
        title = _trim_for_user(str(issue.get("title", "")).strip()) or f"쟁점 {idx+1}"
        rows = gap_map.get(issue_id, [])
        first_gap = rows[0] if rows else {}
        why_it_matters = _trim_for_user(str(first_gap.get("rationale", "")).strip()) or _trim_for_user(
            str(issue.get("description", "")).strip()
        )
        if not why_it_matters:
            why_it_matters = "재심의 반박 논리의 핵심 판단 기준이 되는 쟁점입니다."
        needed_evidence = _trim_for_user(str(first_gap.get("missing_evidence", "")).strip())
        if not needed_evidence:
            needed_evidence = default_docs[min(idx, len(default_docs) - 1)]

        briefs.append(
            IssueBriefItem(
                issue_id=issue_id,
                title=title,
                why_it_matters=why_it_matters,
                needed_evidence=needed_evidence,
            )
        )
    return briefs


def _build_user_guidance_node(state: DataAnalysisState) -> DataAnalysisState:
    band = str(state.get("success_probability_public", {}).get("band", "LOW"))
    actions = _mask_recommended_actions(list(state.get("recommended_actions", [])))
    evidence_pack = list(state.get("evidence_pack", []))
    gap_analysis = list(state.get("gap_analysis", []))
    issue_nodes = list(state.get("issue_tree", {}).get("nodes", []))
    issue_count = len(issue_nodes)
    step2_ctx = _extract_step2_context(state.get("structured_case", StructuredCase()))
    high_impact_gap_count = len([item for item in gap_analysis if str(item.get("impact", "")).lower() == "high"])
    issue_brief = _build_issue_brief(
        issue_nodes=issue_nodes,
        gap_analysis=gap_analysis,
        required_documents=step2_ctx.get("required_documents", []),
        max_items=3,
    )

    plain_summary_lines = [
        f"- 현재 판단: 재심의 성공 가능성은 {band} 구간입니다.",
        f"- 권장 방향: {_direction_by_band(band)}",
        f"- 진행 포인트: 핵심 쟁점 {issue_count}개 중 우선순위 높은 쟁점부터 증빙과 반박 논리를 정리하세요.",
        f"- 준비 상태: 확보 근거 {len(evidence_pack)}건, 추가 보강 필요 항목 {high_impact_gap_count}건.",
    ]
    plain_summary = "\n".join(_trim_for_user(part) for part in plain_summary_lines if part).strip()

    next_steps = _build_next_steps(
        actions=actions,
        required_documents=step2_ctx.get("required_documents", []),
        conclusion=step2_ctx.get("conclusion", ""),
        next_step_hint=step2_ctx.get("next_step_hint", ""),
    )

    evidence_guide: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for item in evidence_pack:
        ref_token = _ref_from_evidence(item)
        if not ref_token or ref_token in seen_refs:
            continue
        seen_refs.add(ref_token)
        source_type = ref_token.split(":", 1)[0] if ":" in ref_token else "CASELAW"
        source_label = _source_label_ko(source_type)
        title = _title_from_evidence(item)
        issue_id = str(item.get("issue_id", "ISSUE-1"))
        action_title = _action_title_for_issue(actions, issue_id)
        why = _trim_for_user(
            _simplify_korean_terms(
                f"이 근거는 {issue_id} 쟁점을 뒷받침하며, '{action_title}' 단계에서 반박 논리를 설명할 때 사용됩니다."
            )
        )
        evidence_guide.append(
            {
                "ref_token": ref_token,
                "source_label": source_label,
                "title": title,
                "why_relevant": why,
            }
        )
        if len(evidence_guide) >= 5:
            break

    user_guidance = UserGuidance(
        plain_summary=f"{plain_summary}\n- 참고: 핵심 쟁점 {issue_count}개, 근거 {len(evidence_pack)}건",
        issue_brief=issue_brief,
        next_steps=next_steps,
        evidence_guide=evidence_guide,
        disclaimer="본 결과는 재심의 준비를 돕기 위한 참고 정보이며, 최종 판단은 담당 전문가 검토가 필요합니다.",
    )
    return {"user_guidance": _mask_user_guidance(user_guidance)}


def _finalize_node(state: DataAnalysisState) -> DataAnalysisState:
    masked_actions = _mask_recommended_actions(list(state.get("recommended_actions", [])))
    raw_user_guidance = state.get(
        "user_guidance",
        UserGuidance(
            plain_summary="근거 정보가 제한적이므로 우선 증빙 보강부터 진행하세요.",
            issue_brief=[],
            next_steps=["1. 필수 증빙을 먼저 확보한 뒤 재심의 전략을 다시 점검하세요."],
            evidence_guide=[],
            disclaimer="본 결과는 참고용입니다. 최종 판단은 전문가와 함께 진행하세요.",
        ),
    )
    result = AnalysisResult(
        issue_tree=state.get("issue_tree", IssueTree(root_title="보험금 부지급 재심의 쟁점", nodes=[])),
        gap_analysis=state.get("gap_analysis", []),
        recommended_actions=masked_actions,
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
        user_guidance=_mask_user_guidance(raw_user_guidance),
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
    graph.add_node("build_user_guidance", _build_user_guidance_node)
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
    graph.add_edge("llm_action_enrichment", "build_user_guidance")
    graph.add_edge("build_user_guidance", "finalize")
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
        raw_doc_id = _strip_known_source_prefix(str(doc.get("doc_id", "")))
        ingestion_inputs.append(
            {
                "source_type": doc.get("source_type", source_filter),
                "doc_id": raw_doc_id,
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
    items.sort(key=lambda x: float(x.get("rerank_score", x.get("score", 0.0)) or 0.0), reverse=True)
    merged_items = items[:6]
    return RAGRetrievalResult(
        query_id=query_id,
        query=f"{precedent.get('query','')} || {dispute.get('query','')}",
        retrieval_mode=cast(RetrievalMode, precedent.get("retrieval_mode", "plain")),
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


def run_pipeline(
    structured_case: StructuredCase | dict[str, Any],
    rag_result: dict[str, Any] | None = None,
    analysis_options: dict[str, Any] | None = None,
) -> AnalysisResult:
    normalized_case = normalize_structured_case(structured_case)
    options = analysis_options or {"risk_level": "BALANCED", "output_style": "USER_READABLE"}

    initial_state = DataAnalysisState(
        structured_case=normalized_case,
        rag_result=rag_result or {},
        analysis_options=options,
        retrieval_mode=_resolve_retrieval_mode(options),
    )
    app = build_graph()
    result_state: DataAnalysisState = app.invoke(initial_state)
    return result_state["analysis_result"]
