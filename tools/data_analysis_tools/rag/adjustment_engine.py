from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from core.schemas.case_context import StructuredCase
from core.schemas.rag_conventions import clamp_case_adjustment
from tools.data_analysis_tools.caselaw.types import RankedCaseLawDoc


class AdjustmentLLM(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class AdjustmentDecision:
    case_adjustment: int
    rationale: str
    cited_case_ids: list[str]
    source: str
    notes: list[str]


def compute_adjustment_with_fallback(
    *,
    structured_case: StructuredCase,
    dispute_ranked_cases: list[RankedCaseLawDoc],
    precedent_score: int,
    features: dict,
    valid_doc_ids: set[str],
    llm: AdjustmentLLM | None = None,
) -> AdjustmentDecision:
    llm_enabled = _is_true(os.getenv("RAG_ENABLE_LLM_ADJUSTMENT", "true"))
    fallback_mode = os.getenv("RAG_ADJUSTMENT_FALLBACK", "heuristic").strip().lower()
    notes: list[str] = []

    llm_decision: AdjustmentDecision | None = None
    llm_failed = False
    if llm_enabled:
        llm_decision = compute_llm_adjustment(
            structured_case=structured_case,
            dispute_ranked_cases=dispute_ranked_cases,
            llm=llm,
        )
        if llm_decision is None:
            llm_failed = True

    if llm_decision is not None:
        raw = llm_decision
    elif fallback_mode == "heuristic":
        if llm_failed:
            notes.append("llm_failed_heuristic_fallback")
        raw = compute_heuristic_adjustment(
            structured_case=structured_case,
            dispute_ranked_cases=dispute_ranked_cases,
            precedent_score=precedent_score,
            features=features,
        )
    else:
        notes.append("fallback_zero_adjustment_applied")
        raw = AdjustmentDecision(
            case_adjustment=0,
            rationale="",
            cited_case_ids=[],
            source="zero",
            notes=[],
        )

    adjusted = clamp_case_adjustment(raw.case_adjustment)
    if adjusted != raw.case_adjustment:
        notes.append("case_adjustment_clamped")

    cited_ids = [cid for cid in raw.cited_case_ids if cid in valid_doc_ids]
    if raw.cited_case_ids and not cited_ids:
        notes.append("adjustment_invalidated_citation_mismatch")
        adjusted = 0

    if adjusted != 0 and not cited_ids:
        notes.append("adjustment_invalidated_missing_citations")
        adjusted = 0

    return AdjustmentDecision(
        case_adjustment=adjusted,
        rationale=raw.rationale.strip(),
        cited_case_ids=cited_ids,
        source=raw.source if adjusted != 0 else ("zero" if raw.source != "heuristic" else "heuristic"),
        notes=raw.notes + notes,
    )


def compute_llm_adjustment(
    *,
    structured_case: StructuredCase,
    dispute_ranked_cases: list[RankedCaseLawDoc],
    llm: AdjustmentLLM | None = None,
) -> AdjustmentDecision | None:
    prompt = _build_adjustment_prompt(structured_case, dispute_ranked_cases)

    try:
        if llm is not None:
            raw_text = llm.generate(prompt)
        else:
            raw_text = os.getenv("RAG_LLM_ADJUSTMENT_MOCK_JSON", "").strip()
        if not raw_text:
            return None
        payload = json.loads(raw_text)
    except Exception:
        return None

    try:
        adjustment = int(round(float(payload.get("case_adjustment", 0))))
    except Exception:
        return None
    cited = [str(x) for x in payload.get("cited_case_ids", []) if str(x).strip()]
    rationale = str(payload.get("rationale", "")).strip()
    return AdjustmentDecision(
        case_adjustment=adjustment,
        rationale=rationale,
        cited_case_ids=cited,
        source="llm",
        notes=[],
    )


def compute_heuristic_adjustment(
    *,
    structured_case: StructuredCase,
    dispute_ranked_cases: list[RankedCaseLawDoc],
    precedent_score: int,
    features: dict,
) -> AdjustmentDecision:
    if not dispute_ranked_cases:
        return AdjustmentDecision(
            case_adjustment=0,
            rationale="사례 없음",
            cited_case_ids=[],
            source="heuristic",
            notes=["사례 기반 보정 미적용: 검색된 사례 없음"],
        )

    top_cases = dispute_ranked_cases[:3]
    adjustment = 0
    notes: list[str] = []
    top_rel = float(top_cases[0].get("relevance_score", 0.0) or 0.0)
    if top_rel >= 0.8:
        adjustment += 6
    elif top_rel >= 0.6:
        adjustment += 3
    elif top_rel < 0.3:
        adjustment -= 3

    win_like = 0
    lose_like = 0
    for row in top_cases:
        result = str(row.get("result", "") or "")
        if any(k in result for k in ["승소", "인용", "조정 성립", "일부 인용"]):
            win_like += 1
        if any(k in result for k in ["패소", "기각", "각하", "불수용"]):
            lose_like += 1
    if win_like >= 2:
        adjustment += 4
    elif lose_like >= 2:
        adjustment -= 4

    if float(features.get("evidence_completeness", 0.0)) >= 0.9 and float(features.get("top_relevance", 0.0)) >= 0.3:
        adjustment += 4
        notes.append("증빙 완결성과 근거 관련도 반영: +4")
    if float(features.get("evidence_completeness", 0.0)) < 0.3 and int(features.get("open_question_count", 0)) >= 2:
        adjustment -= 10
        notes.append("증빙 부족/미해결 쟁점 다수 반영: -10")

    if precedent_score < 20 and adjustment > 8:
        adjustment = 8
        notes.append("저신뢰 1차 점수 보호: +8로 제한")
    if precedent_score > 85 and adjustment < -8:
        adjustment = -8
        notes.append("고신뢰 1차 점수 보호: -8로 제한")
    if precedent_score < 50 and adjustment > 1:
        adjustment = 1
        notes.append("중저신뢰 1차 점수 보호: 양의 보정 +1로 제한")

    cited = [str(row.get("doc_id", "")) for row in top_cases[:2] if row.get("doc_id")]
    return AdjustmentDecision(
        case_adjustment=adjustment,
        rationale="상위 사례 관련도/판정 경향 기반 보정",
        cited_case_ids=cited,
        source="heuristic",
        notes=notes,
    )


def _build_adjustment_prompt(structured_case: StructuredCase, dispute_ranked_cases: list[RankedCaseLawDoc]) -> str:
    summary = str(structured_case.get("denial_summary", "")).strip()
    reasons = ", ".join([str(x) for x in structured_case.get("denial_reasons", [])])
    snippets = []
    for idx, row in enumerate(dispute_ranked_cases[:5], start=1):
        snippets.append(
            f"[{idx}] doc_id={row.get('doc_id','')} title={row.get('title','')} "
            f"score={row.get('relevance_score',0)} result={row.get('result','')}"
        )
    return (
        "다음 분쟁사례를 근거로 case_adjustment를 JSON으로 출력하세요.\n"
        "반드시 JSON 형식: {\"case_adjustment\": int, \"rationale\": str, \"cited_case_ids\": [str,...]}\n"
        f"사건요약: {summary}\n부지급사유: {reasons}\n사례:\n" + "\n".join(snippets)
    )


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
