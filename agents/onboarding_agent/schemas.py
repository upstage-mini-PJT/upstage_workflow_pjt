"""Pydantic schemas used by onboarding graph nodes."""

from typing import Literal

from pydantic import BaseModel, Field


class PlanningResponse(BaseModel):
    """planning 노드 LLM 응답 스키마."""

    plan: str = Field(
        description="피보험자 상황 요약(2~3문장) + 거부 사유에 대한 약관·논리 근거를 반영한 분쟁 신청 전략 및 계획(구체적 서술)."
    )
    required_documents: list[str] = Field(
        description="위 전략에 따라 추가 제출이 필요한 서류 목록. 각 항목은 '서류명 (목적/키워드)' 형식, 최대 3개, 구체적 서류 명칭 사용.",
        default_factory=list,
    )


class QueryIntent(BaseModel):
    intent: str = Field(description="검색 의도 요약")
    query_seed: str = Field(description="검색을 위한 핵심 질의 문장")
    must_keywords: list[str] = Field(default_factory=list, description="반드시 포함되면 좋은 키워드")
    priority: int = Field(default=1, ge=1, le=5, description="우선순위(1이 가장 높음)")


class IssuePlanningResponse(BaseModel):
    issue_hypotheses: list[str] = Field(default_factory=list, description="거절사유 가설 목록")
    query_plan: list[QueryIntent] = Field(default_factory=list, description="RAG 질의 계획")


class HyDEQueryResponse(BaseModel):
    hyde_queries: list[str] = Field(default_factory=list, description="의도별 검색 확장 쿼리")


class ExtractedDocumentInfo(BaseModel):
    """parse_and_extract 노드: 문서 하나에서 LLM이 추출한 분쟁 신청에 필요한 정보."""

    key_data: str = Field(description="분쟁 신청에 필요한 핵심 데이터(날짜, 금액, 진단명 등)")
    evidence_or_grounds: str = Field(description="근거가 되는 문구 또는 사실")
    helpful_notes: str = Field(default="", description="기타 도움이 되는 정보")


class SufficiencyResponse(BaseModel):
    """evaluate_sufficiency 노드 LLM 응답 스키마."""

    sufficient: bool = Field(description="분쟁 신청에 필요한 근거가 충분한지 여부")


class ClauseReference(BaseModel):
    title: str = Field(description="약관 섹션/조항 제목")
    snippet: str = Field(description="사용자 설명에 쓸 짧은 근거 요약")


class EvidenceReference(BaseModel):
    source_index: int = Field(ge=1, description="추가 문서 순서(1-based)")
    key_data: str = Field(default="", description="핵심 데이터")
    evidence: str = Field(default="", description="직접 근거 문구/사실")


class DecisionExplanationResponse(BaseModel):
    user_situation: str = Field(description="사용자 현재 상황 요약")
    insurer_claim: str = Field(description="보험사의 지급거절 주장 요약")
    policy_clauses: list[ClauseReference] = Field(default_factory=list)
    document_evidence: list[EvidenceReference] = Field(default_factory=list)
    conclusion_reason: str = Field(description="거절 결론에 이른 이유")
    plain_explanation: str = Field(description="사용자에게 직접 보여줄 쉬운 설명문")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")


class FinalPlanningResponse(BaseModel):
    plan: str = Field(description="약관 근거 반영 최종 전략")
    required_document_ids: list[str] = Field(
        default_factory=list,
        description="문서 카탈로그에서 선택한 최종 필요 서류 ID 목록",
    )
    confidence: Literal["high", "medium", "low"] = Field(default="medium")
