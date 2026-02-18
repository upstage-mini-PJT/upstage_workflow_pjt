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
