"""
planning 노드 등에서 사용하는 Pydantic 스키마.
"""

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
