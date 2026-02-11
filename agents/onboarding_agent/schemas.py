"""
planning 노드 등에서 사용하는 Pydantic 스키마.
"""

from pydantic import BaseModel, Field


class PlanningResponse(BaseModel):
    """planning 노드 LLM 응답 스키마."""

    plan: str = Field(description="피보험자 상황 요약 및 분쟁신청을 위한 전략/계획")
    required_documents: list[str] = Field(
        description="분쟁 신청 시 추가로 제출이 필요한 서류 목록",
        default_factory=list,
    )
