"""
Provenance (근거 추적) 스키마

모든 판단/주장에는 출처가 명시되어야 함
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class ProvenanceSource(BaseModel):
    """근거 출처 (판례/웹/규정/약관)"""
    
    source_type: Literal["CASELAW", "WEB", "REGULATION", "POLICY"]
    source_id: str = Field(..., description="출처 고유 ID")
    title: str
    url: Optional[str] = None
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    snippet: Optional[str] = Field(None, description="짧은 발췌/요약")
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_type": "CASELAW",
                "source_id": "2019다123456",
                "title": "보험금 부지급 처분 취소 청구",
                "url": "https://example.com/case/2019da123456",
                "relevance_score": 0.85,
                "snippet": "의학적 필요성은 담당 의사의 소견을 우선 고려해야 한다"
            }
        }