"""
판례/사례 데이터 타입 정의

팀원이 구현할 Retrieval Layer와의 인터페이스 계약
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class CaseLawDoc(BaseModel):
    """
    판례 문서 (팀원이 API에서 가져올 데이터 구조)
    
    이 스키마는 팀원과 합의된 계약입니다.
    """
    
    case_id: str = Field(..., description="판례 고유 ID")
    title: str = Field(..., description="사건명")
    court: str = Field(..., description="법원명")
    date: str = Field(..., description="판결일 (YYYY-MM-DD)")
    
    # 내용
    summary: str = Field(..., description="판결 요지")
    full_text: Optional[str] = Field(None, description="전문 (있는 경우)")
    
    # 메타데이터
    case_type: Optional[str] = Field(None, description="사건 유형")
    result: Optional[str] = Field(None, description="판결 결과 (원고승/기각 등)")
    
    # Retrieval 정보
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="검색 관련도")
    matched_keywords: List[str] = Field(default_factory=list)
    
    # 출처
    url: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "2019다123456",
                "title": "보험금 청구의 소",
                "court": "서울중앙지방법원",
                "date": "2019-05-15",
                "summary": "의학적 필요성은 담당 의사의 소견을 우선 고려해야 한다",
                "case_type": "보험금",
                "result": "원고 일부 승소",
                "relevance_score": 0.85,
                "matched_keywords": ["의학적 필요성", "실손보험"],
                "url": "https://example.com/case/2019da123456"
            }
        }


class WebCaseDoc(BaseModel):
    """
    웹 사례 문서 (분쟁조정/언론 보도 등)
    """
    
    source_id: str
    title: str
    publisher: str = Field(..., description="출처 (금융감독원/소비자원 등)")
    published_at: str = Field(..., description="발행일 (YYYY-MM-DD)")
    
    summary: str
    full_text: Optional[str] = None
    
    # Retrieval 정보
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    
    url: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "fss-2023-001",
                "title": "실손보험 의학적 필요성 분쟁조정 사례",
                "publisher": "금융감독원",
                "published_at": "2023-03-20",
                "summary": "응급 상황에서의 치료는 의학적 필요성이 인정됨",
                "relevance_score": 0.78,
                "url": "https://fss.or.kr/cases/2023-001"
            }
        }


class SearchQuery(BaseModel):
    """
    판례 검색 쿼리 (우리가 팀원에게 전달할 구조)
    """
    
    keywords: List[str] = Field(..., description="검색 키워드")
    filters: dict = Field(default_factory=dict, description="필터 조건")
    top_k: int = Field(default=10, description="상위 K개 결과")
    
    class Config:
        json_schema_extra = {
            "example": {
                "keywords": ["실손보험", "의학적 필요성", "맹장염"],
                "filters": {
                    "court": "서울중앙지방법원",
                    "date_from": "2019-01-01",
                    "date_to": "2024-12-31"
                },
                "top_k": 10
            }
        }