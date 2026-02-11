"""
Step3 입력 스키마

Step2에서 전달되는 구조화된 케이스 정보 및 Step3 실행 옵션
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# 1. StructuredCase (Step2 출력)
# ============================================================================

class Fact(BaseModel):
    """사실 타임라인 항목"""
    date: str = Field(..., description="발생일 (YYYY-MM-DD)")
    event: str = Field(..., description="사건 내용")
    source: Optional[str] = Field(None, description="출처 (진료기록/통지서 등)")


class DenialReason(BaseModel):
    """보험사 부지급 사유"""
    reason_code: str = Field(..., description="사유 코드 (예: PREEXISTING, LACK_OF_NECESSITY)")
    reason_text: str = Field(..., description="사유 설명")
    related_clause: Optional[str] = Field(None, description="관련 약관 조항")


class PolicyClause(BaseModel):
    """약관 조항"""
    clause_id: str
    clause_text: str
    interpretation_notes: Optional[str] = None


class StructuredCase(BaseModel):
    """
    Step2에서 전달되는 구조화된 케이스
    
    Step3는 이 정보를 기반으로 분석을 수행함
    """
    
    case_id: str
    
    # 사실 관계
    facts: List[Fact] = Field(..., description="시간순 사실 타임라인")
    
    # 보험사 부지급 사유
    denial_reasons: List[DenialReason] = Field(..., description="부지급 사유 목록")
    
    # 관련 약관
    policy_clauses: List[PolicyClause] = Field(..., description="관련 약관 조항")
    
    # 추가 정보
    open_questions: List[str] = Field(
        default_factory=list, 
        description="아직 해결되지 않은 질문/불명확한 사항"
    )
    
    # 메타데이터
    insurance_type: Optional[str] = Field(None, description="보험 종류 (실손/암/CI 등)")
    claim_amount: Optional[int] = Field(None, description="청구 금액")
    
    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "CASE-2024-001",
                "facts": [
                    {
                        "date": "2024-01-15",
                        "event": "복통으로 응급실 내원",
                        "source": "진료기록"
                    },
                    {
                        "date": "2024-01-16",
                        "event": "맹장염 진단 후 수술",
                        "source": "진료기록"
                    }
                ],
                "denial_reasons": [
                    {
                        "reason_code": "LACK_OF_NECESSITY",
                        "reason_text": "의학적 필요성이 인정되지 않음",
                        "related_clause": "제5조 2항"
                    }
                ],
                "policy_clauses": [
                    {
                        "clause_id": "제5조 2항",
                        "clause_text": "의학적으로 필요하다고 인정되는 경우에 한하여 보장합니다"
                    }
                ],
                "open_questions": [
                    "응급 상황이었는지 여부가 불명확"
                ],
                "insurance_type": "실손의료보험",
                "claim_amount": 3500000
            }
        }


# ============================================================================
# 2. Research Inputs (판례/웹 검색 쿼리)
# ============================================================================

class DateRange(BaseModel):
    """날짜 범위"""
    from_date: str = Field(..., alias="from", description="시작일 (YYYY-MM-DD)")
    to_date: str = Field(..., alias="to", description="종료일 (YYYY-MM-DD)")
    
    class Config:
        populate_by_name = True


class CaseLawQuery(BaseModel):
    """판례 검색 쿼리"""
    keywords: List[str] = Field(..., description="검색 키워드")
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="필터 조건 (domain, date_range 등)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "keywords": ["현대해상", "실손", "입원", "면책", "의학적 필요성"],
                "filters": {
                    "domain": "MEDICAL_INSURANCE_DISPUTE",
                    "date_range": {
                        "from": "2015-01-01",
                        "to": "2026-02-09"
                    }
                }
            }
        }


class WebSearchQuery(BaseModel):
    """웹 검색 쿼리"""
    keywords: List[str] = Field(..., description="검색 키워드")
    max_results: int = Field(default=10, description="최대 결과 수")
    
    class Config:
        json_schema_extra = {
            "example": {
                "keywords": [
                    "보험금 부지급 이의신청",
                    "분쟁조정",
                    "손해보험협회",
                    "금융감독원 분쟁조정"
                ],
                "max_results": 10
            }
        }


class ResearchInputs(BaseModel):
    """리서치 입력 (판례/웹 검색 쿼리)"""
    caselaw_query: CaseLawQuery
    web_search_query: WebSearchQuery


# ============================================================================
# 3. Analysis Options
# ============================================================================

class AnalysisOptions(BaseModel):
    """분석 옵션"""
    
    risk_level: Literal["CONSERVATIVE", "BALANCED", "AGGRESSIVE"] = Field(
        default="BALANCED",
        description="위험 수준 (보수적/균형/공격적)"
    )
    
    output_style: Literal["USER_READABLE", "EXPERT_LIKE"] = Field(
        default="USER_READABLE",
        description="출력 스타일 (사용자 친화적/전문가 스타일)"
    )
    
    compute_success_probability: bool = Field(
        default=True,
        description="성공 확률 계산 여부"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "risk_level": "BALANCED",
                "output_style": "USER_READABLE",
                "compute_success_probability": True
            }
        }


# ============================================================================
# 4. Step3 전체 입력
# ============================================================================

class Meta(BaseModel):
    """공통 메타데이터"""
    step: str = "step3_data_analysis"
    version: str = "1.0"
    timestamp: Optional[str] = None
    case_id: Optional[str] = None


class Step3Input(BaseModel):
    """
    Step3 전체 입력 스키마
    
    Step2에서 전달되는 structured_case와
    Step3 실행을 위한 research_inputs, analysis_options를 포함
    """
    
    meta: Meta
    
    # Step2 출력
    structured_case: StructuredCase
    user_friendly_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Step2에서 생성한 사용자 친화적 요약 (선택)"
    )
    
    # Step3 실행 옵션
    research_inputs: ResearchInputs
    analysis_options: AnalysisOptions = Field(default_factory=AnalysisOptions)
    
    class Config:
        json_schema_extra = {
            "example": {
                "meta": {
                    "step": "step3_data_analysis",
                    "version": "1.0",
                    "timestamp": "2026-02-11T14:50:00+09:00",
                    "case_id": "CASE-2024-001"
                },
                "structured_case": {
                    "case_id": "CASE-2024-001",
                    "facts": [],
                    "denial_reasons": [],
                    "policy_clauses": []
                },
                "research_inputs": {
                    "caselaw_query": {
                        "keywords": ["현대해상", "실손", "입원"],
                        "filters": {
                            "domain": "MEDICAL_INSURANCE_DISPUTE"
                        }
                    },
                    "web_search_query": {
                        "keywords": ["보험금 부지급 이의신청"],
                        "max_results": 10
                    }
                },
                "analysis_options": {
                    "risk_level": "BALANCED",
                    "output_style": "USER_READABLE",
                    "compute_success_probability": True
                }
            }
        }