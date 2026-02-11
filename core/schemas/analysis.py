"""
Step3 Data Analysis 출력 스키마 정의

이 모듈은 Step3의 3가지 출력 구조를 정의합니다:
1. AnalysisReport - 쟁점/갭/확률 분석
2. StrategyPlan - 재심의 전략 계획
3. ResearchPack - 판례/사례 근거
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import date


# ============================================================================
# 1. Analysis Report Schemas
# ============================================================================

class ProvenanceRef(BaseModel):
    """근거 참조"""
    ref_type: Literal["CASELAW", "WEB", "REGULATION", "POLICY"]
    ref_id: str
    snippet: Optional[str] = None  # 짧은 발췌


class Issue(BaseModel):
    """쟁점 트리 노드"""
    issue: str = Field(..., description="쟁점 설명")
    sub_issues: List[str] = Field(default_factory=list, description="하위 쟁점")
    provenance: List[ProvenanceRef] = Field(default_factory=list)


class MissingEvidence(BaseModel):
    """부족한 증빙"""
    evidence_type: Literal["MEDICAL_RECORD", "DOCTOR_NOTE", "RECEIPT", "OTHER"]
    why_needed: str = Field(..., description="필요한 이유")
    how_to_get: str = Field(..., description="확보 방법")
    priority: Literal["LOW", "MED", "HIGH"]


class Inconsistency(BaseModel):
    """데이터 불일치"""
    field: str = Field(..., description="불일치 필드")
    detail: str = Field(..., description="불일치 상세")
    priority: Literal["LOW", "MED", "HIGH"]


class GapAnalysis(BaseModel):
    """증빙 갭 분석"""
    missing_evidence: List[MissingEvidence] = Field(default_factory=list)
    inconsistencies: List[Inconsistency] = Field(default_factory=list)


class RecommendedAction(BaseModel):
    """권장 액션"""
    action_id: str
    action: str = Field(..., description="액션 설명")
    rationale: str = Field(..., description="근거")
    effort: Literal["LOW", "MED", "HIGH"]
    expected_impact: Literal["LOW", "MED", "HIGH"]
    provenance: List[ProvenanceRef] = Field(default_factory=list)


class SuccessProbability(BaseModel):
    """재심의 성공 확률"""
    score_0_to_100: int = Field(..., ge=0, le=100, description="0-100 점수")
    band: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="확률 밴드")
    drivers_positive: List[str] = Field(default_factory=list, description="긍정 요인")
    drivers_negative: List[str] = Field(default_factory=list, description="부정 요인")
    assumptions: List[str] = Field(default_factory=list, description="가정 사항")


class AnalysisReport(BaseModel):
    """분석 리포트 (Step3 핵심 출력)"""
    issue_tree: List[Issue] = Field(..., description="쟁점 트리")
    gap_analysis: GapAnalysis = Field(..., description="증빙 갭 분석")
    recommended_actions: List[RecommendedAction] = Field(..., description="권장 액션")
    success_probability: SuccessProbability = Field(..., description="성공 확률")


# ============================================================================
# 2. Strategy Plan Schemas
# ============================================================================

class SupportingPoint(BaseModel):
    """주장 근거 포인트"""
    point: str
    provenance: List[ProvenanceRef] = Field(default_factory=list)


class CounterArgument(BaseModel):
    """예상 반론"""
    counter: str = Field(..., description="예상되는 반론")
    rebuttal_outline: str = Field(..., description="재반박 개요")


class PrimaryClaim(BaseModel):
    """핵심 주장"""
    claim: str = Field(..., description="주장 내용")
    supporting_points: List[SupportingPoint] = Field(default_factory=list)
    counterarguments_expected: List[CounterArgument] = Field(default_factory=list)


class StepTask(BaseModel):
    """실행 단계"""
    step: int
    task: str
    owner: Literal["USER", "SYSTEM"]
    due_days: int = Field(..., description="완료까지 소요 일수")


class StrategyPlan(BaseModel):
    """재심의 전략 계획"""
    strategy_overview: str = Field(..., description="전략 개요 (한 문단)")
    primary_claims: List[PrimaryClaim] = Field(..., description="핵심 주장")
    step_by_step_plan: List[StepTask] = Field(..., description="단계별 실행 계획")
    recommended_channel: Literal[
        "INSURER_REVIEW", 
        "DISPUTE_MEDIATION", 
        "INSURANCE_ASSOCIATION"
    ]
    provenance: List[ProvenanceRef] = Field(default_factory=list)


# ============================================================================
# 3. Research Pack Schemas
# ============================================================================

class CaseLawResult(BaseModel):
    """판례 검색 결과"""
    case_id: str = Field(..., description="외부 판례 ID")
    title: str
    court: str
    date: str = Field(..., description="판결일 (YYYY-MM-DD)")
    holding_summary: str = Field(..., description="판결 요지")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    key_quotes: List[str] = Field(default_factory=list, description="핵심 인용구")
    provenance: List[ProvenanceRef] = Field(default_factory=list)


class WebSource(BaseModel):
    """웹 소스 (사례/기사 등)"""
    source_id: str
    title: str
    publisher: str
    published_at: str = Field(..., description="발행일 (YYYY-MM-DD)")
    summary: str
    provenance: List[ProvenanceRef] = Field(default_factory=list)


class ResearchPack(BaseModel):
    """리서치 결과 패키지"""
    caselaw_results: List[CaseLawResult] = Field(default_factory=list)
    web_sources: List[WebSource] = Field(default_factory=list)


# ============================================================================
# 4. Final Output (Step3 전체 출력)
# ============================================================================

class Meta(BaseModel):
    """공통 메타데이터"""
    step: str = "step3_data_analysis"
    version: str = "1.0"
    timestamp: str
    case_id: Optional[str] = None


class Step3Output(BaseModel):
    """Step3 최종 출력 (3개 JSON 통합)"""
    meta: Meta
    analysis_report: AnalysisReport
    strategy_plan: StrategyPlan
    research_pack: ResearchPack