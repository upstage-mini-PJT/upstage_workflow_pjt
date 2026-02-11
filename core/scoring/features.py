"""
Feature 추출 로직

케이스와 판례 데이터에서 점수 산정에 필요한 특징(feature)을 추출합니다.
모든 feature는 규칙 기반으로 계산되며, 설명 가능성을 보장합니다.
"""

from typing import List, Dict, Any
import sys
import os

# 상위 디렉토리의 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.schemas.case_context import StructuredCase
from tools.data_analysis_tools.caselaw.types import CaseLawDoc


# ============================================================================
# Feature 추출 메인 함수
# ============================================================================

def extract_features(
    case: StructuredCase,
    caselaws: List[CaseLawDoc]
) -> Dict[str, Any]:
    """
    케이스와 판례 데이터에서 점수 산정용 feature 추출
    
    Args:
        case: Step2에서 전달된 구조화된 케이스
        caselaws: 검색된 판례 목록
    
    Returns:
        feature dict with keys:
        - has_similar_precedent: bool
        - precedent_win_rate: float (0-1)
        - precedent_count: int
        - evidence_completeness: float (0-1)
        - has_medical_records: bool
        - has_doctor_note: bool
        - procedural_compliance: bool
        - timeline_consistency: float (0-1)
        - policy_clarity: float (0-1)
        - clause_interpretation_favorable: bool
        - claim_amount_reasonableness: float (0-1)
        - denial_reason_strength: float (0-1)
        - has_emergency_situation: bool
        - fact_count: int
    """
    
    features = {}
    
    # 1. 판례 관련 features
    features.update(_extract_precedent_features(caselaws))
    
    # 2. 증빙 관련 features
    features.update(_extract_evidence_features(case))
    
    # 3. 절차 관련 features
    features.update(_extract_procedural_features(case))
    
    # 4. 약관 관련 features
    features.update(_extract_policy_features(case))
    
    # 5. 기타 features
    features.update(_extract_misc_features(case))
    
    return features


# ============================================================================
# 1. 판례 관련 Features
# ============================================================================

def _extract_precedent_features(caselaws: List[CaseLawDoc]) -> Dict[str, Any]:
    """판례 관련 feature 추출"""
    
    # 유사 판례 존재 여부 (relevance_score > 0.7)
    high_relevance_cases = [c for c in caselaws if c.relevance_score > 0.7]
    has_similar_precedent = len(high_relevance_cases) > 0
    
    # 판례 개수
    precedent_count = len(caselaws)
    
    # 원고 승소율 계산
    precedent_win_rate = _calculate_win_rate(caselaws)
    
    return {
        "has_similar_precedent": has_similar_precedent,
        "precedent_count": precedent_count,
        "precedent_win_rate": precedent_win_rate,
    }


def _calculate_win_rate(caselaws: List[CaseLawDoc]) -> float:
    """
    판례 중 원고 승소율 계산
    
    result 필드에서 "승소", "일부승소", "인용" 등의 키워드로 판단
    """
    if not caselaws:
        return 0.0
    
    win_keywords = ["승소", "인용", "취소"]
    lose_keywords = ["기각", "패소", "각하"]
    
    wins = 0
    total = 0
    
    for case in caselaws:
        if case.result:
            result_lower = case.result.lower()
            
            if any(keyword in result_lower for keyword in win_keywords):
                wins += 1
                total += 1
            elif any(keyword in result_lower for keyword in lose_keywords):
                total += 1
    
    return wins / total if total > 0 else 0.5  # 정보 없으면 중립 0.5


# ============================================================================
# 2. 증빙 관련 Features
# ============================================================================

def _extract_evidence_features(case: StructuredCase) -> Dict[str, Any]:
    """증빙 관련 feature 추출"""
    
    # 사실 관계에서 증빙 출처 확인
    sources = [fact.source for fact in case.facts if fact.source]
    
    # 진료기록 존재 여부
    has_medical_records = any("진료기록" in s or "의무기록" in s for s in sources)
    
    # 의사 소견서 존재 여부
    has_doctor_note = any("소견서" in s or "진단서" in s for s in sources)
    
    # 증빙 완전성 계산 (0-1)
    evidence_score = 0.0
    if has_medical_records:
        evidence_score += 0.4
    if has_doctor_note:
        evidence_score += 0.3
    if len(sources) >= 3:  # 다양한 출처
        evidence_score += 0.2
    if case.facts:  # 사실 관계가 기록됨
        evidence_score += 0.1
    
    evidence_completeness = min(evidence_score, 1.0)
    
    return {
        "has_medical_records": has_medical_records,
        "has_doctor_note": has_doctor_note,
        "evidence_completeness": evidence_completeness,
    }


# ============================================================================
# 3. 절차 관련 Features
# ============================================================================

def _extract_procedural_features(case: StructuredCase) -> Dict[str, Any]:
    """절차 관련 feature 추출"""
    
    # 타임라인 일관성 체크
    timeline_consistency = _check_timeline_consistency(case.facts)
    
    # 절차 준수 여부 (간단한 휴리스틱)
    # 실제로는 더 복잡한 로직 필요 (청구 기한, 서류 제출 등)
    procedural_compliance = True  # 기본값 True, 문제 발견 시 False
    
    # open_questions가 많으면 절차적 문제가 있을 가능성
    if len(case.open_questions) > 3:
        procedural_compliance = False
    
    return {
        "timeline_consistency": timeline_consistency,
        "procedural_compliance": procedural_compliance,
    }


def _check_timeline_consistency(facts: List) -> float:
    """
    타임라인 일관성 체크
    
    날짜 순서가 올바른지, 날짜 간격이 합리적인지 확인
    """
    if len(facts) < 2:
        return 1.0  # 사실이 1개 이하면 일관성 문제 없음
    
    # 날짜 파싱 및 정렬 체크
    try:
        dates = [fact.date for fact in facts if fact.date]
        sorted_dates = sorted(dates)
        
        # 원본 순서와 정렬된 순서가 같으면 일관성 높음
        if dates == sorted_dates:
            return 1.0
        else:
            return 0.7  # 순서가 맞지 않으면 약간 감점
    except:
        return 0.5  # 날짜 파싱 실패 시 중립


# ============================================================================
# 4. 약관 관련 Features
# ============================================================================

def _extract_policy_features(case: StructuredCase) -> Dict[str, Any]:
    """약관 관련 feature 추출"""
    
    # 약관 명확성 (모호한 표현이 있는지)
    policy_clarity = _assess_policy_clarity(case.policy_clauses)
    
    # 약관 해석이 유리한지 (간단한 휴리스틱)
    clause_interpretation_favorable = _is_interpretation_favorable(case)
    
    return {
        "policy_clarity": policy_clarity,
        "clause_interpretation_favorable": clause_interpretation_favorable,
    }


def _assess_policy_clarity(clauses: List) -> float:
    """
    약관 명확성 평가
    
    모호한 표현이 많을수록 점수 낮음
    """
    if not clauses:
        return 0.5
    
    # 모호한 표현 키워드
    ambiguous_keywords = [
        "인정되는", "필요하다고", "상당한", "합리적인",
        "적절한", "통상적인", "일반적인"
    ]
    
    total_ambiguous = 0
    total_clauses = len(clauses)
    
    for clause in clauses:
        text = clause.clause_text
        ambiguous_count = sum(1 for keyword in ambiguous_keywords if keyword in text)
        total_ambiguous += ambiguous_count
    
    # 모호한 표현이 많을수록 명확성 낮음
    if total_ambiguous == 0:
        return 1.0
    elif total_ambiguous <= 2:
        return 0.7
    else:
        return 0.4


def _is_interpretation_favorable(case: StructuredCase) -> bool:
    """
    약관 해석이 사용자에게 유리한지 판단
    
    간단한 휴리스틱: 부지급 사유가 모호한 조항을 근거로 하면 유리
    """
    # 부지급 사유가 모호한 조항에 기반하면 유리
    for denial in case.denial_reasons:
        if denial.related_clause:
            # 관련 조항 찾기
            for clause in case.policy_clauses:
                if clause.clause_id == denial.related_clause:
                    # 모호한 표현이 있으면 해석 여지가 있어 유리
                    if any(keyword in clause.clause_text for keyword in 
                           ["인정되는", "필요하다고", "상당한"]):
                        return True
    
    return False


# ============================================================================
# 5. 기타 Features
# ============================================================================

def _extract_misc_features(case: StructuredCase) -> Dict[str, Any]:
    """기타 feature 추출"""
    
    # 청구 금액 합리성 (간단한 휴리스틱)
    claim_amount_reasonableness = _assess_claim_amount(case.claim_amount)
    
    # 부지급 사유 강도 (사유가 구체적일수록 강함)
    denial_reason_strength = _assess_denial_strength(case.denial_reasons)
    
    # 응급 상황 여부
    has_emergency_situation = _check_emergency(case.facts)
    
    # 사실 개수
    fact_count = len(case.facts)
    
    return {
        "claim_amount_reasonableness": claim_amount_reasonableness,
        "denial_reason_strength": denial_reason_strength,
        "has_emergency_situation": has_emergency_situation,
        "fact_count": fact_count,
    }


def _assess_claim_amount(amount: int) -> float:
    """
    청구 금액 합리성 평가
    
    실손보험 기준 일반적인 범위인지 확인
    """
    if amount is None:
        return 0.5  # 정보 없으면 중립
    
    # 간단한 휴리스틱 (실제로는 보험 종류별로 다름)
    if amount < 100000:  # 10만원 미만
        return 1.0
    elif amount < 1000000:  # 100만원 미만
        return 0.9
    elif amount < 5000000:  # 500만원 미만
        return 0.7
    elif amount < 10000000:  # 1000만원 미만
        return 0.5
    else:
        return 0.3  # 고액 청구는 심사 엄격


def _assess_denial_strength(denial_reasons: List) -> float:
    """
    부지급 사유 강도 평가
    
    사유가 구체적이고 명확할수록 보험사 입장이 강함
    """
    if not denial_reasons:
        return 0.0
    
    # 구체적인 사유 코드
    strong_reasons = ["PREEXISTING", "EXCLUSION", "FRAUD"]
    weak_reasons = ["LACK_OF_NECESSITY", "INSUFFICIENT_EVIDENCE"]
    
    strong_count = sum(1 for d in denial_reasons if d.reason_code in strong_reasons)
    weak_count = sum(1 for d in denial_reasons if d.reason_code in weak_reasons)
    
    if strong_count > 0:
        return 0.8  # 강한 사유
    elif weak_count > 0:
        return 0.4  # 약한 사유
    else:
        return 0.5  # 중립


def _check_emergency(facts: List) -> bool:
    """응급 상황 여부 확인"""
    emergency_keywords = ["응급", "급성", "위급", "긴급"]
    
    for fact in facts:
        if any(keyword in fact.event for keyword in emergency_keywords):
            return True
    
    return False


# ============================================================================
# 유틸리티 함수
# ============================================================================

def get_feature_description(feature_name: str) -> str:
    """Feature 설명 반환 (디버깅/설명용)"""
    
    descriptions = {
        "has_similar_precedent": "유사 판례 존재 여부 (relevance > 0.7)",
        "precedent_count": "검색된 판례 개수",
        "precedent_win_rate": "유사 판례 중 원고 승소율 (0-1)",
        "has_medical_records": "진료기록 존재 여부",
        "has_doctor_note": "의사 소견서 존재 여부",
        "evidence_completeness": "증빙 자료 완전성 (0-1)",
        "procedural_compliance": "절차 준수 여부",
        "timeline_consistency": "타임라인 일관성 (0-1)",
        "policy_clarity": "약관 명확성 (0-1, 높을수록 명확)",
        "clause_interpretation_favorable": "약관 해석이 사용자에게 유리한지",
        "claim_amount_reasonableness": "청구 금액 합리성 (0-1)",
        "denial_reason_strength": "부지급 사유 강도 (0-1, 높을수록 보험사 입장 강함)",
        "has_emergency_situation": "응급 상황 여부",
        "fact_count": "기록된 사실 개수",
    }
    
    return descriptions.get(feature_name, "설명 없음")