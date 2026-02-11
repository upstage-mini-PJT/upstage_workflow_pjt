"""
Rubric 엔진

YAML 룰셋을 기반으로 성공 확률을 계산합니다.
모든 점수 산정은 설명 가능하며, 근거가 명시됩니다.
"""

import yaml
import os
from typing import Dict, Any, List, Tuple
import sys

# 상위 디렉토리의 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.schemas.analysis import SuccessProbability


# ============================================================================
# 메인 함수
# ============================================================================

def calculate_probability(
    features: Dict[str, Any],
    rubric_path: str = None
) -> SuccessProbability:
    """
    YAML 룰셋 기반 성공 확률 계산
    
    Args:
        features: extract_features()로 추출한 feature dict
        rubric_path: YAML 룰셋 파일 경로 (None이면 기본 경로 사용)
    
    Returns:
        SuccessProbability 객체
    """
    
    # 기본 룰셋 경로
    if rubric_path is None:
        current_dir = os.path.dirname(__file__)
        rubric_path = os.path.join(
            current_dir,
            "../../data/data_analysis_data/rubrics/success_probability_v1.yml"
        )
        rubric_path = os.path.normpath(rubric_path)
    
    # YAML 파일 로드
    rubric = _load_rubric(rubric_path)
    
    # 점수 계산
    base_score = rubric.get("base_score", 50)
    rules = rubric.get("rules", [])
    
    score, drivers_positive, drivers_negative = _apply_rules(features, rules, base_score)
    
    # 점수 범위 제한 (0-100)
    score = max(0, min(100, score))
    
    # 밴드 결정
    band = _determine_band(score, rubric.get("bands", {}))
    
    # 가정 사항
    assumptions = rubric.get("assumptions", [])
    
    return SuccessProbability(
        score_0_to_100=score,
        band=band,
        drivers_positive=drivers_positive,
        drivers_negative=drivers_negative,
        assumptions=assumptions
    )


# ============================================================================
# 내부 함수
# ============================================================================

def _load_rubric(rubric_path: str) -> Dict[str, Any]:
    """YAML 룰셋 파일 로드"""
    
    if not os.path.exists(rubric_path):
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
    
    with open(rubric_path, 'r', encoding='utf-8') as f:
        rubric = yaml.safe_load(f)
    
    return rubric


def _apply_rules(
    features: Dict[str, Any],
    rules: List[Dict[str, Any]],
    base_score: int
) -> Tuple[int, List[str], List[str]]:
    """
    룰 적용 및 점수 계산
    
    Returns:
        (최종 점수, 긍정 요인 리스트, 부정 요인 리스트)
    """
    
    score = base_score
    drivers_positive = []
    drivers_negative = []
    
    for rule in rules:
        condition = rule.get("condition", "")
        score_delta = rule.get("score_delta", 0)
        driver = rule.get("driver", "")
        category = rule.get("category", "positive")
        
        # Condition 평가
        if _evaluate_condition(condition, features):
            score += score_delta
            
            # Driver 추가
            if category == "positive" and score_delta > 0:
                drivers_positive.append(driver)
            elif category == "negative" and score_delta < 0:
                drivers_negative.append(driver)
    
    return score, drivers_positive, drivers_negative


def _evaluate_condition(condition: str, features: Dict[str, Any]) -> bool:
    """
    Condition 평가
    
    condition은 Python 표현식 (예: "has_similar_precedent == True")
    features dict의 값을 사용하여 평가
    
    보안을 위해 제한된 namespace에서만 실행
    """
    
    if not condition:
        return False
    
    try:
        # 안전한 평가를 위한 namespace
        # features의 값만 사용 가능
        namespace = {
            "True": True,
            "False": False,
            "None": None,
            **features  # feature 값들
        }
        
        # eval 실행 (제한된 namespace)
        result = eval(condition, {"__builtins__": {}}, namespace)
        
        return bool(result)
    
    except Exception as e:
        # 평가 실패 시 False 반환 (로그 남기는 것 권장)
        print(f"Warning: Failed to evaluate condition '{condition}': {e}")
        return False


def _determine_band(score: int, bands: Dict[str, Any]) -> str:
    """
    점수에 따른 밴드 결정
    
    Args:
        score: 0-100 점수
        bands: YAML의 bands 섹션
    
    Returns:
        "LOW" | "MEDIUM" | "HIGH"
    """
    
    # 기본 밴드 정의 (YAML에 없을 경우)
    default_bands = {
        "LOW": {"min": 0, "max": 39},
        "MEDIUM": {"min": 40, "max": 69},
        "HIGH": {"min": 70, "max": 100}
    }
    
    bands = bands or default_bands
    
    for band_name, band_config in bands.items():
        min_score = band_config.get("min", 0)
        max_score = band_config.get("max", 100)
        
        if min_score <= score <= max_score:
            return band_name
    
    # 기본값 (범위에 맞지 않으면)
    if score < 40:
        return "LOW"
    elif score < 70:
        return "MEDIUM"
    else:
        return "HIGH"


# ============================================================================
# 유틸리티 함수
# ============================================================================

def explain_score(
    features: Dict[str, Any],
    probability: SuccessProbability,
    rubric_path: str = None
) -> str:
    """
    점수 산정 과정을 설명하는 텍스트 생성 (디버깅/검증용)
    
    Args:
        features: 사용된 feature dict
        probability: 계산된 SuccessProbability
        rubric_path: YAML 룰셋 경로
    
    Returns:
        설명 텍스트
    """
    
    explanation = []
    explanation.append("=" * 60)
    explanation.append("성공 확률 산정 설명")
    explanation.append("=" * 60)
    explanation.append("")
    
    # 최종 점수
    explanation.append(f"최종 점수: {probability.score_0_to_100}점")
    explanation.append(f"밴드: {probability.band}")
    explanation.append("")
    
    # 긍정 요인
    if probability.drivers_positive:
        explanation.append("긍정 요인:")
        for driver in probability.drivers_positive:
            explanation.append(f"  ✓ {driver}")
        explanation.append("")
    
    # 부정 요인
    if probability.drivers_negative:
        explanation.append("부정 요인:")
        for driver in probability.drivers_negative:
            explanation.append(f"  ✗ {driver}")
        explanation.append("")
    
    # 가정 사항
    if probability.assumptions:
        explanation.append("가정 사항:")
        for assumption in probability.assumptions:
            explanation.append(f"  - {assumption}")
        explanation.append("")
    
    # Feature 값들
    explanation.append("사용된 Feature 값:")
    for key, value in sorted(features.items()):
        explanation.append(f"  {key}: {value}")
    
    explanation.append("=" * 60)
    
    return "\n".join(explanation)


def validate_rubric(rubric_path: str) -> Tuple[bool, List[str]]:
    """
    YAML 룰셋 파일 검증
    
    Args:
        rubric_path: YAML 파일 경로
    
    Returns:
        (유효 여부, 오류 메시지 리스트)
    """
    
    errors = []
    
    try:
        rubric = _load_rubric(rubric_path)
    except Exception as e:
        return False, [f"Failed to load rubric: {e}"]
    
    # 필수 필드 체크
    if "base_score" not in rubric:
        errors.append("Missing 'base_score' field")
    
    if "rules" not in rubric:
        errors.append("Missing 'rules' field")
    elif not isinstance(rubric["rules"], list):
        errors.append("'rules' must be a list")
    
    if "bands" not in rubric:
        errors.append("Missing 'bands' field")
    
    # Rule 검증
    for i, rule in enumerate(rubric.get("rules", [])):
        if "condition" not in rule:
            errors.append(f"Rule {i}: missing 'condition'")
        if "score_delta" not in rule:
            errors.append(f"Rule {i}: missing 'score_delta'")
        if "driver" not in rule:
            errors.append(f"Rule {i}: missing 'driver'")
    
    return len(errors) == 0, errors


# ============================================================================
# 테스트용 함수
# ============================================================================

def test_rubric_engine():
    """Rubric 엔진 테스트"""
    
    # 테스트 features
    test_features = {
        "has_similar_precedent": True,
        "precedent_win_rate": 0.75,
        "precedent_count": 6,
        "evidence_completeness": 0.85,
        "has_medical_records": True,
        "has_doctor_note": True,
        "procedural_compliance": True,
        "timeline_consistency": 0.9,
        "policy_clarity": 0.4,
        "clause_interpretation_favorable": True,
        "claim_amount_reasonableness": 0.9,
        "denial_reason_strength": 0.3,
        "has_emergency_situation": True,
        "fact_count": 5
    }
    
    # 점수 계산
    probability = calculate_probability(test_features)
    
    # 결과 출력
    print(explain_score(test_features, probability))
    
    return probability


if __name__ == "__main__":
    # 테스트 실행
    test_rubric_engine()