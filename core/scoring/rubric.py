from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml


class ScoreEstimate(TypedDict, total=False):
    score: int
    band: Literal["LOW", "MEDIUM", "HIGH"]
    positive_drivers: list[str]
    negative_drivers: list[str]
    assumptions: list[str]


DEFAULT_RUBRIC: dict[str, Any] = {
    "base_score": 45,
    "bands": {
        "LOW": {"min": 0, "max": 39},
        "MEDIUM": {"min": 40, "max": 69},
        "HIGH": {"min": 70, "max": 100},
    },
    "assumptions": [
        "본 점수는 규칙 기반 참고 지표이며 법률 자문을 대체하지 않음",
        "입력 데이터 품질과 검색 품질에 따라 변동 가능",
    ],
    "rules": [
        {"condition": "has_similar_precedent == True", "score_delta": 10, "driver": "유사 판례 존재", "category": "positive"},
        {"condition": "precedent_win_rate >= 0.6", "score_delta": 8, "driver": "유사 판례 승소 경향", "category": "positive"},
        {"condition": "top_relevance >= 0.5", "score_delta": 10, "driver": "상위 근거 관련도 높음", "category": "positive"},
        {"condition": "evidence_completeness >= 0.7", "score_delta": 8, "driver": "증빙 완결도 높음", "category": "positive"},
        {"condition": "evidence_completeness >= 0.9", "score_delta": 6, "driver": "핵심 증빙이 충분히 확보됨", "category": "positive"},
        {"condition": "evidence_completeness < 0.3", "score_delta": -10, "driver": "핵심 증빙 부족", "category": "negative"},
        {"condition": "open_question_count >= 2", "score_delta": -7, "driver": "미해결 쟁점 다수", "category": "negative"},
        {"condition": "avg_relevance < 0.2", "score_delta": -8, "driver": "근거 관련도 낮음", "category": "negative"},
    ],
}


def _load_rubric(rubric_path: str | None) -> dict[str, Any]:
    if not rubric_path:
        return DEFAULT_RUBRIC

    path = Path(rubric_path)
    if not path.exists():
        return DEFAULT_RUBRIC

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            merged = dict(DEFAULT_RUBRIC)
            merged.update(loaded)
            return merged
    except Exception:
        return DEFAULT_RUBRIC
    return DEFAULT_RUBRIC


def _evaluate_condition(condition: str, features: dict[str, Any]) -> bool:
    if not condition:
        return False
    namespace = {"True": True, "False": False, "None": None, **features}
    try:
        return bool(eval(condition, {"__builtins__": {}}, namespace))
    except Exception:
        return False


def _score_to_band(score: int, bands: dict[str, Any]) -> Literal["LOW", "MEDIUM", "HIGH"]:
    for name, config in bands.items():
        mn = int(config.get("min", 0))
        mx = int(config.get("max", 100))
        if mn <= score <= mx and name in {"LOW", "MEDIUM", "HIGH"}:
            return name
    if score <= 39:
        return "LOW"
    if score <= 69:
        return "MEDIUM"
    return "HIGH"


def score_success(features: dict[str, Any], rubric_path: str | None = None) -> ScoreEstimate:
    rubric = _load_rubric(rubric_path)
    score = int(rubric.get("base_score", 45))

    positive_drivers: list[str] = []
    negative_drivers: list[str] = []

    for rule in rubric.get("rules", []):
        condition = str(rule.get("condition", ""))
        if not _evaluate_condition(condition, features):
            continue

        delta = int(rule.get("score_delta", 0))
        driver = str(rule.get("driver", ""))
        category = str(rule.get("category", "positive"))

        score += delta
        if delta > 0 and category == "positive" and driver:
            positive_drivers.append(driver)
        if delta < 0 and category == "negative" and driver:
            negative_drivers.append(driver)

    score = max(0, min(100, score))
    band = _score_to_band(score, rubric.get("bands", {}))

    return ScoreEstimate(
        score=score,
        band=band,
        positive_drivers=positive_drivers,
        negative_drivers=negative_drivers,
        assumptions=[str(x) for x in rubric.get("assumptions", [])],
    )
