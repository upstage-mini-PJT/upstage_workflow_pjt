from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from core.schemas.analysis import SuccessProbability

DEFAULT_WEIGHTS: dict[str, float] = {
    "evidence_count": 8.0,
    "retrieved_case_count": 5.0,
    "top_relevance": 40.0,
    "avg_relevance": 30.0,
    "open_question_count": -5.0,
}


def _load_weights(rubric_path: str | None) -> dict[str, float]:
    if not rubric_path:
        return DEFAULT_WEIGHTS
    path = Path(rubric_path)
    if not path.exists():
        return DEFAULT_WEIGHTS
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        weights = loaded.get("weights", {})
        if isinstance(weights, dict) and weights:
            return {str(k): float(v) for k, v in weights.items()}
    except Exception:
        return DEFAULT_WEIGHTS
    return DEFAULT_WEIGHTS


def _band(score: int) -> Literal["LOW", "MEDIUM", "HIGH"]:
    if score <= 39:
        return "LOW"
    if score <= 69:
        return "MEDIUM"
    return "HIGH"


def score_success(
    features: dict[str, float],
    rubric_path: str | None = None,
) -> SuccessProbability:
    weights = _load_weights(rubric_path)
    weighted_sum = 20.0
    for key, weight in weights.items():
        weighted_sum += features.get(key, 0.0) * weight

    raw_score = int(max(0, min(100, round(weighted_sum))))
    band = _band(raw_score)

    positive_drivers: list[str] = []
    negative_drivers: list[str] = []
    if features.get("evidence_count", 0.0) > 0:
        positive_drivers.append("기초 증빙이 확보되어 있음")
    if features.get("retrieved_case_count", 0.0) > 0:
        positive_drivers.append("유사 사례 근거를 확보함")
    if features.get("avg_relevance", 0.0) < 0.1:
        negative_drivers.append("유사 판례와의 정합성이 낮음")
    if features.get("open_question_count", 0.0) > 0:
        negative_drivers.append("해결되지 않은 쟁점 질문이 남아 있음")

    assumptions = [
        "본 점수는 규칙 기반 참고 지표이며 법률 자문을 대체하지 않음",
        "입력 StructuredCase와 mock retrieval 품질에 따라 변동 가능",
    ]

    return SuccessProbability(
        score=raw_score,
        band=band,
        positive_drivers=positive_drivers,
        negative_drivers=negative_drivers,
        assumptions=assumptions,
    )
