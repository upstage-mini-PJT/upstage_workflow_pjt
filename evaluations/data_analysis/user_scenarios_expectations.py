from __future__ import annotations

EXPECTED_SCENARIO_CONSTRAINTS: dict[str, dict[str, object]] = {
    "S01": {"expected_band": "MEDIUM", "min_total_score": 45, "max_total_score": 50, "min_rag_items": 3},
    "S02": {"expected_band": "MEDIUM", "min_total_score": 44, "max_total_score": 48, "min_rag_items": 3},
    "S03": {"expected_band": "MEDIUM", "min_total_score": 40, "max_total_score": 46, "min_rag_items": 3},
    "S04": {"expected_band": "LOW", "min_total_score": 36, "max_total_score": 40, "min_rag_items": 3},
    "S05": {"expected_band": "MEDIUM", "min_total_score": 39, "max_total_score": 44, "min_rag_items": 3},
    "S06": {"expected_band": "LOW", "min_total_score": 37, "max_total_score": 41, "min_rag_items": 3},
    "S07": {"expected_band": "LOW", "min_total_score": 37, "max_total_score": 41, "min_rag_items": 3},
    "S08": {"expected_band": "MEDIUM", "min_total_score": 39, "max_total_score": 43, "min_rag_items": 3},
    "S09": {"expected_band": "MEDIUM", "min_total_score": 39, "max_total_score": 43, "min_rag_items": 3},
    "S10": {"expected_band": "MEDIUM", "min_total_score": 42, "max_total_score": 46, "min_rag_items": 3},
    "S11": {"expected_band": "LOW", "min_total_score": 27, "max_total_score": 33, "min_rag_items": 3},
    "S12": {"expected_band": "LOW", "min_total_score": 28, "max_total_score": 34, "min_rag_items": 3},
    "S13": {"expected_band": "MEDIUM", "min_total_score": 52, "max_total_score": 57, "min_rag_items": 3},
    "S14": {"expected_band": "MEDIUM", "min_total_score": 53, "max_total_score": 58, "min_rag_items": 3},
    "S15": {"expected_band": "MEDIUM", "min_total_score": 38, "max_total_score": 43, "min_rag_items": 3},
    "S16": {"expected_band": "LOW", "min_total_score": 29, "max_total_score": 35, "min_rag_items": 3},
}
