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
}
