from __future__ import annotations

EXPECTED_SCENARIO_CONSTRAINTS: dict[str, dict[str, object]] = {
    "S01": {"expected_band": "MEDIUM", "min_total_score": 64, "max_total_score": 72, "min_rag_items": 3},
    "S02": {"expected_band": "MEDIUM", "min_total_score": 58, "max_total_score": 66, "min_rag_items": 3},
    "S03": {"expected_band": "MEDIUM", "min_total_score": 54, "max_total_score": 62, "min_rag_items": 3},
    "S04": {"expected_band": "MEDIUM", "min_total_score": 51, "max_total_score": 59, "min_rag_items": 3},
    "S05": {"expected_band": "MEDIUM", "min_total_score": 53, "max_total_score": 61, "min_rag_items": 3},
    "S06": {"expected_band": "MEDIUM", "min_total_score": 52, "max_total_score": 60, "min_rag_items": 3},
    "S07": {"expected_band": "MEDIUM", "min_total_score": 52, "max_total_score": 60, "min_rag_items": 3},
    "S08": {"expected_band": "MEDIUM", "min_total_score": 53, "max_total_score": 61, "min_rag_items": 3},
    "S09": {"expected_band": "MEDIUM", "min_total_score": 52, "max_total_score": 60, "min_rag_items": 3},
    "S10": {"expected_band": "MEDIUM", "min_total_score": 56, "max_total_score": 64, "min_rag_items": 3},
    "S11": {"expected_band": "MEDIUM", "min_total_score": 42, "max_total_score": 50, "min_rag_items": 3},
    "S12": {"expected_band": "MEDIUM", "min_total_score": 42, "max_total_score": 50, "min_rag_items": 3},
    "S13": {"expected_band": "HIGH", "min_total_score": 70, "max_total_score": 78, "min_rag_items": 3},
    "S14": {"expected_band": "HIGH", "min_total_score": 69, "max_total_score": 77, "min_rag_items": 3},
    "S15": {"expected_band": "MEDIUM", "min_total_score": 53, "max_total_score": 61, "min_rag_items": 3},
    "S16": {"expected_band": "MEDIUM", "min_total_score": 43, "max_total_score": 51, "min_rag_items": 3},
}
