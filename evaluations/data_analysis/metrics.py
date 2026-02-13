from __future__ import annotations

from core.schemas.analysis import AnalysisResult


REQUIRED_KEYS = {
    "issue_tree",
    "gap_analysis",
    "recommended_actions",
    "success_probability",
    "evidence_pack",
}


def schema_completeness(result: AnalysisResult) -> float:
    present = sum(1 for key in REQUIRED_KEYS if key in result)
    return present / len(REQUIRED_KEYS)


def provenance_coverage(result: AnalysisResult) -> float:
    evidence_pack = result.get("evidence_pack", [])
    if not evidence_pack:
        return 0.0
    covered = sum(1 for item in evidence_pack if item.get("provenance"))
    return covered / len(evidence_pack)


def band_validity(result: AnalysisResult) -> bool:
    band = result.get("success_probability", {}).get("band")
    return band in {"LOW", "MEDIUM", "HIGH"}
