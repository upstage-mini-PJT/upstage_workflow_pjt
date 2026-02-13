from __future__ import annotations

from typing import Any

from agents.data_analysis_agent.pipeline import run_pipeline
from core.schemas.analysis import AnalysisResult
from core.schemas.case_context import StructuredCase


def run(structured_case: StructuredCase | dict[str, Any]) -> AnalysisResult:
    return run_pipeline(structured_case)
