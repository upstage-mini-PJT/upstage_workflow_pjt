from __future__ import annotations

from typing import Final

from core.schemas.rag_contract import SourceType

DOC_ID_SEPARATOR: Final[str] = ":"
CHUNK_ID_SEPARATOR: Final[str] = "#c"

SCORE_MIN: Final[float] = 0.0
SCORE_MAX: Final[float] = 1.0
CASE_ADJUSTMENT_MIN: Final[int] = -15
CASE_ADJUSTMENT_MAX: Final[int] = 15
TOTAL_SCORE_MIN: Final[int] = 0
TOTAL_SCORE_MAX: Final[int] = 100

PROVENANCE_REQUIRED_FIELDS: Final[tuple[str, str, str]] = (
    "index_name",
    "retrieved_at",
    "retrieval_method",
)


def make_doc_id(source_type: SourceType, raw_id: str) -> str:
    cleaned_raw_id = raw_id.strip()
    if not cleaned_raw_id:
        raise ValueError("raw_id must not be empty")
    return f"{source_type}{DOC_ID_SEPARATOR}{cleaned_raw_id}"


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    if not doc_id.strip():
        raise ValueError("doc_id must not be empty")
    if chunk_index < 0:
        raise ValueError("chunk_index must be >= 0")
    return f"{doc_id}{CHUNK_ID_SEPARATOR}{chunk_index}"


def clamp_score(value: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, value))


def clamp_case_adjustment(value: int) -> int:
    return max(CASE_ADJUSTMENT_MIN, min(CASE_ADJUSTMENT_MAX, value))


def clamp_total_score(value: int) -> int:
    return max(TOTAL_SCORE_MIN, min(TOTAL_SCORE_MAX, value))
