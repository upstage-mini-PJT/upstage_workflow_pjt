from __future__ import annotations

from typing import Literal, TypedDict

from core.schemas.provenance import Provenance


class RetrievalQuery(TypedDict, total=False):
    query: str
    reason: str
    source: Literal["caselaw", "dispute_case"]
    top_k: int


class CaseLawDoc(TypedDict, total=False):
    doc_id: str
    title: str
    summary: str
    holding: str
    result: str
    keywords: list[str]
    source: str
    source_type: Literal["caselaw", "dispute_case"]
    provenance: list[Provenance]


class RankedCaseLawDoc(CaseLawDoc, total=False):
    relevance_score: float
    matched_keywords: list[str]
