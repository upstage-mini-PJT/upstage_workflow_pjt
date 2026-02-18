from __future__ import annotations

from typing import Literal, TypedDict

from core.schemas.provenance import Provenance


class RetrievalQuery(TypedDict, total=False):
    query: str
    reason: str
    source: Literal["CASELAW", "DISPUTE", "WEB"]
    top_k: int


class CaseLawDoc(TypedDict, total=False):
    doc_id: str
    title: str
    summary: str
    holding: str
    result: str
    keywords: list[str]
    source: str
    source_type: Literal["CASELAW", "DISPUTE", "WEB"]
    url: str
    published_at: str
    provenance: list[Provenance]


class RankedCaseLawDoc(CaseLawDoc, total=False):
    relevance_score: float
    score: float
    rerank_score: float
    matched_keywords: list[str]
