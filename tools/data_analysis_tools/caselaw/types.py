from __future__ import annotations

from typing import TypedDict

from core.schemas.provenance import Provenance


class RetrievalQuery(TypedDict, total=False):
    query: str
    reason: str
    top_k: int


class CaseLawDoc(TypedDict, total=False):
    doc_id: str
    title: str
    summary: str
    holding: str
    keywords: list[str]
    source: str
    provenance: list[Provenance]


class RankedCaseLawDoc(CaseLawDoc, total=False):
    relevance_score: float
