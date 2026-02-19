from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from core.schemas.rag_contract import IngestionDoc
from tools.data_analysis_tools.rag.normalizer import (
    IngestionNormalizationError,
    normalize_ingestion_doc,
)


class WebSearchProvider(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        ...


@dataclass(frozen=True)
class MockWebSearchProvider:
    results: list[dict]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        # Local development stub; callers should replace with real provider.
        return self.results[:max(0, top_k)]


def search_web_docs(
    provider: WebSearchProvider,
    query: str,
    top_k: int = 10,
) -> list[IngestionDoc]:
    if not query.strip():
        raise ValueError("query must not be empty")

    raw_results = provider.search(query=query, top_k=top_k)
    return normalize_web_results(raw_results)


def normalize_web_results(raw_results: list[dict]) -> list[IngestionDoc]:
    normalized: list[IngestionDoc] = []
    seen: set[str] = set()

    for raw in raw_results:
        candidate = dict(raw)
        candidate["source_type"] = "WEB"

        try:
            doc = normalize_ingestion_doc(candidate)
        except IngestionNormalizationError:
            continue

        dedup_key = _dedup_key(doc)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        meta = dict(doc.get("meta", {}))
        domain = _extract_domain(doc.get("url"))
        meta["publisher_domain"] = domain
        meta["publisher_grade"] = _publisher_grade(domain)

        normalized.append(
            IngestionDoc(
                **{
                    **doc,
                    "meta": meta,
                }
            )
        )

    return normalized


def _dedup_key(doc: IngestionDoc) -> str:
    url = str(doc.get("url") or "").strip().lower()
    title = str(doc.get("title") or "").strip().lower()
    if url:
        return f"url:{url}"
    return f"title:{title}"


def _extract_domain(url: str | None) -> str:
    if not url:
        return "unknown"
    parsed = urlparse(url)
    return parsed.netloc.lower() or "unknown"


def _publisher_grade(domain: str) -> str:
    if domain.endswith(".go.kr") or domain.endswith(".gov"):
        return "A"
    if domain.endswith(".or.kr") or domain.endswith(".org"):
        return "B"
    if domain.endswith(".ac.kr") or domain.endswith(".edu"):
        return "B"
    if domain == "unknown":
        return "D"
    return "C"
