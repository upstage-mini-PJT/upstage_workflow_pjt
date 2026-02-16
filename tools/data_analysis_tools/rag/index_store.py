from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from core.schemas.rag_contract import VectorChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: VectorChunk
    score: float


class InMemoryVectorIndexStore:
    """Simple in-memory vector store with CRUD-like retrieval helpers."""

    def __init__(self, index_name: str = "in_memory_rag") -> None:
        self.index_name = index_name
        self._vectors: dict[str, list[float]] = {}
        self._chunks: dict[str, VectorChunk] = {}

    def upsert(self, chunks: list[VectorChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings size mismatch")

        for chunk, vector in zip(chunks, embeddings, strict=True):
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id:
                raise ValueError("chunk_id must not be empty")

            expected_dim = int(chunk.get("embedding_dim", 0))
            if expected_dim > 0 and expected_dim != len(vector):
                raise ValueError("embedding dimension mismatch")

            self._chunks[chunk_id] = chunk
            self._vectors[chunk_id] = vector

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        if not query_embedding:
            return []

        filtered_ids = self._apply_filters(filters)
        scored: list[SearchResult] = []

        for chunk_id in filtered_ids:
            vector = self._vectors.get(chunk_id)
            chunk = self._chunks.get(chunk_id)
            if vector is None or chunk is None:
                continue

            score = _cosine_similarity(query_embedding, vector)
            scored.append(SearchResult(chunk=chunk, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def get_by_doc_id(self, doc_id: str) -> list[VectorChunk]:
        target = doc_id.strip()
        return [
            chunk
            for chunk in self._chunks.values()
            if str(chunk.get("doc_id", "")).strip() == target
        ]

    def count(self) -> int:
        return len(self._chunks)

    def count_filtered(self, filters: dict[str, Any] | None = None) -> int:
        return len(self._apply_filters(filters))

    def _apply_filters(self, filters: dict[str, Any] | None) -> list[str]:
        if not filters:
            return list(self._chunks.keys())

        source_type = filters.get("source_type")
        doc_id = filters.get("doc_id")

        out: list[str] = []
        for chunk_id, chunk in self._chunks.items():
            if source_type and chunk.get("source_type") != source_type:
                continue
            if doc_id and chunk.get("doc_id") != doc_id:
                continue
            out.append(chunk_id)
        return out


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0

    numerator = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)
