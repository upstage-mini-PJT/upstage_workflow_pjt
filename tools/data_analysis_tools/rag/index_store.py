from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from core.schemas.rag_contract import VectorChunk

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency guard
    chromadb = None  # type: ignore[assignment]


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


class ChromaVectorIndexStore:
    """Chroma-backed vector store with the same CRUD-like interface."""

    def __init__(
        self,
        index_name: str = "step3_rag_index",
        persist_directory: str = ".chroma_db",
    ) -> None:
        if chromadb is None:
            raise ImportError("chromadb is required for ChromaVectorIndexStore")

        self.index_name = index_name
        self.persist_directory = persist_directory
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=index_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[VectorChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings size mismatch")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk, vector in zip(chunks, embeddings, strict=True):
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id:
                raise ValueError("chunk_id must not be empty")

            expected_dim = int(chunk.get("embedding_dim", 0))
            if expected_dim > 0 and expected_dim != len(vector):
                raise ValueError("embedding dimension mismatch")

            metadata = dict(chunk.get("metadata", {}))
            metadata.update(
                {
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_id": chunk_id,
                    "source_type": chunk.get("source_type", "WEB"),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "embedding_model": chunk.get("embedding_model", ""),
                    "embedding_dim": int(chunk.get("embedding_dim", 0)),
                    "token_count": int(chunk.get("token_count", 0)),
                    "tags_csv": ",".join([str(x) for x in metadata.get("tags", [])]),
                }
            )

            ids.append(chunk_id)
            documents.append(str(chunk.get("chunk_text", "")))
            metadatas.append(metadata)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0 or not query_embedding:
            return []

        where = _filters_to_chroma_where(filters)
        query_result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        ids = query_result.get("ids", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        documents = query_result.get("documents", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        out: list[SearchResult] = []
        for chunk_id, metadata, document, distance in zip(
            ids, metadatas, documents, distances, strict=False
        ):
            if not isinstance(metadata, dict):
                metadata = {}
            chunk = _chroma_row_to_chunk(str(chunk_id), str(document or ""), metadata)
            out.append(SearchResult(chunk=chunk, score=_cosine_from_distance(distance)))
        return out

    def get_by_doc_id(self, doc_id: str) -> list[VectorChunk]:
        rows = self._collection.get(
            where={"doc_id": doc_id.strip()},
            include=["metadatas", "documents"],
        )
        ids = rows.get("ids", [])
        metadatas = rows.get("metadatas", [])
        documents = rows.get("documents", [])

        out: list[VectorChunk] = []
        for chunk_id, metadata, document in zip(ids, metadatas, documents, strict=False):
            if not isinstance(metadata, dict):
                metadata = {}
            out.append(_chroma_row_to_chunk(str(chunk_id), str(document or ""), metadata))
        return out

    def count(self) -> int:
        return int(self._collection.count())

    def count_filtered(self, filters: dict[str, Any] | None = None) -> int:
        if not filters:
            return self.count()
        rows = self._collection.get(where=_filters_to_chroma_where(filters), include=[])
        return len(rows.get("ids", []))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0

    numerator = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def _filters_to_chroma_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    where: dict[str, Any] = {}
    if filters.get("source_type"):
        where["source_type"] = filters["source_type"]
    if filters.get("doc_id"):
        where["doc_id"] = filters["doc_id"]
    return where or None


def _chroma_row_to_chunk(chunk_id: str, document: str, metadata: dict[str, Any]) -> VectorChunk:
    tags_csv = str(metadata.get("tags_csv", ""))
    tags = [tag for tag in tags_csv.split(",") if tag]
    return VectorChunk(
        chunk_id=chunk_id,
        doc_id=str(metadata.get("doc_id", "")),
        source_type=str(metadata.get("source_type", "WEB")),
        chunk_text=document,
        chunk_index=int(metadata.get("chunk_index", 0)),
        embedding_model=str(metadata.get("embedding_model", "")),
        embedding_dim=int(metadata.get("embedding_dim", 0)),
        token_count=int(metadata.get("token_count", 0)),
        metadata={
            "title": str(metadata.get("title", "")),
            "published_at": metadata.get("published_at"),
            "url": metadata.get("url"),
            "tags": tags,
        },
    )


def _cosine_from_distance(distance: Any) -> float:
    try:
        d = float(distance)
    except Exception:
        return 0.0
    # Chroma cosine distance (0 is best); convert to cosine-like score.
    return max(-1.0, min(1.0, 1.0 - d))
