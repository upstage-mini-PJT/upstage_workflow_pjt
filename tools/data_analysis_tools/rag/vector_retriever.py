from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from core.schemas.rag_contract import RAGItem, RAGRetrievalResult, RetrievalProvenance, RetrievalStats, TreeNode
from core.schemas.rag_conventions import SCORE_MAX, SCORE_MIN, clamp_score
from tools.data_analysis_tools.rag.embedder import Embedder
from tools.data_analysis_tools.rag.index_store import InMemoryVectorIndexStore


@dataclass(frozen=True)
class RetrieveRequest:
    query: str
    top_k: int = 10
    filters: dict | None = None
    query_id: str | None = None


class VectorRetriever:
    def __init__(
        self,
        index_store: InMemoryVectorIndexStore,
        embedder: Embedder,
        retrieval_method: str = "vector",
    ) -> None:
        self.index_store = index_store
        self.embedder = embedder
        self.retrieval_method = retrieval_method

    def retrieve(self, request: RetrieveRequest) -> RAGRetrievalResult:
        started_at = perf_counter()
        retrieved_at = datetime.now(tz=timezone.utc).isoformat()

        query = request.query.strip()
        if not query:
            raise ValueError("query must not be empty")

        query_vector = self.embedder.embed_texts([query])[0]
        candidate_count = self.index_store.count_filtered(request.filters)
        results = self.index_store.search(query_vector, top_k=request.top_k, filters=request.filters)

        items: list[RAGItem] = []
        for result in results:
            chunk = result.chunk
            score = _normalize_score(result.score)
            provenance = RetrievalProvenance(
                index_name=self.index_store.index_name,
                retrieved_at=retrieved_at,
                retrieval_method=self.retrieval_method,
            )
            items.append(
                RAGItem(
                    source_type=chunk.get("source_type", "WEB"),
                    doc_id=str(chunk.get("doc_id", "")),
                    chunk_id=str(chunk.get("chunk_id", "")),
                    title=str(chunk.get("metadata", {}).get("title", "")),
                    snippet=_build_snippet(str(chunk.get("chunk_text", ""))),
                    score=score,
                    rerank_score=score,
                    url=chunk.get("metadata", {}).get("url"),
                    published_at=chunk.get("metadata", {}).get("published_at"),
                    provenance=provenance,
                )
            )

        latency_ms = int((perf_counter() - started_at) * 1000)
        stats = RetrievalStats(
            candidate_count=candidate_count,
            returned_count=len(items),
            latency_ms=latency_ms,
        )

        tree = _build_retrieval_tree(query=query, query_id=request.query_id, items=items)

        return RAGRetrievalResult(
            query_id=request.query_id or str(uuid4()),
            query=query,
            filters=request.filters or {},
            items=items,
            tree=tree,
            stats=stats,
        )


def _normalize_score(raw_cosine: float) -> float:
    # Cosine range [-1, 1] -> [0, 1] to match the fixed score contract.
    normalized = (raw_cosine + 1.0) / 2.0
    return clamp_score(max(SCORE_MIN, min(SCORE_MAX, normalized)))


def _build_snippet(text: str, max_len: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len].rstrip()}..."


def _build_retrieval_tree(query: str, query_id: str | None, items: list[RAGItem]) -> TreeNode:
    root_id = f"retrieval:{query_id or 'generated'}"
    children: list[TreeNode] = []

    for idx, item in enumerate(items):
        children.append(
            TreeNode(
                node_id=f"{root_id}:item:{idx}",
                node_type="CHUNK",
                parent_id=root_id,
                title=str(item.get("title", "")) or str(item.get("doc_id", "")),
                payload={
                    "doc_id": item.get("doc_id"),
                    "chunk_id": item.get("chunk_id"),
                    "source_type": item.get("source_type"),
                    "score": item.get("score"),
                    "rerank_score": item.get("rerank_score"),
                },
                children=[],
            )
        )

    return TreeNode(
        node_id=root_id,
        node_type="ROOT",
        parent_id=None,
        title=f"Retrieval: {query}",
        payload={"query": query, "item_count": len(items)},
        children=children,
    )
