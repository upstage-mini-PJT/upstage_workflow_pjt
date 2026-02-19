from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from core.schemas.rag_contract import (
    RAGItem,
    RAGRetrievalResult,
    RetrievalProvenance,
    RetrievalStats,
    RetrievalMode,
    TreeNode,
)
from core.schemas.rag_conventions import SCORE_MAX, SCORE_MIN, clamp_score
from tools.data_analysis_tools.rag.embedder import Embedder
from tools.data_analysis_tools.rag.hyde import generate_hypothetical_doc, generate_reverse_hypothesis
from tools.data_analysis_tools.rag.index_store import SearchResult, VectorIndexStore


@dataclass(frozen=True)
class RetrieveRequest:
    query: str
    top_k: int = 10
    filters: dict | None = None
    query_id: str | None = None
    retrieval_mode: RetrievalMode = "plain"
    structured_case: dict | None = None


class VectorRetriever:
    def __init__(
        self,
        index_store: VectorIndexStore,
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

        mode = request.retrieval_mode
        candidate_count = self.index_store.count_filtered(request.filters)
        variants = _build_query_variants(query=query, mode=mode, structured_case=request.structured_case or {})

        if mode in {"reverse_hyde", "hybrid_hyde"}:
            plain_results = self._search_variant(
                query_text=query,
                query_variant="plain",
                retrieval_method="vector_plain",
                top_k=request.top_k,
                filters=request.filters,
                retrieved_at=retrieved_at,
            )
            seed_items = [item for item, _ in plain_results]
            reverse_text = generate_reverse_hypothesis(query, seed_items)
            variants.append(("reverse_hyde", reverse_text, "vector_reverse_hyde"))
            if mode == "reverse_hyde":
                variants = [("reverse_hyde", reverse_text, "vector_reverse_hyde")]

        all_results: list[tuple[RAGItem, float]] = []
        for query_variant, query_text, retrieval_method in variants:
            all_results.extend(
                self._search_variant(
                    query_text=query_text,
                    query_variant=query_variant,
                    retrieval_method=retrieval_method,
                    top_k=request.top_k,
                    filters=request.filters,
                    retrieved_at=retrieved_at,
                )
            )

        merged_items = _merge_ranked_items(all_results, top_k=request.top_k)

        latency_ms = int((perf_counter() - started_at) * 1000)
        stats = RetrievalStats(
            candidate_count=candidate_count,
            returned_count=len(merged_items),
            latency_ms=latency_ms,
        )

        tree = _build_retrieval_tree(query=query, query_id=request.query_id, items=merged_items)

        return RAGRetrievalResult(
            query_id=request.query_id or str(uuid4()),
            query=query,
            retrieval_mode=mode,
            query_variants=[v[0] for v in variants],
            filters=request.filters or {},
            items=merged_items,
            tree=tree,
            stats=stats,
        )

    def _search_variant(
        self,
        query_text: str,
        query_variant: str,
        retrieval_method: str,
        top_k: int,
        filters: dict | None,
        retrieved_at: str,
    ) -> list[tuple[RAGItem, float]]:
        query_vector = self.embedder.embed_texts([query_text])[0]
        raw_results = self.index_store.search(query_vector, top_k=top_k, filters=filters)
        ranked: list[tuple[RAGItem, float]] = []

        for result in raw_results:
            item = _search_result_to_item(
                result=result,
                index_name=self.index_store.index_name,
                retrieved_at=retrieved_at,
                retrieval_method=retrieval_method,
                query_variant=query_variant,
            )
            ranked.append((item, float(item.get("score", 0.0))))
        return ranked


def _build_query_variants(
    query: str,
    mode: RetrievalMode,
    structured_case: dict,
) -> list[tuple[str, str, str]]:
    variants: list[tuple[str, str, str]] = []

    if mode == "plain":
        return [("plain", query, "vector_plain")]
    if mode == "hyde":
        hyde_text = generate_hypothetical_doc(query=query, structured_case=structured_case)
        return [("hyde", hyde_text, "vector_hyde")]
    if mode == "reverse_hyde":
        return []
    if mode == "hybrid_hyde":
        hyde_text = generate_hypothetical_doc(query=query, structured_case=structured_case)
        variants.append(("plain", query, "vector_plain"))
        variants.append(("hyde", hyde_text, "vector_hyde"))
        return variants

    return [("plain", query, "vector_plain")]


def _search_result_to_item(
    result: SearchResult,
    index_name: str,
    retrieved_at: str,
    retrieval_method: str,
    query_variant: str,
) -> RAGItem:
    chunk = result.chunk
    score = _normalize_score(result.score)
    provenance = RetrievalProvenance(
        index_name=index_name,
        retrieved_at=retrieved_at,
        retrieval_method=retrieval_method,
        query_variant=query_variant,
    )
    return RAGItem(
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


def _merge_ranked_items(all_results: list[tuple[RAGItem, float]], top_k: int) -> list[RAGItem]:
    merged: dict[str, RAGItem] = {}
    duplicate_count: dict[str, int] = {}

    for item, _ in all_results:
        key = f"{item.get('doc_id','')}::{item.get('chunk_id','')}"
        duplicate_count[key] = duplicate_count.get(key, 0) + 1

        existing = merged.get(key)
        if existing is None or float(item.get("score", 0.0)) > float(existing.get("score", 0.0)):
            merged[key] = item

    merged_items = list(merged.values())
    for item in merged_items:
        key = f"{item.get('doc_id','')}::{item.get('chunk_id','')}"
        dup = duplicate_count.get(key, 1)
        base = float(item.get("score", 0.0))
        fused = clamp_score(base + min(0.06, (dup - 1) * 0.03))
        item["score"] = fused
        item["rerank_score"] = fused

    merged_items.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
    return merged_items[:top_k]


def _normalize_score(raw_cosine: float) -> float:
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
                    "query_variant": item.get("provenance", {}).get("query_variant"),
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
