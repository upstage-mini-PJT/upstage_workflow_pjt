from __future__ import annotations

from dataclasses import dataclass

from core.schemas.rag_contract import RAGItem, RAGRetrievalResult, TreeNode
from core.schemas.rag_conventions import clamp_score


@dataclass(frozen=True)
class RerankConfig:
    base_score_weight: float = 0.7
    lexical_overlap_weight: float = 0.3


class RerankError(ValueError):
    """Raised when reranking cannot be performed with the given input."""


def rerank_retrieval_result(
    result: RAGRetrievalResult,
    query: str | None = None,
    config: RerankConfig | None = None,
) -> RAGRetrievalResult:
    cfg = config or RerankConfig()
    _validate_config(cfg)

    actual_query = (query or result.get("query") or "").strip()
    if not actual_query:
        raise RerankError("query must not be empty for reranking")

    query_tokens = _tokenize(actual_query)
    reranked_items: list[RAGItem] = []

    for item in result.get("items", []):
        base_score = float(item.get("score", 0.0))
        lexical_score = _lexical_overlap_score(query_tokens, item)
        rerank_score = clamp_score(
            (cfg.base_score_weight * base_score)
            + (cfg.lexical_overlap_weight * lexical_score)
        )

        updated = dict(item)
        updated["rerank_score"] = rerank_score
        reranked_items.append(RAGItem(**updated))

    reranked_items.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
    return RAGRetrievalResult(
        **{
            **result,
            "items": reranked_items,
            "tree": _build_reranked_tree(result.get("tree"), reranked_items),
        }
    )


def _lexical_overlap_score(query_tokens: set[str], item: RAGItem) -> float:
    if not query_tokens:
        return 0.0

    text = f"{item.get('title', '')} {item.get('snippet', '')}"
    item_tokens = _tokenize(text)
    if not item_tokens:
        return 0.0

    overlap = query_tokens.intersection(item_tokens)
    return len(overlap) / max(1, len(query_tokens))


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in text.split() if token.strip()}


def _build_reranked_tree(original_tree: TreeNode | None, items: list[RAGItem]) -> TreeNode:
    if not original_tree:
        return TreeNode(
            node_id="retrieval:reranked",
            node_type="ROOT",
            parent_id=None,
            title="Retrieval Reranked",
            payload={"item_count": len(items)},
            children=[],
        )

    root_id = str(original_tree.get("node_id", "retrieval:reranked"))
    children: list[TreeNode] = []
    for idx, item in enumerate(items):
        children.append(
            TreeNode(
                node_id=f"{root_id}:rerank:{idx}",
                node_type="CHUNK",
                parent_id=root_id,
                title=str(item.get("title", "")) or str(item.get("doc_id", "")),
                payload={
                    "doc_id": item.get("doc_id"),
                    "chunk_id": item.get("chunk_id"),
                    "score": item.get("score"),
                    "rerank_score": item.get("rerank_score"),
                },
                children=[],
            )
        )

    return TreeNode(
        node_id=root_id,
        node_type="ROOT",
        parent_id=original_tree.get("parent_id"),
        title=str(original_tree.get("title", "Retrieval Reranked")),
        payload={**dict(original_tree.get("payload", {})), "reranked": True},
        children=children,
    )


def _validate_config(config: RerankConfig) -> None:
    if config.base_score_weight < 0:
        raise RerankError("base_score_weight must be >= 0")
    if config.lexical_overlap_weight < 0:
        raise RerankError("lexical_overlap_weight must be >= 0")
    if config.base_score_weight + config.lexical_overlap_weight == 0:
        raise RerankError("at least one weight must be > 0")
