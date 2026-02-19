from __future__ import annotations

from dataclasses import dataclass

from core.schemas.rag_contract import IngestionDoc, VectorChunk, VectorChunkMetadata
from core.schemas.rag_conventions import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    make_chunk_id,
)


class ChunkingError(ValueError):
    """Raised when chunking configuration or input is invalid."""


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    embedding_model: str = "pending"
    embedding_dim: int = 0


def chunk_ingestion_doc(
    doc: IngestionDoc,
    config: ChunkingConfig | None = None,
) -> list[VectorChunk]:
    cfg = config or ChunkingConfig()
    _validate_config(cfg)

    doc_id = str(doc.get("doc_id", "")).strip()
    source_type = doc.get("source_type")
    body = str(doc.get("body", "")).strip()

    if not doc_id:
        raise ChunkingError("doc_id must not be empty")
    if source_type is None:
        raise ChunkingError("source_type must not be empty")
    if not body:
        return []

    tokens = body.split()
    step = cfg.chunk_size_tokens - cfg.chunk_overlap_tokens
    chunks: list[VectorChunk] = []

    for chunk_index, start in enumerate(range(0, len(tokens), step)):
        end = min(start + cfg.chunk_size_tokens, len(tokens))
        window_tokens = tokens[start:end]
        if not window_tokens:
            continue

        chunk_text = " ".join(window_tokens)
        metadata = _build_metadata(doc)
        chunks.append(
            VectorChunk(
                chunk_id=make_chunk_id(doc_id, chunk_index),
                doc_id=doc_id,
                source_type=source_type,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                embedding_model=cfg.embedding_model,
                embedding_dim=cfg.embedding_dim,
                token_count=len(window_tokens),
                metadata=metadata,
            )
        )

        if end >= len(tokens):
            break

    return chunks


def chunk_ingestion_docs(
    docs: list[IngestionDoc],
    config: ChunkingConfig | None = None,
) -> list[VectorChunk]:
    out: list[VectorChunk] = []
    for doc in docs:
        out.extend(chunk_ingestion_doc(doc, config=config))
    return out


def _build_metadata(doc: IngestionDoc) -> VectorChunkMetadata:
    return VectorChunkMetadata(
        title=str(doc.get("title", "")),
        published_at=doc.get("published_at"),
        url=doc.get("url"),
        tags=list(doc.get("tags", [])),
    )


def _validate_config(config: ChunkingConfig) -> None:
    if config.chunk_size_tokens <= 0:
        raise ChunkingError("chunk_size_tokens must be > 0")
    if config.chunk_overlap_tokens < 0:
        raise ChunkingError("chunk_overlap_tokens must be >= 0")
    if config.chunk_overlap_tokens >= config.chunk_size_tokens:
        raise ChunkingError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
    if config.embedding_dim < 0:
        raise ChunkingError("embedding_dim must be >= 0")
