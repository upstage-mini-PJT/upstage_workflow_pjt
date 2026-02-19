from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from typing import Protocol

from core.schemas.rag_contract import VectorChunk


class Embedder(Protocol):
    model_name: str
    embedding_dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class HashingEmbedder:
    """Deterministic lightweight embedder for local development and tests."""

    model_name: str = "hashing-embedder-v1"
    embedding_dim: int = 256

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")

        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(_hash_to_vector(text, self.embedding_dim))
        return vectors


def embed_chunks(chunks: list[VectorChunk], embedder: Embedder) -> list[list[float]]:
    texts = [str(chunk.get("chunk_text", "")) for chunk in chunks]
    return embedder.embed_texts(texts)


def _hash_to_vector(text: str, dim: int) -> list[float]:
    if not text.strip():
        return [0.0 for _ in range(dim)]

    vec = [0.0 for _ in range(dim)]
    for token in text.split():
        digest = sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        vec[idx] += sign * weight

    norm = sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]
