"""
retrieve_terms.py

Small-to-big retrieval tool for insurance policy terms.
Retrieves full articles from the vector database based on relevant chunks.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "solar-embedding-1-large")
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", str((BASE_DIR / "data/chroma").resolve()))).resolve()

_CACHED_VECTORDB: Chroma | None = None
_CACHED_VECTOR_PATH: str | None = None

# ============================================================================
# Load Vector Database
# ============================================================================

def _build_embeddings() -> UpstageEmbeddings:
    if not UPSTAGE_API_KEY:
        raise RuntimeError("UPSTAGE_API_KEY is required for vector retrieval.")
    return UpstageEmbeddings(
        api_key=UPSTAGE_API_KEY,
        model=EMBEDDING_MODEL,
    )


def load_vectordb(*, persist_directory: str | None = None) -> Chroma:
    """Load the persistent vector database."""
    global _CACHED_VECTORDB, _CACHED_VECTOR_PATH
    vector_path = str(Path(persist_directory).expanduser().resolve()) if persist_directory else str(VECTOR_DIR)
    if _CACHED_VECTORDB is not None and _CACHED_VECTOR_PATH == vector_path:
        return _CACHED_VECTORDB

    _CACHED_VECTORDB = Chroma(
        persist_directory=vector_path,
        embedding_function=_build_embeddings(),
    )
    _CACHED_VECTOR_PATH = vector_path
    return _CACHED_VECTORDB


def _collection_count(vectordb: Chroma) -> int:
    try:
        return int(vectordb._collection.count())
    except Exception:
        return 0


def ensure_vectordb_ready(vectordb: Chroma | None = None) -> Chroma:
    """Cold start support: index policy docs if vector DB is empty."""
    target_vectordb = vectordb or load_vectordb()
    if _collection_count(target_vectordb) > 0:
        return target_vectordb

    from agents.onboarding_agent.chunker_embedder import build_policy_index

    return build_policy_index(vectordb=target_vectordb, skip_if_populated=False)


def vectordb_document_count(vectordb: Chroma | None = None) -> int:
    target_vectordb = vectordb or load_vectordb()
    return _collection_count(target_vectordb)


def retrieve_terms_text(
    query: str,
    policy_date: str,
    k: int = 3,
    *,
    vectordb: Chroma | None = None,
) -> str:
    """
    Retrieve full insurance policy articles relevant to a query.
    Returns merged article text blocks.
    """
    cleaned_query = str(query).strip()
    cleaned_policy_date = str(policy_date).strip()
    if not cleaned_query or not cleaned_policy_date:
        return ""

    try:
        target_vectordb = ensure_vectordb_ready(vectordb)
    except Exception:
        return ""

    try:
        # Step 1: Find top k most relevant chunks
        relevant_chunks = target_vectordb.similarity_search(
            cleaned_query,
            k=k,
            filter={"policy_date": cleaned_policy_date},
        )
    except Exception:
        return ""

    if not relevant_chunks:
        return ""

    # Step 2: Extract unique (section, article) pairs
    articles_to_fetch: set[tuple[str | None, str]] = set()
    for chunk in relevant_chunks:
        article = chunk.metadata.get("Article")
        section = chunk.metadata.get("Document Section")
        if article:
            articles_to_fetch.add((section, article))

    # Step 3: Retrieve and combine all chunks per article
    full_terms_docs: list[str] = []
    for section, article in articles_to_fetch:
        if section is not None:
            where_filter = {
                "$and": [
                    {"policy_date": cleaned_policy_date},
                    {"Document Section": section},
                    {"Article": article},
                ]
            }
        else:
            where_filter = {
                "$and": [
                    {"policy_date": cleaned_policy_date},
                    {"Article": article},
                ]
            }

        try:
            article_chunks = target_vectordb.get(where=where_filter)
        except Exception:
            continue

        metadatas = article_chunks.get("metadatas", [])
        documents = article_chunks.get("documents", [])
        paired = list(zip(metadatas, documents))
        if not paired:
            continue

        sorted_pairs = sorted(paired, key=lambda x: x[0].get("chunk_id", ""))
        combined_text = "\n".join([text for _, text in sorted_pairs])

        first_metadata = sorted_pairs[0][0]
        formatted = (
            f"[Section: {first_metadata.get('Document Section', 'N/A')} | "
            f"Article: {first_metadata.get('Article', 'N/A')} | "
            f"Policy: {first_metadata.get('policy_date', 'N/A')}]\n"
            f"{combined_text}"
        )
        full_terms_docs.append(formatted)

    return "\n\n".join(full_terms_docs)

# ============================================================================
# Retrieval Tool
# ============================================================================

class RetrieveTermsSchema(BaseModel):
    query: str = Field(
        ...,
        description="The question or denial statement text to search for in the policy."
    )
    policy_date: str = Field(
        ...,
        description="Policy version date in YYYYMMDD format (e.g., '20200101'). Must match the client's contract date."
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of articles to retrieve. Default is 3."
    )

@tool(args_schema=RetrieveTermsSchema)
def retrieve_terms(query: str, policy_date: str, k: int = 3) -> str:
    """
    Retrieve full insurance policy articles relevant to a query.
    Use this tool when you need to look up what the policy says about a
    specific topic such as coverage exclusions, deductibles, or claim procedures.
    Returns the full text of the most relevant articles including section and article metadata.
    """
    return retrieve_terms_text(query=query, policy_date=policy_date, k=k)
