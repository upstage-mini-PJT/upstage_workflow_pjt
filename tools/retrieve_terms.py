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


def list_policy_dates(vectordb: Chroma | None = None) -> list[str]:
    """
    Return sorted unique policy_date values present in the vector DB.
    """
    target_vectordb = vectordb or load_vectordb()
    count = _collection_count(target_vectordb)
    if count <= 0:
        return []

    try:
        data = target_vectordb.get(include=["metadatas"], limit=count)
    except Exception:
        return []

    metadatas = data.get("metadatas", []) or []
    dates = {
        date
        for date in (str(md.get("policy_date", "")).strip() for md in metadatas)
        if len(date) == 8 and date.isdigit()
    }
    return sorted(dates)


def retrieve_terms_text(
    query: str,
    policy_date: str,
    k: int = 3,
    *,
    vectordb: Chroma | None = None,
) -> str:
    candidates = retrieve_terms_candidates(
        query=query,
        policy_date=policy_date,
        k=k,
        vectordb=vectordb,
    )
    return format_candidates_to_terms(candidates, top_n=k)


def retrieve_terms_candidates(
    query: str,
    policy_date: str,
    k: int = 5,
    *,
    vectordb: Chroma | None = None,
) -> list[dict[str, object]]:
    cleaned_query = str(query).strip()
    cleaned_policy_date = str(policy_date).strip()
    if not cleaned_query or not cleaned_policy_date:
        return []

    try:
        target_vectordb = ensure_vectordb_ready(vectordb)
    except Exception:
        return []

    try:
        relevant_chunks = target_vectordb.similarity_search(
            cleaned_query,
            k=k,
            filter={"policy_date": cleaned_policy_date},
        )
    except Exception:
        return []

    if not relevant_chunks:
        return []

    # Keep best rank per unique article.
    pair_rank: dict[tuple[str | None, str], int] = {}
    for rank, chunk in enumerate(relevant_chunks, start=1):
        article = chunk.metadata.get("Article")
        section = chunk.metadata.get("Document Section")
        if not article:
            continue
        key = (section, article)
        if key not in pair_rank:
            pair_rank[key] = rank

    ordered_pairs = sorted(pair_rank.items(), key=lambda item: item[1])
    candidates: list[dict[str, object]] = []
    for (section, article), rank in ordered_pairs:
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
        section_title = first_metadata.get("Document Section", "N/A")
        article_title = first_metadata.get("Article", "N/A")
        source_id = f"{cleaned_policy_date}:{section_title}:{article_title}"
        title = (
            f"[Section: {section_title} | "
            f"Article: {article_title} | "
            f"Policy: {first_metadata.get('policy_date', 'N/A')}]"
        )
        full_text = f"{title}\n{combined_text}"
        snippet = " ".join(combined_text.split())[:220]
        candidates.append(
            {
                "source_id": source_id,
                "title": title,
                "snippet": snippet,
                "full_text": full_text,
                "query": cleaned_query,
                "rank": rank,
                "score": 0.0,
                "matched_keywords": [],
            }
        )
    return candidates


def merge_candidates_rrf(
    candidates_by_query: list[list[dict[str, object]]],
    must_keywords_by_query: list[list[str]] | None = None,
    *,
    rrf_k: int = 60,
    keyword_bonus: float = 0.2,
    keyword_bonus_cap: float = 0.8,
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    all_keywords = must_keywords_by_query or []

    for query_idx, candidates in enumerate(candidates_by_query):
        keywords = all_keywords[query_idx] if query_idx < len(all_keywords) else []
        normalized_keywords = [str(item).strip() for item in keywords if str(item).strip()]

        for rank, item in enumerate(candidates, start=1):
            source_id = str(item.get("source_id", "")).strip()
            if not source_id:
                continue

            entry = merged.get(source_id)
            if not entry:
                entry = dict(item)
                entry["score"] = 0.0
                entry["_matched_keyword_set"] = set()
                entry["_best_rank"] = int(item.get("rank", rank) or rank)
                merged[source_id] = entry

            base_score = 1.0 / float(rrf_k + rank)
            haystack = f"{item.get('title', '')}\n{item.get('full_text', '')}".lower()
            matched = [kw for kw in normalized_keywords if kw.lower() in haystack]
            bonus = min(keyword_bonus * len(matched), keyword_bonus_cap)
            entry["score"] = float(entry["score"]) + base_score + bonus
            entry["_best_rank"] = min(int(entry.get("_best_rank", rank)), rank)
            matched_set = entry.get("_matched_keyword_set")
            if isinstance(matched_set, set):
                matched_set.update(matched)

    ranked = list(merged.values())
    for item in ranked:
        matched_set = item.get("_matched_keyword_set")
        item["matched_keywords"] = sorted(matched_set) if isinstance(matched_set, set) else []
        item.pop("_matched_keyword_set", None)
        item.pop("_best_rank", None)

    ranked.sort(
        key=lambda x: (
            -float(x.get("score", 0.0)),
            int(x.get("rank", 9999) or 9999),
            str(x.get("source_id", "")),
        )
    )
    return ranked


def format_candidates_to_terms(candidates: list[dict[str, object]], *, top_n: int = 5) -> str:
    selected = candidates[:top_n] if top_n > 0 else candidates
    full_terms = [str(item.get("full_text", "")).strip() for item in selected if str(item.get("full_text", "")).strip()]
    return "\n\n".join(full_terms)

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
