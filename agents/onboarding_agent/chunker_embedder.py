"""
chunker_embedder.py

Chunks and embeds all insurance policy markdown files from ./data/policies/
into a persistent ChromaDB vector database.

File naming convention: YYYYMMDD.md (e.g., 20200101.md)
Policy date and title are extracted from the filename.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "solar-embedding-1-large")
BASE_DIR = Path(__file__).resolve().parent
POLICIES_DIR = Path(os.getenv("POLICIES_DIR", str((BASE_DIR / "../../data/policies").resolve()))).resolve()
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", str((BASE_DIR / "../../data/chroma").resolve()))).resolve()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 30

# Policy titles mapped by date — add new versions here as needed
POLICY_TITLES = {
    "20200101": "노후실손의료비보장보험(갱신형)(Hi2001)",
    "20220101": "노후실손의료비보장보험(갱신형)(Hi2201)",
    "20240101": "노후실손의료비보장보험(갱신형)(Hi2401)",
    "20260101": "노후실손의료비보장보험(갱신형)(Hi2601)",
}

# ============================================================================
# Chunking
# ============================================================================

def chunk_policy(file_path: str, policy_date: str, policy_title: str) -> list:
    """
    Load and chunk a single policy markdown file.

    Args:
        file_path: Full path to the markdown file
        policy_date: Extracted from filename e.g. "20200101"
        policy_title: Human-readable policy name

    Returns:
        List of Document objects with metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_document = f.read()

    headers_to_split_on = [
        ("#", "Document"),
        ("##", "Document Section"),
        ("###", "Article"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on,
        strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(md_header_splits)

    # Forward-fill missing Document Section values
    current_section = None
    for doc in splits:
        if doc.metadata.get('Document Section'):
            current_section = doc.metadata['Document Section']
        else:
            doc.metadata['Document Section'] = current_section

    # Add policy metadata and chunk_id
    for i, doc in enumerate(splits):
        doc.metadata['policy_date'] = policy_date
        doc.metadata['policy_title'] = policy_title
        doc.metadata['chunk_id'] = f"chunk_{i:04d}"

    print(f"  Chunked {len(splits)} chunks for {policy_date}")
    return splits

# ============================================================================
# Duplication Check — run after embedding, checks reconstructed articles
# ============================================================================

def check_for_duplicate_articles(vectordb, policy_date: str):
    from collections import defaultdict

    data = vectordb.get(where={"policy_date": policy_date})
    pair_to_chunk_ids = defaultdict(list)

    for metadata in data['metadatas']:
        section = metadata.get('Document Section')
        article = metadata.get('Article')
        chunk_id = metadata.get('chunk_id', '')
        if section and article and chunk_id:
            pair_to_chunk_ids[(section, article)].append(chunk_id)

    duplicates_found = False
    for (section, article), chunk_ids in sorted(pair_to_chunk_ids.items()):
        indices = sorted(int(c.split('_')[1]) for c in chunk_ids)
        gaps = [
            (indices[i], indices[i+1])
            for i in range(len(indices) - 1)
            if indices[i+1] - indices[i] > 1
        ]
        if gaps:
            duplicates_found = True
            print(f"  ⚠️  NON-CONTIGUOUS: '{article}' | '{section}'")
            print(f"     chunk indices: {indices}")
            print(f"     gaps at: {gaps}")

    if not duplicates_found:
        print(f"  ✅ No duplicate articles detected for {policy_date}.")


def load_policy_vectordb() -> Chroma:
    embeddings_model = UpstageEmbeddings(
        api_key=UPSTAGE_API_KEY,
        model=EMBEDDING_MODEL,
    )
    return Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings_model,
    )


def _collection_count(vectordb: Chroma) -> int:
    try:
        return int(vectordb._collection.count())
    except Exception:
        return 0


def build_policy_index(vectordb: Chroma | None = None, *, skip_if_populated: bool = True) -> Chroma:
    """
    Build vector index from policy markdown files.

    If `skip_if_populated` is True and existing documents are present, indexing
    is skipped to avoid duplicate inserts.
    """
    target_vectordb = vectordb or load_policy_vectordb()

    existing_count = _collection_count(target_vectordb)
    if skip_if_populated and existing_count > 0:
        print(f"Vector DB already populated ({existing_count} docs). Skip indexing.")
        return target_vectordb

    policy_files = sorted([f for f in os.listdir(POLICIES_DIR) if f.endswith(".md")])
    if not policy_files:
        raise FileNotFoundError(f"No markdown files found in {POLICIES_DIR}")

    for filename in policy_files:
        policy_date = filename.replace(".md", "")
        policy_title = POLICY_TITLES.get(policy_date, f"보험약관_{policy_date}")

        print(f"\nProcessing {filename}...")
        file_path = os.path.join(POLICIES_DIR, filename)

        splits = chunk_policy(file_path, policy_date, policy_title)
        target_vectordb.add_documents(splits)
        print(f"  Embedded {len(splits)} chunks for {policy_date}")
        check_for_duplicate_articles(target_vectordb, policy_date)

    print(f"\nDone. Total docs in DB: {_collection_count(target_vectordb)}")
    return target_vectordb

# ============================================================================
# Main
# ============================================================================

def main():
    build_policy_index(skip_if_populated=True)

if __name__ == "__main__":
    main()
