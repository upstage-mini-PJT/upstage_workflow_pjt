from tools.data_analysis_tools.rag.normalizer import normalize_ingestion_doc
from tools.data_analysis_tools.rag.chunker import ChunkingConfig, chunk_ingestion_doc


def test_normalize_ingestion_doc_required_fields() -> None:
    doc = normalize_ingestion_doc(
        {
            "source_type": "CASELAW",
            "doc_id": "100",
            "title": "판례",
            "body": "본문 텍스트",
            "published_at": "2025-01-01",
            "tags": ["보험"],
        }
    )

    assert doc["doc_id"] == "CASELAW:100"
    assert doc["source_type"] == "CASELAW"
    assert doc["published_at"] == "2025-01-01"


def test_vector_chunk_schema_fields() -> None:
    ingestion = normalize_ingestion_doc(
        {
            "source_type": "DISPUTE",
            "doc_id": "D-1",
            "title": "분쟁사례",
            "body": "a b c d e f g h i j",
        }
    )

    chunks = chunk_ingestion_doc(
        ingestion,
        ChunkingConfig(chunk_size_tokens=4, chunk_overlap_tokens=1, embedding_model="m", embedding_dim=8),
    )

    assert chunks[0]["chunk_id"] == "DISPUTE:D-1#c0"
    assert chunks[0]["doc_id"] == "DISPUTE:D-1"
    assert chunks[0]["embedding_model"] == "m"
    assert chunks[0]["embedding_dim"] == 8
    assert chunks[0]["token_count"] == 4
