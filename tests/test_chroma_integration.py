from pathlib import Path

from core.schemas.rag_contract import IngestionDoc
from tools.data_analysis_tools.rag.chunker import ChunkingConfig, chunk_ingestion_doc
from tools.data_analysis_tools.rag.embedder import HashingEmbedder, embed_chunks
from tools.data_analysis_tools.rag.index_store import ChromaVectorIndexStore
from tools.data_analysis_tools.rag.vector_retriever import RetrieveRequest, VectorRetriever


def _build_store(tmp_path: Path, index_name: str) -> ChromaVectorIndexStore:
    return ChromaVectorIndexStore(index_name=index_name, persist_directory=str(tmp_path / "chroma"))


def test_chroma_upsert_search_and_get_by_doc_id(tmp_path: Path) -> None:
    store = _build_store(tmp_path, "test_chroma_upsert")

    doc = IngestionDoc(
        doc_id="CASELAW:CH-1",
        source_type="CASELAW",
        title="면책조항 판례",
        body="면책 조항 입증책임은 보험사에게 있다 " * 8,
        published_at="2025-01-01",
        tags=["면책", "입증책임"],
        url="https://example.com/ch-1",
    )

    chunks = chunk_ingestion_doc(
        doc,
        ChunkingConfig(chunk_size_tokens=30, chunk_overlap_tokens=10, embedding_model="hashing", embedding_dim=64),
    )
    embedder = HashingEmbedder(embedding_dim=64)
    store.upsert(chunks, embed_chunks(chunks, embedder))

    assert store.count() == len(chunks)
    assert store.count_filtered({"source_type": "CASELAW"}) == len(chunks)
    assert len(store.get_by_doc_id("CASELAW:CH-1")) == len(chunks)

    q = embedder.embed_texts(["면책조항 입증책임"])[0]
    results = store.search(q, top_k=3, filters={"source_type": "CASELAW"})
    assert len(results) >= 1
    assert results[0].chunk["doc_id"] == "CASELAW:CH-1"


def test_vector_retriever_with_chroma_store(tmp_path: Path) -> None:
    store = _build_store(tmp_path, "test_chroma_retriever")
    embedder = HashingEmbedder(embedding_dim=64)

    doc = IngestionDoc(
        doc_id="DISPUTE:CH-2",
        source_type="DISPUTE",
        title="분쟁 조정 사례",
        body="입원 필요성 분쟁에서 추가 소견서 제출로 조정 성립 " * 6,
        published_at="2025-02-01",
        tags=["분쟁", "조정"],
        url=None,
    )
    chunks = chunk_ingestion_doc(
        doc,
        ChunkingConfig(chunk_size_tokens=24, chunk_overlap_tokens=8, embedding_model=embedder.model_name, embedding_dim=embedder.embedding_dim),
    )
    store.upsert(chunks, embed_chunks(chunks, embedder))

    retriever = VectorRetriever(index_store=store, embedder=embedder)
    result = retriever.retrieve(
        RetrieveRequest(query="입원 필요성 분쟁", top_k=5, filters={"source_type": "DISPUTE"})
    )

    assert result["stats"]["candidate_count"] >= 1
    assert result["stats"]["returned_count"] >= 1
    assert result["items"][0]["source_type"] == "DISPUTE"
    assert result["items"][0]["provenance"]["index_name"] == "test_chroma_retriever"
    assert result["tree"]["node_type"] == "ROOT"
