from core.schemas.rag_contract import IngestionDoc
from tools.data_analysis_tools.rag.chunker import ChunkingConfig, chunk_ingestion_doc
from tools.data_analysis_tools.rag.embedder import HashingEmbedder, embed_chunks
from tools.data_analysis_tools.rag.index_store import InMemoryVectorIndexStore
from tools.data_analysis_tools.rag.vector_retriever import RetrieveRequest, VectorRetriever


def test_vector_retriever_returns_contract_shape() -> None:
    doc = IngestionDoc(
        doc_id="CASELAW:1",
        source_type="CASELAW",
        title="면책조항",
        body="면책 조항 입증책임 보험사 책임",
        published_at="2025-01-01",
        tags=["면책"],
        url=None,
    )

    chunks = chunk_ingestion_doc(doc, ChunkingConfig(chunk_size_tokens=10, chunk_overlap_tokens=2, embedding_dim=32))
    embedder = HashingEmbedder(embedding_dim=32)
    store = InMemoryVectorIndexStore(index_name="test_idx")
    store.upsert(chunks, embed_chunks(chunks, embedder))

    retriever = VectorRetriever(store, embedder)
    result = retriever.retrieve(RetrieveRequest(query="면책 입증책임", top_k=3))

    assert result["query"] == "면책 입증책임"
    assert result["stats"]["candidate_count"] >= 1
    assert result["stats"]["returned_count"] >= 1
    assert result["items"][0]["provenance"]["index_name"] == "test_idx"
    assert 0.0 <= result["items"][0]["score"] <= 1.0
    assert result["tree"]["node_type"] == "ROOT"
