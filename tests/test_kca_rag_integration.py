from core.schemas.rag_contract import IngestionDoc
from tools.data_analysis_tools.caselaw.kca_loader import load_normalized_kca_disputes
from tools.data_analysis_tools.rag.chunker import chunk_ingestion_docs
from tools.data_analysis_tools.rag.embedder import HashingEmbedder, embed_chunks
from tools.data_analysis_tools.rag.index_store import InMemoryVectorIndexStore
from tools.data_analysis_tools.rag.reranker import rerank_retrieval_result
from tools.data_analysis_tools.rag.vector_retriever import RetrieveRequest, VectorRetriever


def test_kca_data_flows_through_rag_index() -> None:
    kca_rows = load_normalized_kca_disputes(limit=5)
    docs: list[IngestionDoc] = []
    for row in kca_rows:
        docs.append(
            IngestionDoc(
                doc_id=str(row.get("doc_id", "")),
                source_type="DISPUTE",
                title=str(row.get("title", "")),
                body=f"{row.get('summary', '')} {row.get('holding', '')}".strip(),
                published_at=str(row.get("published_at", "1970-01-01")),
                tags=[str(x) for x in row.get("keywords", [])],
                url=row.get("url"),
                meta={"source": row.get("source", "kca_dispute")},
            )
        )

    embedder = HashingEmbedder(embedding_dim=64)
    chunks = chunk_ingestion_docs(docs)
    vectors = embed_chunks(chunks, embedder)
    store = InMemoryVectorIndexStore(index_name="kca_rag_test")
    store.upsert(chunks, vectors)

    retriever = VectorRetriever(index_store=store, embedder=embedder)
    rag = rerank_retrieval_result(
        retriever.retrieve(
            RetrieveRequest(
                query="보험금 지급 거절 분쟁",
                top_k=5,
                filters={"source_type": "DISPUTE"},
            )
        )
    )

    assert rag["stats"]["candidate_count"] >= 1
    assert rag["stats"]["returned_count"] >= 1
    assert any(item.get("source_type") == "DISPUTE" for item in rag["items"])
