from core.schemas.rag_contract import IngestionDoc
from tools.data_analysis_tools.rag.chunker import chunk_ingestion_docs
from tools.data_analysis_tools.rag.comparative_scoring import (
    ComparativeScoringInput,
    compute_comparative_scoring,
)
from tools.data_analysis_tools.rag.embedder import HashingEmbedder, embed_chunks
from tools.data_analysis_tools.rag.index_store import InMemoryVectorIndexStore
from tools.data_analysis_tools.rag.reranker import rerank_retrieval_result
from tools.data_analysis_tools.rag.search_client import MockWebSearchProvider, search_web_docs
from tools.data_analysis_tools.rag.vector_retriever import RetrieveRequest, VectorRetriever


def _run_e2e(docs: list[IngestionDoc], query: str) -> tuple[int, int]:
    embedder = HashingEmbedder(embedding_dim=64)
    chunks = chunk_ingestion_docs(docs)
    vectors = embed_chunks(chunks, embedder)

    store = InMemoryVectorIndexStore(index_name="e2e")
    store.upsert(chunks, vectors)

    retriever = VectorRetriever(store, embedder)
    rag = rerank_retrieval_result(retriever.retrieve(RetrieveRequest(query=query, top_k=5)))
    scoring = compute_comparative_scoring(
        ComparativeScoringInput(
            rag_result=rag,
            case_adjustment=3,
            rationale="상위 문서 관련성 확인",
            cited_case_ids=[rag["items"][0]["doc_id"]] if rag["items"] else [],
        )
    )
    return len(rag["items"]), int(scoring["total_score"])


def test_e2e_caselaw_only() -> None:
    docs = [
        IngestionDoc(
            doc_id="CASELAW:1",
            source_type="CASELAW",
            title="판례",
            body="보험금 부지급 면책 입증책임",
            published_at="2025-01-01",
            tags=["보험"],
            url=None,
        )
    ]
    item_count, total_score = _run_e2e(docs, query="면책 입증책임")
    assert item_count >= 1
    assert 0 <= total_score <= 100


def test_e2e_web_only() -> None:
    provider = MockWebSearchProvider(
        results=[
            {
                "id": "W1",
                "title": "금융위 안내",
                "content": "보험 분쟁조정 절차",
                "url": "https://fsc.go.kr/notice/1",
                "published_at": "2025-02-01",
            }
        ]
    )
    docs = search_web_docs(provider, query="보험 분쟁", top_k=5)
    item_count, total_score = _run_e2e(docs, query="분쟁조정")
    assert item_count >= 1
    assert 0 <= total_score <= 100


def test_e2e_mixed_sources() -> None:
    provider = MockWebSearchProvider(
        results=[
            {
                "id": "W2",
                "title": "협회 자료",
                "content": "보험금 심사 분쟁 사례",
                "url": "https://example.org/rag",
                "published_at": "2025-02-02",
            }
        ]
    )
    docs = [
        IngestionDoc(
            doc_id="DISPUTE:D1",
            source_type="DISPUTE",
            title="분쟁사례",
            body="입원 필요성 관련 조정 성립 사례",
            published_at="2024-12-01",
            tags=["분쟁"],
            url=None,
        )
    ] + search_web_docs(provider, query="보험 사례", top_k=5)

    item_count, total_score = _run_e2e(docs, query="입원 필요성 분쟁")
    assert item_count >= 1
    assert 0 <= total_score <= 100
