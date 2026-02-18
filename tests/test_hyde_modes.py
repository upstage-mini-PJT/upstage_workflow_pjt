from __future__ import annotations

import pytest

from agents.data_analysis_agent.pipeline import run_pipeline
from core.schemas.rag_contract import IngestionDoc
from evaluations.data_analysis.user_scenarios import USER_SCENARIOS
from tools.data_analysis_tools.rag.chunker import ChunkingConfig, chunk_ingestion_doc
from tools.data_analysis_tools.rag.embedder import HashingEmbedder, embed_chunks
from tools.data_analysis_tools.rag.hyde import generate_hypothetical_doc
from tools.data_analysis_tools.rag.index_store import InMemoryVectorIndexStore
from tools.data_analysis_tools.rag.vector_retriever import RetrieveRequest, VectorRetriever


def _build_retriever() -> VectorRetriever:
    docs = [
        IngestionDoc(
            doc_id="CASELAW:HY-1",
            source_type="CASELAW",
            title="입증책임 판례",
            body="보험금 부지급 사안에서 면책 사유 입증책임은 보험사에 있다.",
            published_at="2025-01-01",
            tags=["면책", "입증책임"],
            url=None,
        ),
        IngestionDoc(
            doc_id="DISPUTE:HY-2",
            source_type="DISPUTE",
            title="분쟁 조정 사례",
            body="입원 필요성 분쟁에서 추가 소견서 제출 후 조정 성립.",
            published_at="2025-02-01",
            tags=["입원", "소견서"],
            url=None,
        ),
    ]
    embedder = HashingEmbedder(embedding_dim=64)
    store = InMemoryVectorIndexStore(index_name="hyde_test_idx")

    all_chunks = []
    for doc in docs:
        all_chunks.extend(
            chunk_ingestion_doc(
                doc,
                ChunkingConfig(
                    chunk_size_tokens=30,
                    chunk_overlap_tokens=10,
                    embedding_model=embedder.model_name,
                    embedding_dim=embedder.embedding_dim,
                ),
            )
        )
    store.upsert(all_chunks, embed_chunks(all_chunks, embedder))
    return VectorRetriever(index_store=store, embedder=embedder)


def test_hyde_masks_sensitive_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_HYDE", "true")
    out = generate_hypothetical_doc(
        query="연락처 010-1234-5678, 이메일 test@example.com, 주민번호 900101-1234567",
        structured_case={},
    )
    assert "[PHONE]" in out
    assert "[EMAIL]" in out
    assert "[RRN]" in out


def test_hyde_disabled_falls_back_to_plain_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_HYDE", "false")
    out = generate_hypothetical_doc(query="면책조항 해석", structured_case={})
    assert out == "면책조항 해석"


@pytest.mark.parametrize(
    ("mode", "expected_variants"),
    [
        ("plain", {"plain"}),
        ("hyde", {"hyde"}),
        ("reverse_hyde", {"reverse_hyde"}),
        ("hybrid_hyde", {"plain", "hyde", "reverse_hyde"}),
    ],
)
def test_vector_retriever_modes_return_expected_variants(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_variants: set[str],
) -> None:
    monkeypatch.setenv("RAG_ENABLE_HYDE", "true")
    retriever = _build_retriever()
    result = retriever.retrieve(
        RetrieveRequest(
            query="면책 입원 필요성 분쟁",
            top_k=4,
            retrieval_mode=mode,  # type: ignore[arg-type]
            structured_case={"denial_reasons": ["면책조항 적용"]},
        )
    )

    assert result["retrieval_mode"] == mode
    assert expected_variants.issubset(set(result["query_variants"]))
    assert result["stats"]["returned_count"] >= 1
    assert all(0.0 <= float(item["score"]) <= 1.0 for item in result["items"])


def test_pipeline_uses_env_retrieval_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "hyde")
    case = USER_SCENARIOS[0]["structured_case"]
    result = run_pipeline(case)  # type: ignore[arg-type]
    assert result["rag_result"]["retrieval_mode"] == "hyde"
