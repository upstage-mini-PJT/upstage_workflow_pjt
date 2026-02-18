from tools.data_analysis_tools.caselaw.client import retrieve_cases
import tools.data_analysis_tools.caselaw.client as client_module
from tools.data_analysis_tools.caselaw.types import RetrievalQuery


def _clear_caches() -> None:
    client_module._load_kca_disputes.cache_clear()
    client_module._load_fss_pool.cache_clear()
    client_module._load_precedent_pool.cache_clear()


def test_retrieve_cases_includes_kca_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("RAG_INCLUDE_KCA", "true")
    monkeypatch.setenv("RAG_INCLUDE_FSS", "true")
    monkeypatch.setenv(
        "RAG_KCA_DATA_PATH",
        "data/data_analysis_data/converted_cases/kca_finance_insurance_cases_filtered.json",
    )
    _clear_caches()

    queries = [RetrievalQuery(source="DISPUTE", query="보험금 분쟁", top_k=10)]
    rows = retrieve_cases(queries, limit=200)
    assert any(str(r.get("doc_id", "")).startswith("DISPUTE:KCA-") for r in rows)


def test_retrieve_cases_excludes_kca_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("RAG_INCLUDE_KCA", "false")
    monkeypatch.setenv("RAG_INCLUDE_FSS", "true")
    _clear_caches()

    queries = [RetrievalQuery(source="DISPUTE", query="보험금 분쟁", top_k=10)]
    rows = retrieve_cases(queries, limit=200)
    assert all(not str(r.get("doc_id", "")).startswith("DISPUTE:KCA-") for r in rows)


def test_retrieve_cases_reads_precedent_json_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("RAG_INCLUDE_PRECEDENTS", "true")
    _clear_caches()

    queries = [RetrievalQuery(source="CASELAW", query="보험금 판례", top_k=10)]
    rows = retrieve_cases(queries, limit=200)
    assert any(str(r.get("doc_id", "")).startswith("CASELAW:") for r in rows)
    assert any(str(r.get("source", "")) == "law_precedents" for r in rows)


def test_retrieve_cases_reads_fss_json_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("RAG_INCLUDE_FSS", "true")
    monkeypatch.setenv("RAG_INCLUDE_KCA", "false")
    _clear_caches()

    queries = [RetrievalQuery(source="DISPUTE", query="보험금 분쟁", top_k=10)]
    rows = retrieve_cases(queries, limit=200)
    assert any(str(r.get("doc_id", "")).startswith("DISPUTE:FSS-") for r in rows)
    assert any(str(r.get("source", "")) == "fss_dispute" for r in rows)
