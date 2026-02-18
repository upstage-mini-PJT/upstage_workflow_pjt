from tools.data_analysis_tools.caselaw.json_source_loader import (
    load_fss_disputes,
    load_precedent_cases,
)


def test_load_precedent_cases_from_json() -> None:
    rows = load_precedent_cases()
    assert len(rows) >= 100
    assert rows[0]["source_type"] == "CASELAW"
    assert str(rows[0]["doc_id"]).startswith("CASELAW:")


def test_load_fss_disputes_from_json() -> None:
    rows = load_fss_disputes()
    assert len(rows) >= 50
    assert rows[0]["source_type"] == "DISPUTE"
    assert str(rows[0]["doc_id"]).startswith("DISPUTE:FSS-")
