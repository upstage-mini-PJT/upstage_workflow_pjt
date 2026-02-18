from tools.data_analysis_tools.caselaw.kca_loader import (
    DEFAULT_KCA_PATH,
    filter_insurance_cases,
    load_kca_raw_cases,
    load_normalized_kca_disputes,
    normalize_kca_case,
)


def test_load_kca_raw_cases_exists() -> None:
    rows = load_kca_raw_cases(DEFAULT_KCA_PATH)
    assert len(rows) >= 1
    assert "제목" in rows[0]


def test_filter_insurance_cases() -> None:
    rows = [
        {"제목": "보험금 분쟁 사례", "상세내용": {"사건개요": "보험금 지급 거절"}},
        {"제목": "유사투자자문 계약", "상세내용": {"사건개요": "투자 손실"}},
    ]
    filtered = filter_insurance_cases(rows)
    assert len(filtered) == 1
    assert "보험" in filtered[0]["제목"]


def test_normalize_kca_case_to_dispute_shape() -> None:
    normalized = normalize_kca_case(
        {
            "번호": "999",
            "제목": "실손보험 입원비 분쟁",
            "수정일": "2025-10-01",
            "상세내용": {"사건개요": "보험금 청구", "판단": "약관 해석상 일부 인정"},
        }
    )
    assert normalized["doc_id"].startswith("DISPUTE:KCA-")
    assert normalized["source_type"] == "DISPUTE"
    assert normalized["source"] == "kca_dispute"
    assert normalized["title"] != ""
    assert normalized["summary"] != ""


def test_load_normalized_kca_disputes() -> None:
    rows = load_normalized_kca_disputes()
    assert len(rows) >= 1
    assert all(r["source_type"] == "DISPUTE" for r in rows[:5])
