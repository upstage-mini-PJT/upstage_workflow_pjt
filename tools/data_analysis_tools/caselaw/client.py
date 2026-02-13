from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tools.data_analysis_tools.caselaw.types import RetrievalQuery


MOCK_CASELAW: list[dict[str, Any]] = [
    {
        "doc_id": "CASE-001",
        "title": "보험금 부지급 처분 취소 관련 판례",
        "summary": "입원 필요성 판단에서 주치의 소견서의 증명력을 인정한 사례",
        "holding": "약관 해석이 모호한 경우 가입자에게 유리하게 해석 가능",
        "result": "원고 일부 승소",
        "keywords": ["부지급", "입원", "의학적 필요성", "약관 해석"],
        "source": "caselaw_sample",
        "source_type": "caselaw",
    },
    {
        "doc_id": "CASE-002",
        "title": "면책조항 적용 범위 분쟁 판례",
        "summary": "면책 사유 적용 요건은 보험사가 구체적으로 입증해야 한다는 판단",
        "holding": "면책 사유의 해당성은 엄격하게 해석해야 함",
        "result": "원고 승소",
        "keywords": ["면책조항", "입증책임", "보험사"],
        "source": "caselaw_sample",
        "source_type": "caselaw",
    },
    {
        "doc_id": "CASE-003",
        "title": "진단코드 단독 부지급 관련 판례",
        "summary": "진단코드만으로 부지급을 확정하기 어렵고 전체 진료 맥락이 필요",
        "holding": "단일 코드 기반 거절은 사안 종합 판단 원칙에 반할 수 있음",
        "result": "원고 일부 승소",
        "keywords": ["진단코드", "의무기록", "재심의"],
        "source": "caselaw_sample",
        "source_type": "caselaw",
    },
]

MOCK_DISPUTE_CASES: list[dict[str, Any]] = [
    {
        "doc_id": "DISPUTE-001",
        "title": "실손보험 입원 필요성 분쟁조정 사례",
        "summary": "추가 진단서 제출 후 보험금 일부 지급으로 조정 성립",
        "holding": "보강 증빙 제출이 조정 성립에 유효",
        "result": "조정 성립",
        "keywords": ["분쟁조정", "입원 필요성", "진단서"],
        "source": "dispute_case_sample",
        "source_type": "dispute_case",
    },
    {
        "doc_id": "DISPUTE-002",
        "title": "면책조항 해석 이견 분쟁 사례",
        "summary": "약관 문언 해석 차이로 조정 위원회가 가입자 측 일부 인용",
        "holding": "약관 해석의 불명확성은 가입자에게 유리하게 반영",
        "result": "일부 인용",
        "keywords": ["분쟁조정", "면책조항", "약관 해석"],
        "source": "dispute_case_sample",
        "source_type": "dispute_case",
    },
]


def retrieve_cases(queries: list[RetrievalQuery], limit: int = 20) -> list[dict[str, Any]]:
    if not queries:
        return []

    now = datetime.now(tz=timezone.utc).isoformat()
    out: list[dict[str, Any]] = []

    for query in queries:
        source = query.get("source", "caselaw")
        pool = MOCK_CASELAW if source == "caselaw" else MOCK_DISPUTE_CASES
        for doc in pool:
            copied = dict(doc)
            copied["retrieved_at"] = now
            out.append(copied)

    dedup: dict[str, dict[str, Any]] = {}
    for row in out:
        dedup[row.get("doc_id", "")] = row
    return list(dedup.values())[:limit]
