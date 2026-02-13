from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tools.data_analysis_tools.caselaw.types import RetrievalQuery


def retrieve_cases(queries: list[RetrievalQuery], limit: int = 20) -> list[dict[str, Any]]:
    if not queries:
        return []

    timestamp = datetime.now(tz=timezone.utc).isoformat()
    mock_docs: list[dict[str, Any]] = [
        {
            "doc_id": "MOCK-CASE-001",
            "title": "보험금 부지급 처분 취소 관련 사례",
            "summary": "입원 필요성 판단 기준과 의학적 소견서의 증명력 쟁점.",
            "holding": "약관 해석이 모호할 경우 가입자에게 유리하게 해석될 여지가 있음.",
            "keywords": ["부지급", "입원", "약관 해석", "의학적 필요성"],
            "source": "caselaw_sample",
            "retrieved_at": timestamp,
        },
        {
            "doc_id": "MOCK-CASE-002",
            "title": "면책조항 적용 범위 분쟁 사례",
            "summary": "면책조항의 적용 요건 입증 책임이 보험사에 있다는 판단 포인트.",
            "holding": "면책 사유 해당성은 구체적 자료로 입증되어야 함.",
            "keywords": ["면책조항", "입증책임", "보험사"],
            "source": "caselaw_sample",
            "retrieved_at": timestamp,
        },
        {
            "doc_id": "MOCK-CASE-003",
            "title": "진단코드 해석 관련 재심의 사례",
            "summary": "진단코드만으로 부지급하기 어렵고 진료기록 전체 맥락이 필요.",
            "holding": "단일 코드 기반 거절은 사안 종합 판단 원칙에 반할 수 있음.",
            "keywords": ["진단코드", "의무기록", "재심의"],
            "source": "dispute_case_sample",
            "retrieved_at": timestamp,
        },
    ]
    return mock_docs[:limit]
