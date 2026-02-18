from __future__ import annotations

from core.schemas.case_context import StructuredCase


USER_SCENARIOS: list[dict[str, object]] = [
    {
        "scenario_id": "S01",
        "name": "입원 필요성 부인",
        "structured_case": StructuredCase(
            user_info={"age": 46, "gender": "F"},
            denial_summary="입원 필요성이 인정되지 않아 보험금 지급 거절",
            denial_reasons=["의학적 필요성 부족"],
            policy_clauses=["입원 필요성 인정 시 지급"],
            timeline=[
                {"date": "2025-01-03", "description": "입원 치료", "actor": "병원"},
                {"date": "2025-01-20", "description": "부지급 통지", "actor": "보험사"},
            ],
            evidence_summary=[{"title": "진단서", "summary": "입원 치료 권고", "document_type": "진단서", "source": "병원"}],
            open_questions=["추가 소견서 제출 필요 여부"],
        ),
    },
    {
        "scenario_id": "S02",
        "name": "면책조항 적용 다툼",
        "structured_case": StructuredCase(
            user_info={"age": 53, "gender": "M"},
            denial_summary="면책조항 해당으로 지급 제외",
            denial_reasons=["면책조항 적용"],
            policy_clauses=["기왕증 관련 면책"],
            timeline=[{"date": "2024-11-10", "description": "치료 개시", "actor": "병원"}],
            evidence_summary=[{"title": "의무기록", "summary": "사고 후 증상 악화", "document_type": "의무기록", "source": "병원"}],
            open_questions=["면책조항 해석 범위"],
        ),
    },
    {
        "scenario_id": "S03",
        "name": "진단코드 단독 거절",
        "structured_case": StructuredCase(
            user_info={"age": 37, "gender": "F"},
            denial_summary="진단코드 상 경증으로 판단되어 거절",
            denial_reasons=["진단코드 해석"],
            policy_clauses=["중증도 기준"],
            timeline=[{"date": "2025-02-01", "description": "외래 치료", "actor": "병원"}],
            evidence_summary=[{"title": "진료기록", "summary": "지속 통증 및 기능 저하", "document_type": "진료기록", "source": "병원"}],
            open_questions=["기능저하 평가서 보강 필요"],
        ),
    },
    {
        "scenario_id": "S04",
        "name": "서류 미비 거절",
        "structured_case": StructuredCase(
            user_info={"age": 29, "gender": "M"},
            denial_summary="필수 서류 누락으로 청구 반려",
            denial_reasons=["증빙 부족"],
            policy_clauses=["청구서류 제출 의무"],
            timeline=[{"date": "2025-01-12", "description": "보험금 청구", "actor": "가입자"}],
            evidence_summary=[{"title": "청구서", "summary": "기본 서류만 제출", "document_type": "청구서", "source": "가입자"}],
            open_questions=["소견서/입퇴원확인서 추가 필요"],
        ),
    },
    {
        "scenario_id": "S05",
        "name": "고지의무 위반 주장",
        "structured_case": StructuredCase(
            user_info={"age": 58, "gender": "F"},
            denial_summary="계약 체결 시 고지의무 위반으로 해지 및 부지급",
            denial_reasons=["고지의무 위반"],
            policy_clauses=["고지의무 위반 시 계약 해지"],
            timeline=[{"date": "2023-05-02", "description": "보험 가입", "actor": "가입자"}],
            evidence_summary=[{"title": "문진표", "summary": "기왕증 질문 응답", "document_type": "문진표", "source": "보험사"}],
            open_questions=["중요사항 해당성 여부"],
        ),
    },
    {
        "scenario_id": "S06",
        "name": "자동차사고 후 후유장해",
        "structured_case": StructuredCase(
            user_info={"age": 44, "gender": "M"},
            denial_summary="후유장해율 기준 미달로 부지급",
            denial_reasons=["장해율 기준 미달"],
            policy_clauses=["장해율 50% 이상 지급"],
            timeline=[{"date": "2024-08-20", "description": "교통사고", "actor": "가입자"}],
            evidence_summary=[{"title": "장해진단서", "summary": "장해율 48% 산정", "document_type": "장해진단서", "source": "병원"}],
            open_questions=["추가 감정 필요성"],
        ),
    },
    {
        "scenario_id": "S07",
        "name": "치료 기간 과다 주장",
        "structured_case": StructuredCase(
            user_info={"age": 33, "gender": "F"},
            denial_summary="치료 기간이 과도하여 일부 기간 불인정",
            denial_reasons=["치료 적정성 다툼"],
            policy_clauses=["통상 치료기간 인정"],
            timeline=[{"date": "2025-01-05", "description": "입원 시작", "actor": "병원"}],
            evidence_summary=[{"title": "경과기록", "summary": "통증 지속 및 재활 필요", "document_type": "경과기록", "source": "병원"}],
            open_questions=["재활 필요성 근거 추가"],
        ),
    },
    {
        "scenario_id": "S08",
        "name": "중복보상 주장",
        "structured_case": StructuredCase(
            user_info={"age": 49, "gender": "M"},
            denial_summary="타 보험사 지급 이력으로 중복보상 불가 주장",
            denial_reasons=["중복보상 제한"],
            policy_clauses=["실손 비례보상"],
            timeline=[{"date": "2025-02-10", "description": "타사 일부 지급", "actor": "타 보험사"}],
            evidence_summary=[{"title": "지급내역서", "summary": "일부 항목만 지급", "document_type": "지급내역서", "source": "타 보험사"}],
            open_questions=["미지급 항목 분리 가능성"],
        ),
    },
    {
        "scenario_id": "S09",
        "name": "사고 인과관계 부인",
        "structured_case": StructuredCase(
            user_info={"age": 41, "gender": "F"},
            denial_summary="기왕증 영향으로 사고 인과관계 불충분",
            denial_reasons=["인과관계 부정"],
            policy_clauses=["사고와 직접 인과관계 필요"],
            timeline=[{"date": "2024-12-15", "description": "넘어짐 사고", "actor": "가입자"}],
            evidence_summary=[{"title": "영상판독", "summary": "사고 직후 악화 소견", "document_type": "영상자료", "source": "병원"}],
            open_questions=["전문의 소견서 추가"],
        ),
    },
    {
        "scenario_id": "S10",
        "name": "약관 문언 모호성",
        "structured_case": StructuredCase(
            user_info={"age": 55, "gender": "M"},
            denial_summary="약관 문언 해석상 지급 대상 아님",
            denial_reasons=["약관 해석 다툼"],
            policy_clauses=["해당 치료는 보장 제외"],
            timeline=[{"date": "2025-01-28", "description": "재심의 요청", "actor": "가입자"}],
            evidence_summary=[{"title": "치료확인서", "summary": "치료 목적과 필요성 명시", "document_type": "치료확인서", "source": "병원"}],
            open_questions=["약관 불명확성 주장 가능 여부"],
        ),
    },
]
