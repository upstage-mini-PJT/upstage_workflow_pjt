"""Canonical document catalog for insurance dispute onboarding."""

from __future__ import annotations

from copy import deepcopy


DOCUMENT_CATALOG: list[dict[str, object]] = [
    {
        "id": "denial_notice",
        "name": "보험금 지급거절 통지서 원문",
        "purpose": "보험사가 제시한 거절 사유, 적용 조항, 심사 결과를 확인",
        "issue_types": ["common", "exclusion", "period", "disclosure", "causality", "proof", "code_mismatch"],
    },
    {
        "id": "policy_schedule",
        "name": "보험증권/가입내역서",
        "purpose": "가입일, 담보 구성, 보장개시일, 특약 적용 여부 확인",
        "issue_types": ["common", "period", "code_mismatch"],
    },
    {
        "id": "claim_form",
        "name": "보험금 청구서 사본",
        "purpose": "청구 항목, 청구 사유, 사고일/진단일 기재 내용 확인",
        "issue_types": ["common", "code_mismatch", "proof"],
    },
    {
        "id": "medical_certificate",
        "name": "진단서/의사 소견서",
        "purpose": "질병·상해 진단명, 발병/사고 시점, 치료 필요성 확인",
        "issue_types": ["common", "causality", "proof", "code_mismatch"],
    },
    {
        "id": "discharge_summary",
        "name": "입퇴원 확인서/퇴원요약서",
        "purpose": "입원 필요성, 치료 경과, 입원기간 확인",
        "issue_types": ["period", "proof", "causality"],
    },
    {
        "id": "outpatient_records",
        "name": "외래 진료기록지",
        "purpose": "증상, 진료 경과, 의학적 판단 근거 확인",
        "issue_types": ["proof", "causality", "code_mismatch"],
    },
    {
        "id": "operation_record",
        "name": "수술기록지/마취기록지",
        "purpose": "수술 시행 사실, 코드/수술명, 시행 일시 확인",
        "issue_types": ["proof", "code_mismatch", "causality"],
    },
    {
        "id": "imaging_pathology_reports",
        "name": "영상/병리 판독지",
        "purpose": "객관적 검사 결과와 진단 근거 확인",
        "issue_types": ["proof", "causality", "code_mismatch"],
    },
    {
        "id": "billing_statement",
        "name": "진료비 세부산정내역서/영수증",
        "purpose": "청구 금액, 비급여 항목, 항목별 비용 산출 근거 확인",
        "issue_types": ["proof", "code_mismatch"],
    },
    {
        "id": "nhis_statement",
        "name": "건강보험 급여/비급여 확인서",
        "purpose": "급여·비급여 구분 및 본인부담 범위 확인",
        "issue_types": ["proof", "code_mismatch"],
    },
    {
        "id": "accident_statement",
        "name": "사고경위서(자필 포함)",
        "purpose": "사고 발생 경위, 시간·장소·원인, 책임 관계 확인",
        "issue_types": ["causality", "exclusion", "proof"],
    },
    {
        "id": "police_or_fire_report",
        "name": "경찰/소방/산재 등 공적 사고 확인서",
        "purpose": "사고 사실과 객관적 발생 기록 확인",
        "issue_types": ["causality", "proof", "exclusion"],
    },
    {
        "id": "duty_of_disclosure_docs",
        "name": "계약 전 알릴의무 관련 서류(청약서·질문서)",
        "purpose": "고지 문항, 답변 내용, 서명/설명 절차 확인",
        "issue_types": ["disclosure", "common"],
    },
    {
        "id": "prior_medical_records",
        "name": "과거 진료기록/건강검진 결과",
        "purpose": "기왕증 여부, 증상 지속성, 발병 시점 관련성 확인",
        "issue_types": ["disclosure", "causality", "exclusion"],
    },
    {
        "id": "specialist_opinion",
        "name": "전문의 추가 소견서",
        "purpose": "보험사 판단과 다른 의학적 해석·인과관계 반박 근거 제시",
        "issue_types": ["causality", "proof", "code_mismatch"],
    },
    {
        "id": "beneficiary_identity_docs",
        "name": "수익자 신분/관계 확인서류",
        "purpose": "청구권자 적법성 및 지급 대상자 확인",
        "issue_types": ["common", "proof"],
    },
]

DOCUMENT_BY_ID: dict[str, dict[str, object]] = {
    str(item["id"]): item for item in DOCUMENT_CATALOG
}


def get_document_catalog() -> list[dict[str, object]]:
    """Return a defensive copy of the catalog."""
    return deepcopy(DOCUMENT_CATALOG)


def normalize_required_document_ids(candidate_ids: list[str], *, max_items: int = 5) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in candidate_ids:
        key = str(raw).strip()
        if not key or key not in DOCUMENT_BY_ID or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
        if len(normalized) >= max_items:
            break
    return normalized


def document_display_names(document_ids: list[str]) -> list[str]:
    names: list[str] = []
    for doc_id in document_ids:
        item = DOCUMENT_BY_ID.get(str(doc_id).strip())
        if not item:
            continue
        names.append(str(item.get("name", doc_id)))
    return names


def document_request_items(document_ids: list[str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for doc_id in document_ids:
        item = DOCUMENT_BY_ID.get(str(doc_id).strip())
        if not item:
            continue
        items.append(
            {
                "id": str(item.get("id", "")),
                "name": str(item.get("name", "")),
                "purpose": str(item.get("purpose", "")),
            }
        )
    return items


def build_catalog_prompt_text() -> str:
    lines: list[str] = []
    for item in DOCUMENT_CATALOG:
        issue_types = ", ".join(str(x) for x in item.get("issue_types", []))
        lines.append(
            "- "
            f"id={item.get('id')} | name={item.get('name')} | "
            f"purpose={item.get('purpose')} | issue_types={issue_types}"
        )
    return "\n".join(lines)
