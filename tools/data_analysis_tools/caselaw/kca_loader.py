from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_KCA_PATH = Path("data/data_analysis_data/converted_cases/kca_finance_insurance_cases_all.json")

INSURANCE_KEYWORDS: tuple[str, ...] = (
    "보험",
    "보험금",
    "실손",
    "실비",
    "암보험",
    "입원",
    "통원",
    "면책",
    "약관",
    "재심의",
)


def load_kca_raw_cases(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or DEFAULT_KCA_PATH
    if not source.exists():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def filter_insurance_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _is_insurance_case(row)]


def normalize_kca_case(row: dict[str, Any]) -> dict[str, Any]:
    raw_id = str(row.get("번호", "")).strip() or _fallback_id(row)
    title = str(row.get("제목", "")).strip()
    detail = row.get("상세내용", {})
    detail_dict = detail if isinstance(detail, dict) else {}

    summary = _compose_text(detail_dict, ("사건개요", "신청인주장", "당사자주장"))
    holding = _compose_text(detail_dict, ("판단", "처리결과", "결정내용"))
    result = str(detail_dict.get("처리결과") or detail_dict.get("결정내용") or "").strip()
    keywords = _extract_keywords(title, summary + " " + holding)

    return {
        "doc_id": f"DISPUTE:KCA-{raw_id}",
        "title": title or f"KCA 분쟁사례 {raw_id}",
        "summary": summary or title,
        "holding": holding,
        "result": result,
        "keywords": keywords,
        "source": "kca_dispute",
        "source_type": "DISPUTE",
        "published_at": _normalize_date(str(row.get("수정일", ""))),
        "url": None,
    }


def load_normalized_kca_disputes(
    path: Path | None = None,
    insurance_only: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = load_kca_raw_cases(path=path)
    if insurance_only:
        rows = filter_insurance_cases(rows)

    normalized = [normalize_kca_case(row) for row in rows]
    if limit is not None:
        return normalized[: max(0, limit)]
    return normalized


def _is_insurance_case(row: dict[str, Any]) -> bool:
    title = str(row.get("제목", ""))
    detail = row.get("상세내용", {})
    detail_text = ""
    if isinstance(detail, dict):
        detail_text = " ".join(str(v) for v in detail.values())
    text = f"{title} {detail_text}"
    return any(keyword in text for keyword in INSURANCE_KEYWORDS)


def _compose_text(detail: dict[str, Any], keys: tuple[str, ...]) -> str:
    parts: list[str] = []
    for key in keys:
        value = str(detail.get(key, "")).strip()
        if value:
            parts.append(value)
    return " ".join(parts)


def _extract_keywords(title: str, body: str) -> list[str]:
    text = f"{title} {body}"
    found = [keyword for keyword in INSURANCE_KEYWORDS if keyword in text]
    return found[:8]


def _normalize_date(value: str) -> str:
    clean = value.strip()
    if not clean:
        return "1970-01-01"
    return clean


def _fallback_id(row: dict[str, Any]) -> str:
    title = str(row.get("제목", "")).strip()
    if title:
        return str(abs(hash(title)))[:10]
    return "unknown"
