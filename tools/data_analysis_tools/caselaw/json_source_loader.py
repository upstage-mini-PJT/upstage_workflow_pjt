from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path("data/data_analysis_data/converted_cases")

PRECEDENT_FILES = (
    BASE_DIR / "law_precedents_cleaned.json",
    BASE_DIR / "law_precedents.json",
    BASE_DIR / "law_precedents_followup.json",
)
FSS_FILE = BASE_DIR / "fss_disputes_cleaned.json"


def load_precedent_cases(limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for path in PRECEDENT_FILES:
        for raw in _load_json_rows(path):
            case_id = str(raw.get("case_id", "")).strip()
            if not case_id:
                continue
            doc_id = f"CASELAW:{case_id}"
            if doc_id in seen:
                continue
            seen.add(doc_id)

            rows.append(
                {
                    "doc_id": doc_id,
                    "title": str(raw.get("title", "")).strip(),
                    "summary": str(raw.get("summary", "")).strip(),
                    "holding": str(raw.get("full_text", "")).strip(),
                    "result": str(raw.get("result", "")).strip(),
                    "keywords": _normalize_keywords(raw.get("keywords")),
                    "source": "law_precedents",
                    "source_type": "CASELAW",
                    "url": _string_or_none(raw.get("url")),
                    "published_at": str(raw.get("date", "")).strip() or None,
                }
            )

    if limit is not None:
        return rows[: max(0, limit)]
    return rows


def load_fss_disputes(limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _load_json_rows(FSS_FILE):
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id:
            continue
        rows.append(
            {
                "doc_id": f"DISPUTE:FSS-{case_id}",
                "title": str(raw.get("title", "")).strip(),
                "summary": str(raw.get("summary", "")).strip(),
                "holding": str(raw.get("full_text", "")).strip(),
                "result": str(raw.get("result", "")).strip(),
                "keywords": _normalize_keywords(raw.get("keywords")),
                "source": "fss_dispute",
                "source_type": "DISPUTE",
                "url": _string_or_none(raw.get("url")),
                "published_at": str(raw.get("date", "")).strip() or None,
            }
        )

    if limit is not None:
        return rows[: max(0, limit)]
    return rows


@lru_cache(maxsize=16)
def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _normalize_keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        out = [str(x).strip() for x in value if str(x).strip()]
        return out[:10]
    if isinstance(value, str) and value.strip():
        parts = [x.strip() for x in value.replace("|", ",").split(",")]
        return [x for x in parts if x][:10]
    return []


def _string_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
