from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

from core.schemas.rag_contract import IngestionDoc, SourceType
from core.schemas.rag_conventions import make_doc_id

_ALLOWED_SOURCE_TYPES: set[str] = {"CASELAW", "DISPUTE", "WEB"}
_DEFAULT_PUBLISHED_AT = "1970-01-01"


class IngestionNormalizationError(ValueError):
    """Raised when a raw document cannot be normalized into IngestionDoc."""


def normalize_ingestion_doc(raw: dict[str, Any]) -> IngestionDoc:
    source_type = _normalize_source_type(raw.get("source_type"))

    title = str(raw.get("title", "")).strip()
    body = _extract_body(raw)
    if not title:
        raise IngestionNormalizationError("title must not be empty")
    if not body:
        raise IngestionNormalizationError("body must not be empty")

    raw_id = _resolve_raw_id(raw, source_type)
    doc_id = make_doc_id(source_type, raw_id)

    published_at, parse_meta = _normalize_published_at(raw.get("published_at"))
    tags = _normalize_tags(raw.get("tags") or raw.get("keywords"))
    jurisdiction = _to_optional_str(raw.get("jurisdiction") or raw.get("court"))
    url = _to_optional_str(raw.get("url"))

    meta = _extract_meta(raw)
    if parse_meta:
        meta.update(parse_meta)

    return IngestionDoc(
        doc_id=doc_id,
        source_type=source_type,
        title=title,
        body=body,
        published_at=published_at,
        jurisdiction=jurisdiction,
        tags=tags,
        url=url,
        meta=meta,
    )


def normalize_ingestion_docs(raw_docs: list[dict[str, Any]]) -> list[IngestionDoc]:
    normalized: list[IngestionDoc] = []
    for raw in raw_docs:
        normalized.append(normalize_ingestion_doc(raw))
    return normalized


def _normalize_source_type(value: Any) -> SourceType:
    source = str(value or "WEB").strip().upper()
    if source not in _ALLOWED_SOURCE_TYPES:
        raise IngestionNormalizationError(f"unsupported source_type: {source}")
    return source  # type: ignore[return-value]


def _extract_body(raw: dict[str, Any]) -> str:
    for key in ("body", "content", "summary", "holding"):
        value = str(raw.get(key, "")).strip()
        if value:
            return value
    return ""


def _resolve_raw_id(raw: dict[str, Any], source_type: SourceType) -> str:
    candidate = str(raw.get("doc_id") or raw.get("id") or raw.get("source_id") or "").strip()
    if candidate:
        return candidate

    fingerprint = "|".join(
        [
            source_type,
            str(raw.get("title", "")).strip(),
            str(raw.get("url", "")).strip(),
        ]
    )
    if not fingerprint.replace("|", ""):
        raise IngestionNormalizationError("cannot derive raw_id from empty document")
    return sha1(fingerprint.encode("utf-8")).hexdigest()[:16]


def _normalize_published_at(value: Any) -> tuple[str, dict[str, Any]]:
    if value is None:
        return _DEFAULT_PUBLISHED_AT, {"published_at_parse_failed": True, "published_at_raw": None}

    raw = str(value).strip()
    if not raw:
        return _DEFAULT_PUBLISHED_AT, {"published_at_parse_failed": True, "published_at_raw": raw}

    parsed = _try_parse_date(raw)
    if parsed is None:
        return _DEFAULT_PUBLISHED_AT, {"published_at_parse_failed": True, "published_at_raw": raw}

    return parsed, {}


def _try_parse_date(value: str) -> str | None:
    if len(value) == 10:
        try:
            return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None

    iso_candidate = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate).date().isoformat()
    except ValueError:
        return None


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    out: list[str] = []
    for item in value:
        tag = str(item).strip()
        if tag:
            out.append(tag)
    return out


def _extract_meta(raw: dict[str, Any]) -> dict[str, Any]:
    provided_meta = raw.get("meta")
    if isinstance(provided_meta, dict):
        meta = dict(provided_meta)
    else:
        meta = {}

    for key in ("source", "source_name", "publisher", "language"):
        if key in raw and raw[key] is not None:
            meta[key] = raw[key]
    meta.setdefault("normalized_at", datetime.now(tz=timezone.utc).isoformat())
    return meta


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
