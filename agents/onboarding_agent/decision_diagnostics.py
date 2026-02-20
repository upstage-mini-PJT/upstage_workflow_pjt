"""Diagnostics helpers for onboarding decision trace and reproducibility."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def diagnostics_enabled() -> bool:
    return _env_bool("ONBOARDING_DIAGNOSTICS_ENABLED", True)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_to_jsonable(item) for item in value), key=lambda x: json.dumps(x, ensure_ascii=False))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(obj: Any) -> str:
    return json.dumps(
        _to_jsonable(obj),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def merge_quality_flags(existing: list[str] | None, *flags: str) -> list[str]:
    merged = [str(item).strip() for item in (existing or []) if str(item).strip()]
    seen = set(merged)
    for flag in flags:
        normalized = str(flag).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def build_trace_entry(
    *,
    step_id: str,
    input_obj: Any,
    output_obj: Any,
    rationale_summary: str = "",
    evidence_refs: list[str] | None = None,
    fallback_used: bool = False,
    latency_ms: int | None = None,
    flags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "step_id": str(step_id).strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_fingerprint": fingerprint(input_obj),
        "output_fingerprint": fingerprint(output_obj),
        "decision_payload": _to_jsonable(output_obj),
        "rationale_summary": str(rationale_summary).strip(),
        "evidence_refs": [str(item).strip() for item in (evidence_refs or []) if str(item).strip()],
        "fallback_used": bool(fallback_used),
    }
    if latency_ms is not None:
        entry["latency_ms"] = int(max(latency_ms, 0))
    if flags:
        entry["flags"] = [str(item).strip() for item in flags if str(item).strip()]
    if meta:
        entry["meta"] = _to_jsonable(meta)
    return entry


def append_trace(
    state: dict[str, Any],
    entry: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    trace = list(state.get("decision_trace") or [])
    trace.append(entry)

    step_latency_ms = dict(state.get("step_latency_ms") or {})
    step_id = str(entry.get("step_id", "")).strip()
    latency_raw = entry.get("latency_ms")
    if step_id and isinstance(latency_raw, (int, float)):
        step_latency_ms[step_id] = int(max(float(latency_raw), 0))

    quality_flags = list(state.get("quality_flags") or [])
    entry_flags = [str(item).strip() for item in (entry.get("flags") or []) if str(item).strip()]
    quality_flags = merge_quality_flags(quality_flags, *entry_flags)
    if step_id and bool(entry.get("fallback_used")):
        quality_flags = merge_quality_flags(quality_flags, f"fallback:{step_id}")

    return trace, step_latency_ms, quality_flags


def compute_decision_signature(state: dict[str, Any]) -> str:
    required_ids = sorted(
        {
            str(item).strip()
            for item in (state.get("required_document_ids") or [])
            if str(item).strip()
        }
    )

    retrieval_source_ids: list[str] = []
    for item in list(state.get("retrieval_candidates") or [])[:5]:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        if source_id:
            retrieval_source_ids.append(source_id)

    evidence_sufficient_raw = state.get("evidence_sufficient")
    evidence_sufficient = evidence_sufficient_raw if isinstance(evidence_sufficient_raw, bool) else None

    signature_payload = {
        "required_document_ids": required_ids,
        "evidence_sufficient": evidence_sufficient,
        "final_plan_confidence": str(state.get("final_plan_confidence", "")).strip().lower(),
        "retrieval_top_source_ids": retrieval_source_ids,
    }
    return fingerprint(signature_payload)


def build_run_profile(
    *,
    policy_date: str,
    chat_client: Any = None,
) -> dict[str, Any]:
    model_name = str(
        getattr(chat_client, "model_name", None)
        or getattr(chat_client, "model", None)
        or ""
    ).strip()
    timeout_sec = getattr(chat_client, "timeout", None)
    max_retries = getattr(chat_client, "max_retries", None)

    sample_rate_raw = str(os.getenv("ONBOARDING_EVAL_SAMPLE_RATE", "0.1")).strip()
    try:
        eval_sample_rate = float(sample_rate_raw)
    except ValueError:
        eval_sample_rate = 0.1

    return {
        "policy_date": str(policy_date).strip(),
        "model": model_name,
        "timeout_sec": float(timeout_sec) if isinstance(timeout_sec, (int, float)) else None,
        "max_retries": int(max_retries) if isinstance(max_retries, int) else None,
        "determinism_mode": str(os.getenv("ONBOARDING_DETERMINISM_MODE", "decision_lock")).strip() or "decision_lock",
        "prompt_version": str(os.getenv("ONBOARDING_PROMPT_VERSION", "v1")).strip() or "v1",
        "diagnostics_enabled": diagnostics_enabled(),
        "eval_sample_rate": eval_sample_rate,
    }
