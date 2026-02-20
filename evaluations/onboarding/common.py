from __future__ import annotations

import json
import os
from pathlib import Path
import re
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_upstage import ChatUpstage, UpstageUniversalInformationExtraction
from langgraph.types import Command

from agents.onboarding_agent.decision_diagnostics import build_run_profile, compute_decision_signature
from agents.onboarding_agent.temp_builder import onboarding_graph
from tools.retrieve_terms import ensure_vectordb_ready, list_policy_dates, load_vectordb


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _apply_eval_llm_mode() -> None:
    """
    Eval path defaults to strict+retry to avoid "all fallback but stable" false positives.
    Can be overridden by explicit environment values.
    """
    if not _env_bool("ONBOARDING_EVAL_STRICT_LLM", True):
        return
    os.environ.setdefault("ONBOARDING_STRICT_LLM_MODE", "true")
    os.environ.setdefault("ONBOARDING_LLM_RETRY_ON_CONNECTION", "true")
    os.environ.setdefault("ONBOARDING_LLM_WAIT_FOREVER", "false")
    os.environ.setdefault("ONBOARDING_LLM_MAX_ATTEMPTS", "120")
    os.environ.setdefault("ONBOARDING_LLM_MAX_WAIT_SEC", "3600")
    os.environ.setdefault("ONBOARDING_LLM_RETRY_BACKOFF_SEC", "3")


def _build_chat_client() -> ChatUpstage:
    model_name = str(os.getenv("ONBOARDING_CHAT_MODEL", "solar-pro2")).strip() or "solar-pro2"
    try:
        timeout_sec = float(str(os.getenv("ONBOARDING_CHAT_TIMEOUT", "45")).strip())
    except ValueError:
        timeout_sec = 45.0
    try:
        max_retries = int(str(os.getenv("ONBOARDING_CHAT_MAX_RETRIES", "1")).strip())
    except ValueError:
        max_retries = 1
    return ChatUpstage(model=model_name, timeout=max(timeout_sec, 5.0), max_retries=max(max_retries, 0))


def load_dataset(path: str) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise SystemExit(f"dataset 파일이 존재하지 않습니다: {dataset_path}")
    rows: list[dict[str, Any]] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_mock_manifest(path: str) -> dict[str, str]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw_map = payload.get("catalog_to_file", {})
    if not isinstance(raw_map, dict):
        return {}
    resolved: dict[str, str] = {}
    for key, value in raw_map.items():
        doc_id = str(key).strip()
        doc_path = str(value).strip()
        if not doc_id or not doc_path:
            continue
        resolved[doc_id] = doc_path
    return resolved


def _resolve_mock_resume_paths(required_ids: list[str], mock_map: dict[str, str]) -> list[str]:
    if not required_ids:
        return []
    missing = [doc_id for doc_id in required_ids if not str(mock_map.get(doc_id, "")).strip()]
    if missing:
        raise SystemExit(f"manifest 미매핑 doc_id: {missing}")
    paths: list[str] = []
    for doc_id in required_ids:
        path = str(mock_map.get(doc_id, "")).strip()
        if not path or not Path(path).exists():
            raise SystemExit(f"manifest 매핑 경로가 유효하지 않습니다: {doc_id} -> {path}")
        paths.append(path)
    return paths


def _validate_yyyymmdd(value: str, *, label: str) -> str:
    cleaned = str(value).strip()
    if len(cleaned) != 8 or not cleaned.isdigit():
        raise SystemExit(f"{label}는 YYYYMMDD 형식이어야 합니다. 입력값: {value}")
    return cleaned


def select_policy_date(join_date: str, available_dates: list[str]) -> str:
    normalized = _validate_yyyymmdd(join_date, label="join_date")
    sorted_dates = sorted({d for d in available_dates if len(d) == 8 and d.isdigit()})
    if not sorted_dates:
        raise SystemExit("vector DB에 유효한 policy_date가 없습니다.")
    if normalized in sorted_dates:
        return normalized
    prior_dates = [d for d in sorted_dates if d <= normalized]
    if prior_dates:
        return prior_dates[-1]
    return sorted_dates[0]


def prepare_runtime() -> dict[str, Any]:
    _apply_eval_llm_mode()
    policy_vectordb = ensure_vectordb_ready(load_vectordb())
    available_policy_dates = list_policy_dates(policy_vectordb)
    chat_client = _build_chat_client()
    return {
        "policy_vectordb": policy_vectordb,
        "available_policy_dates": available_policy_dates,
        "chat_client": chat_client,
    }


def run_onboarding_once(case: dict[str, Any], runtime: dict[str, Any], *, repeat_index: int = 0) -> dict[str, Any]:
    denial_file_path = str(case.get("denial_file_path", "")).strip()
    if not denial_file_path or not Path(denial_file_path).exists():
        raise SystemExit(f"denial_file_path가 유효하지 않습니다: {denial_file_path}")

    join_date = str(case.get("join_date", "")).strip() or "20240301"
    policy_date = str(case.get("policy_date", "")).strip()
    if not policy_date:
        policy_date = select_policy_date(join_date, list(runtime.get("available_policy_dates", [])))

    manifest_path = str(
        case.get("mock_manifest_path")
        or os.getenv("ONBOARDING_MOCK_MANIFEST", "data/mock_documents/hyundai_senior_silson_case/manifest.json")
    ).strip()
    mock_map = _load_mock_manifest(manifest_path)
    auto_resume_mock = bool(case.get("auto_resume_mock", True))

    chat_client = runtime.get("chat_client")
    run_profile = build_run_profile(policy_date=policy_date, chat_client=chat_client)
    thread_id = f"eval-{str(case.get('scenario_id', 'unknown')).strip()}-{repeat_index}-{uuid.uuid4()}"

    config: RunnableConfig = {
        "configurable": {
            "ie_client": UpstageUniversalInformationExtraction(),
            "chat_client": chat_client,
            "policy_vectordb": runtime.get("policy_vectordb"),
            "policy_date": policy_date,
            "thread_id": thread_id,
            # Evaluation path: bypass interrupt and inject mapped mock docs directly.
            "test_auto_resume_documents": auto_resume_mock,
            "test_auto_resume_max_rounds": int(case.get("test_auto_resume_max_rounds", 2) or 2),
            "mock_document_map": mock_map,
        }
    }

    result = onboarding_graph.invoke(
        {
            "denial_file_path": denial_file_path,
            "policy_date": policy_date,
            "run_profile": run_profile,
        },
        config=config,
    )

    interrupt_count = 0
    max_interrupts = int(case.get("max_interrupts", 5) or 5)
    while result.get("__interrupt__"):
        interrupt_count += 1
        if interrupt_count > max_interrupts:
            payload = result["__interrupt__"][0].value if result.get("__interrupt__") else {}
            required_ids = payload.get("required_document_ids", []) if isinstance(payload, dict) else []
            required_ids = [str(item).strip() for item in required_ids if str(item).strip()]
            truncated_state = {
                key: value
                for key, value in dict(result).items()
                if key != "__interrupt__"
            }
            quality_flags = [str(item).strip() for item in (truncated_state.get("quality_flags") or []) if str(item).strip()]
            if "interrupt_limit_exceeded" not in quality_flags:
                quality_flags.append("interrupt_limit_exceeded")
            truncated_state["quality_flags"] = quality_flags
            if required_ids and not truncated_state.get("required_document_ids"):
                truncated_state["required_document_ids"] = required_ids
            truncated_state["evidence_sufficient"] = False
            truncated_state["final_plan_confidence"] = str(
                truncated_state.get("final_plan_confidence", "")
            ).strip() or "low"
            truncated_state["run_profile"] = dict(truncated_state.get("run_profile") or run_profile)
            truncated_state["decision_signature"] = compute_decision_signature(truncated_state)
            return truncated_state
        payload = result["__interrupt__"][0].value if result.get("__interrupt__") else {}
        required_ids = payload.get("required_document_ids", []) if isinstance(payload, dict) else []
        required_ids = [str(item).strip() for item in required_ids if str(item).strip()]
        if not required_ids:
            raise SystemExit(f"interrupt payload에 required_document_ids가 없습니다: {payload}")

        if auto_resume_mock:
            resume_paths = _resolve_mock_resume_paths(required_ids, mock_map)
        else:
            raise SystemExit("evaluation runner는 auto_resume_mock=false 모드를 지원하지 않습니다.")
        result = onboarding_graph.invoke(Command(resume=resume_paths), config=config)

    final_state = dict(result)
    final_state["run_profile"] = dict(final_state.get("run_profile") or run_profile)
    final_state["decision_signature"] = compute_decision_signature(final_state)
    return final_state


def normalize_run_summary(state: dict[str, Any], *, scenario_id: str, repeat_index: int) -> dict[str, Any]:
    required_ids = sorted({str(item).strip() for item in (state.get("required_document_ids") or []) if str(item).strip()})
    retrieval_top_source_ids = [
        str(item.get("source_id", "")).strip()
        for item in list(state.get("retrieval_candidates") or [])[:5]
        if isinstance(item, dict) and str(item.get("source_id", "")).strip()
    ]
    issue_hypotheses = [
        str(item).strip()
        for item in (state.get("issue_hypotheses") or [])
        if str(item).strip()
    ]

    raw_query_plan = list(state.get("query_plan") or [])
    issue_query_plan: list[dict[str, Any]] = []
    issue_query_seeds: list[str] = []
    for item in raw_query_plan:
        if not isinstance(item, dict):
            continue
        intent = str(item.get("intent", "")).strip()
        query_seed = str(item.get("query_seed", "")).strip()
        must_keywords = [
            str(keyword).strip()
            for keyword in (item.get("must_keywords") or [])
            if str(keyword).strip()
        ]
        priority = item.get("priority")
        issue_query_plan.append(
            {
                "intent": intent,
                "query_seed": query_seed,
                "must_keywords": must_keywords,
                "priority": priority,
            }
        )
        if query_seed:
            issue_query_seeds.append(query_seed)

    final_plan_text = re.sub(r"\s+", " ", str(state.get("plan", "")).strip())
    final_plan_preview = final_plan_text[:400]

    extracted_document_infos: list[dict[str, str]] = []
    for item in list(state.get("extracted_document_infos") or [])[:6]:
        if not isinstance(item, dict):
            continue
        extracted_document_infos.append(
            {
                "source_name": str(item.get("source_name", "")).strip(),
                "key_data": str(item.get("key_data", "")).strip(),
                "evidence_or_grounds": str(item.get("evidence_or_grounds", "")).strip(),
                "helpful_notes": str(item.get("helpful_notes", "")).strip(),
            }
        )

    return {
        "scenario_id": scenario_id,
        "repeat_index": repeat_index,
        "decision_signature": str(state.get("decision_signature", "")).strip(),
        "issue_hypotheses": issue_hypotheses,
        "issue_query_plan": issue_query_plan,
        "issue_query_seeds": issue_query_seeds,
        "final_plan_text": final_plan_text,
        "final_plan_preview": final_plan_preview,
        "required_document_ids": required_ids,
        "extracted_document_infos": extracted_document_infos,
        "evidence_sufficient": state.get("evidence_sufficient"),
        "final_plan_confidence": str(state.get("final_plan_confidence", "")).strip(),
        "retrieval_top_source_ids": retrieval_top_source_ids,
        "quality_flags": list(state.get("quality_flags") or []),
    }
