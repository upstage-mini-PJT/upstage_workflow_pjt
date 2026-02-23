"""
Shared workflow logic for both the FastAPI backend and the CLI.

Public sync helpers (used directly by CLI):
    build_runnable_config(thread_id, join_date) -> tuple[RunnableConfig, str]
    invoke_onboarding_graph(config, denial_file_path, policy_date) -> dict
    resume_onboarding_graph(config, doc_paths) -> dict
    run_analysis(thread_id, onboarding_state) -> dict
    build_structured_case(onboarding_state) -> StructuredCase

Async wrappers (used by FastAPI routes via asyncio.to_thread):
    async start_onboarding_workflow(thread_id, denial_file_path, join_date) -> dict
    async resume_onboarding_workflow(doc_paths, config) -> dict
    async run_analysis_workflow(thread_id, onboarding_state) -> dict
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: load .env and resolve relative path env vars to absolute BEFORE
# importing any downstream module that computes paths at module level
# (e.g. tools/retrieve_terms.py, agents/onboarding_agent/chunker_embedder.py).
# ---------------------------------------------------------------------------
import logging  # noqa: E402
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_wlog = logging.getLogger("api.workflow.bootstrap")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_wlog.info("PROJECT_ROOT  = %s", _PROJECT_ROOT)
_wlog.info("CWD           = %s", Path.cwd())

from dotenv import load_dotenv  # noqa: E402
load_dotenv(_PROJECT_ROOT / ".env")

for _env_key in ("POLICIES_DIR", "VECTOR_DIR"):
    _env_val = os.getenv(_env_key, "")
    if _env_val and not Path(_env_val).is_absolute():
        os.environ[_env_key] = str((_PROJECT_ROOT / _env_val).resolve())

_wlog.info("POLICIES_DIR  = %s", os.getenv("POLICIES_DIR", "<not set — using default>"))
_wlog.info("VECTOR_DIR    = %s", os.getenv("VECTOR_DIR",   "<not set — using default>"))

from langchain_core.runnables import RunnableConfig
from langchain_upstage import ChatUpstage, UpstageUniversalInformationExtraction
from langgraph.types import Command

from agents.data_analysis_agent.agent import run as run_data_analysis
from agents.onboarding_agent.temp_builder import onboarding_graph
from core.schemas.case_context import StructuredCase, normalize_structured_case
from tools.retrieve_terms import ensure_vectordb_ready, list_policy_dates, load_vectordb


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    return ChatUpstage(
        model=model_name,
        timeout=max(timeout_sec, 5.0),
        max_retries=max(max_retries, 0),
    )


def _select_policy_date(join_date: str, available_dates: list[str]) -> tuple[str, str]:
    if not available_dates:
        raise ValueError("vector DB에 policy_date가 없어 가입일 기반 매핑을 수행할 수 없습니다.")
    sorted_dates = sorted({d for d in available_dates if len(d) == 8 and d.isdigit()})
    if not sorted_dates:
        raise ValueError("vector DB의 policy_date 형식이 유효하지 않습니다.")
    if join_date in sorted_dates:
        return join_date, "가입일과 동일한 policy_date 사용"
    prior_dates = [d for d in sorted_dates if d <= join_date]
    if prior_dates:
        chosen = prior_dates[-1]
        return chosen, f"가입일({join_date}) 기준 가장 가까운 이전 policy_date({chosen}) 선택"
    chosen = sorted_dates[0]
    return chosen, f"가입일({join_date})이 모든 버전보다 이전이라 최소 policy_date({chosen}) 선택"


def _extract_interrupt_info(graph_result: dict[str, Any]) -> tuple[bool, list[str], list[str], list[dict]]:
    """Return (interrupted, required_document_ids, required_documents, required_document_items)."""
    interrupted = bool(graph_result.get("__interrupt__"))
    if not interrupted:
        return False, [], [], []
    payload = graph_result["__interrupt__"][0].value if graph_result.get("__interrupt__") else {}
    if not isinstance(payload, dict):
        payload = {}
    ids = [str(i).strip() for i in payload.get("required_document_ids", []) if str(i).strip()]
    docs = payload.get("required_documents", [])
    items = payload.get("required_document_items", [])
    return True, ids, docs, items


def _normalize_policy_clauses(decision_summary: dict[str, Any]) -> list[str]:
    clauses: list[str] = []
    raw = decision_summary.get("policy_clauses", [])
    if not isinstance(raw, list):
        return clauses
    for item in raw:
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            merged = " - ".join([part for part in [title, snippet] if part])
            if merged:
                clauses.append(merged)
        else:
            text = str(item).strip()
            if text:
                clauses.append(text)
    return clauses


def _normalize_evidence_summary(state: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    extracted = state.get("extracted_document_infos", [])
    if not isinstance(extracted, list):
        return result
    for idx, info in enumerate(extracted, start=1):
        if not isinstance(info, dict):
            continue
        key_data = str(info.get("key_data", "")).strip()
        evidence = str(info.get("evidence_or_grounds", "")).strip()
        notes = str(info.get("helpful_notes", "")).strip()
        summary = " / ".join([part for part in [key_data, evidence, notes] if part])
        result.append({
            "title": f"추가서류 {idx}",
            "summary": summary or "추출 정보 없음",
            "document_type": "additional_document",
            "source": "onboarding.extracted_document_infos",
        })
    return result


# ---------------------------------------------------------------------------
# Public sync helpers
# ---------------------------------------------------------------------------

def build_runnable_config(thread_id: str, join_date: str) -> tuple[RunnableConfig, str]:
    """Prepare LangGraph RunnableConfig and return (config, selected_policy_date)."""
    policy_vectordb = ensure_vectordb_ready(load_vectordb())
    available_dates = list_policy_dates(policy_vectordb)
    selected_policy_date, _ = _select_policy_date(join_date, available_dates)
    config: RunnableConfig = {
        "configurable": {
            "ie_client": UpstageUniversalInformationExtraction(),
            "chat_client": _build_chat_client(),
            "policy_vectordb": policy_vectordb,
            "policy_date": selected_policy_date,
            "thread_id": thread_id,
        }
    }
    return config, selected_policy_date


def invoke_onboarding_graph(
    config: RunnableConfig,
    denial_file_path: str,
    policy_date: str,
) -> dict[str, Any]:
    """Initial invocation of the onboarding graph.

    Returns a dict with keys:
        graph_result, config, interrupted,
        required_document_ids, required_documents, required_document_items
    """
    graph_result = onboarding_graph.invoke(
        {"denial_file_path": denial_file_path, "policy_date": policy_date},
        config=config,
    )
    interrupted, ids, docs, items = _extract_interrupt_info(dict(graph_result))
    return {
        "graph_result": dict(graph_result),
        "config": config,
        "interrupted": interrupted,
        "required_document_ids": ids,
        "required_documents": docs,
        "required_document_items": items,
    }


def resume_onboarding_graph(
    config: RunnableConfig,
    doc_paths: list[str],
) -> dict[str, Any]:
    """Resume the onboarding graph after an interrupt with provided document paths.

    Returns same shape as invoke_onboarding_graph.
    """
    graph_result = onboarding_graph.invoke(Command(resume=doc_paths), config=config)
    interrupted, ids, docs, items = _extract_interrupt_info(dict(graph_result))
    return {
        "graph_result": dict(graph_result),
        "config": config,
        "interrupted": interrupted,
        "required_document_ids": ids,
        "required_documents": docs,
        "required_document_items": items,
    }


def build_structured_case(onboarding_state: dict[str, Any]) -> StructuredCase:
    """Convert onboarding state dict into a StructuredCase for analysis."""
    decision_summary = onboarding_state.get("decision_summary", {})
    if not isinstance(decision_summary, dict):
        decision_summary = {}

    denial_reasons: list[str] = []
    insurer_claim = str(decision_summary.get("insurer_claim", "")).strip()
    conclusion_reason = str(decision_summary.get("conclusion_reason", "")).strip()
    if insurer_claim:
        denial_reasons.append(insurer_claim)
    if conclusion_reason and conclusion_reason != insurer_claim:
        denial_reasons.append(conclusion_reason)

    timeline: list[dict[str, str]] = []
    user_situation = str(decision_summary.get("user_situation", "")).strip()
    if user_situation:
        timeline.append({
            "date": "",
            "description": user_situation,
            "actor": "insured",
            "source": "onboarding.decision_summary",
        })

    structured_case = {
        "user_info": {
            "policy_date": str(onboarding_state.get("policy_date", "")).strip(),
            "final_plan_confidence": str(onboarding_state.get("final_plan_confidence", "")).strip(),
            "decision_explanation": str(onboarding_state.get("decision_explanation", "")).strip(),
            "decision_summary_user_situation": str(decision_summary.get("user_situation", "")).strip(),
            "decision_summary_insurer_claim": str(decision_summary.get("insurer_claim", "")).strip(),
            "decision_summary_conclusion_reason": str(decision_summary.get("conclusion_reason", "")).strip(),
            "required_documents": [
                str(x).strip()
                for x in onboarding_state.get("required_documents", [])
                if str(x).strip()
            ],
        },
        "denial_summary": (
            str(onboarding_state.get("plan", "")).strip()
            or str(onboarding_state.get("denial_statement_text", "")).strip()[:800]
        ),
        "denial_reasons": denial_reasons,
        "policy_clauses": _normalize_policy_clauses(decision_summary),
        "timeline": timeline,
        "evidence_summary": _normalize_evidence_summary(onboarding_state),
        "open_questions": [
            str(x).strip()
            for x in onboarding_state.get("required_documents", [])
            if str(x).strip()
        ],
    }
    return normalize_structured_case(structured_case)


def run_analysis(thread_id: str, onboarding_state: dict[str, Any]) -> dict[str, Any]:
    """Run data analysis pipeline and return AnalysisResult as a plain dict."""
    risk_level = str(os.getenv("STEP3_RISK_LEVEL", "BALANCED")).strip().upper() or "BALANCED"
    output_style = str(os.getenv("STEP3_OUTPUT_STYLE", "USER_READABLE")).strip().upper() or "USER_READABLE"
    analysis_options: dict[str, Any] = {
        "risk_level": risk_level,
        "output_style": output_style,
        "thread_id": thread_id,
        "entrypoint": "api",
    }
    structured_case = build_structured_case(onboarding_state)
    result = run_data_analysis(
        structured_case,
        rag_result=None,
        analysis_options=analysis_options,
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Async wrappers for FastAPI (run blocking calls in a thread pool)
# ---------------------------------------------------------------------------

async def start_onboarding_workflow(
    thread_id: str,
    denial_file_path: str,
    join_date: str,
) -> dict[str, Any]:
    """Async: build config + initial graph invocation."""
    def _run() -> dict[str, Any]:
        config, policy_date = build_runnable_config(thread_id, join_date)
        return invoke_onboarding_graph(config, denial_file_path, policy_date)

    return await asyncio.to_thread(_run)


async def resume_onboarding_workflow(
    doc_paths: list[str],
    config: RunnableConfig,
) -> dict[str, Any]:
    """Async: resume graph after interrupt."""
    return await asyncio.to_thread(resume_onboarding_graph, config, doc_paths)


async def run_analysis_workflow(
    thread_id: str,
    onboarding_state: dict[str, Any],
) -> dict[str, Any]:
    """Async: run data analysis pipeline."""
    return await asyncio.to_thread(run_analysis, thread_id, onboarding_state)
