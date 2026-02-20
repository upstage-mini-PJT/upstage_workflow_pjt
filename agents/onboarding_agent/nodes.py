"""
onboarding_agent용 그래프 노드 정의.

- temp_builder.py: 여기서 노드를 import 해서 테스트용 그래프 구성
- workflow/builder.py: 여기서 노드를 import 해서 메인 그래프에 add_node
"""

from time import perf_counter, sleep
from typing import Any
import os
import re

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from tools.document_parser import parse_document

from agents.onboarding_agent.document_catalog import (
    build_catalog_prompt_text,
    document_display_names,
    document_request_items,
    normalize_required_document_ids,
)
from agents.onboarding_agent.decision_diagnostics import (
    append_trace,
    build_trace_entry,
    diagnostics_enabled,
)
from agents.onboarding_agent.schemas import (
    DecisionExplanationResponse,
    EvidenceReference,
    ClauseReference,
    FinalPlanningResponse,
    HyDEQueryResponse,
    IssuePlanningResponse,
    QueryIntent,
    ExtractedDocumentInfo,
    SufficiencyResponse,
)


def _llm_from_config(config: RunnableConfig | None):
    configurable = (config or {}).get("configurable", {})
    return configurable.get("chat_client")


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(value, minimum)


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(value, minimum)


def _is_retryable_llm_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    retryable_names = (
        "apiconnectionerror",
        "ratelimiterror",
        "timeout",
        "serviceunavailable",
    )
    retryable_fragments = (
        "connection error",
        "timed out",
        "timeout",
        "rate limit",
        "temporarily unavailable",
        "service unavailable",
        "try again",
    )
    return any(fragment in name for fragment in retryable_names) or any(
        fragment in message for fragment in retryable_fragments
    )


def _invoke_retry_enabled() -> bool:
    strict = _env_bool("ONBOARDING_STRICT_LLM_MODE", False)
    retry_on_connection = _env_bool("ONBOARDING_LLM_RETRY_ON_CONNECTION", False)
    return strict or retry_on_connection


def _invoke_structured_with_meta(chat_client: Any, schema: Any, prompt: str, fallback: Any) -> tuple[Any, bool]:
    if not chat_client:
        if _env_bool("ONBOARDING_STRICT_LLM_MODE", False):
            raise RuntimeError("chat_client is required when ONBOARDING_STRICT_LLM_MODE=true")
        return fallback, True

    structured_llm = chat_client.with_structured_output(schema)
    strict_mode = _env_bool("ONBOARDING_STRICT_LLM_MODE", False)
    retry_enabled = _invoke_retry_enabled()
    wait_forever = _env_bool("ONBOARDING_LLM_WAIT_FOREVER", False)
    max_attempts = _env_int("ONBOARDING_LLM_MAX_ATTEMPTS", 40, minimum=1)
    max_wait_sec = _env_float("ONBOARDING_LLM_MAX_WAIT_SEC", 1800.0, minimum=1.0)
    backoff_sec = _env_float("ONBOARDING_LLM_RETRY_BACKOFF_SEC", 3.0, minimum=0.1)

    started = perf_counter()
    failed_attempts = 0

    while True:
        try:
            response = structured_llm.invoke([HumanMessage(content=prompt)])
            if response is None:
                if strict_mode:
                    raise RuntimeError("structured LLM returned None response")
                return fallback, True
            return response, False
        except Exception as exc:
            failed_attempts += 1
            elapsed_sec = perf_counter() - started
            retryable = _is_retryable_llm_error(exc)
            within_retry_budget = wait_forever or (
                failed_attempts < max_attempts and elapsed_sec < max_wait_sec
            )
            if retry_enabled and retryable and within_retry_budget:
                # Keep waiting on transient infra errors during strict eval runs.
                sleep(min(backoff_sec * (2 ** min(failed_attempts - 1, 4)), 30.0))
                continue
            if strict_mode:
                raise
            return fallback, True


def _invoke_structured_or_fallback(chat_client: Any, schema: Any, prompt: str, fallback: Any):
    response, _ = _invoke_structured_with_meta(chat_client, schema, prompt, fallback)
    return response


def _record_trace(
    state: dict,
    *,
    step_id: str,
    input_obj: Any,
    output_obj: Any,
    rationale_summary: str,
    fallback_used: bool = False,
    latency_ms: int | None = None,
    evidence_refs: list[str] | None = None,
    flags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict:
    if not diagnostics_enabled():
        return {}
    entry = build_trace_entry(
        step_id=step_id,
        input_obj=input_obj,
        output_obj=output_obj,
        rationale_summary=rationale_summary,
        evidence_refs=evidence_refs,
        fallback_used=fallback_used,
        latency_ms=latency_ms,
        flags=flags,
        meta=meta,
    )
    decision_trace, step_latency_ms, quality_flags = append_trace(state, entry)
    return {
        "decision_trace": decision_trace,
        "step_latency_ms": step_latency_ms,
        "quality_flags": quality_flags,
    }


def _default_issue_plan(denial_text: str) -> tuple[list[str], list[dict[str, object]]]:
    seed = (denial_text or "").strip()
    seed = seed[:220] if seed else "보험금 지급거절 사유 및 약관 근거"
    return (
        [
            "보험사의 지급거절 사유 분류 필요",
            "보장 요건 및 면책 조항 해당 여부 확인 필요",
        ],
        [
            {
                "intent": "거절 통지서 직접 근거 조항 확인",
                "query_seed": seed,
                "must_keywords": ["지급거절", "면책", "보장"],
                "priority": 1,
            },
            {
                "intent": "보장 요건/면책 조항 일반 확인",
                "query_seed": "보험금 지급 요건 및 면책 사유 관련 약관 조항",
                "must_keywords": ["보장", "요건", "면책"],
                "priority": 2,
            },
        ],
    )


def _normalize_query_plan(items: list[object]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, QueryIntent):
            raw_intent = item.intent
            raw_seed = item.query_seed
            raw_keywords = item.must_keywords
            raw_priority = item.priority
        elif isinstance(item, dict):
            raw_intent = str(item.get("intent", ""))
            raw_seed = str(item.get("query_seed", ""))
            raw_keywords = item.get("must_keywords", [])
            raw_priority = item.get("priority", 3)
        else:
            continue

        intent = str(raw_intent).strip() or "약관 확인"
        query_seed = str(raw_seed).strip()
        if not query_seed:
            continue
        key = query_seed.lower()
        if key in seen:
            continue
        seen.add(key)

        keywords = [str(kw).strip() for kw in (raw_keywords or []) if str(kw).strip()]
        try:
            priority = int(raw_priority)
        except (TypeError, ValueError):
            priority = 3
        priority = min(max(priority, 1), 5)
        normalized.append(
            {
                "intent": intent,
                "query_seed": query_seed[:300],
                "must_keywords": keywords[:6],
                "priority": priority,
            }
        )

    normalized.sort(key=lambda x: int(x.get("priority", 3)))
    return normalized


def parse_denial_node(state: dict, config: RunnableConfig) -> dict:
    """
    denial_file_path를 DP로 파싱해 denial_statement_text만 state에 채운다.
    읽기: denial_file_path, (선택) config.dp_client
    쓰기: denial_statement_text
    """
    file_path = (state.get("denial_file_path") or "").strip()
    if not file_path:
        return {"denial_statement_text": ""}
    try:
        denial_text = parse_document(file_path)
    except (FileNotFoundError, OSError):
        denial_text = ""
    return {"denial_statement_text": denial_text}


def issue_planning_node(state: dict, config: RunnableConfig) -> dict:
    """
    입력 텍스트 기반으로 이슈 가설과 검색 계획(query_plan)을 만든다.
    읽기: denial_statement_text / 쓰기: issue_hypotheses, query_plan
    """
    started = perf_counter()
    denial_text = str(state.get("denial_statement_text", "")).strip()
    fallback_hypotheses, fallback_query_plan = _default_issue_plan(denial_text)
    fallback = IssuePlanningResponse(
        issue_hypotheses=fallback_hypotheses,
        query_plan=[QueryIntent(**item) for item in fallback_query_plan],
    )

    chat_client = _llm_from_config(config)
    prompt = _build_issue_planning_prompt(denial_text)
    response, fallback_used = _invoke_structured_with_meta(chat_client, IssuePlanningResponse, prompt, fallback)

    issue_hypotheses = [str(item).strip() for item in (response.issue_hypotheses or []) if str(item).strip()]
    if not issue_hypotheses:
        issue_hypotheses = fallback_hypotheses

    query_plan = _normalize_query_plan(
        [item.model_dump() if hasattr(item, "model_dump") else item for item in (response.query_plan or [])]
    )
    if not query_plan:
        query_plan = fallback_query_plan

    result = {
        "issue_hypotheses": issue_hypotheses[:6],
        "query_plan": query_plan[:6],
    }
    latency_ms = int((perf_counter() - started) * 1000)
    result.update(
        _record_trace(
            state,
            step_id="issue_planning",
            input_obj={"denial_text_length": len(denial_text)},
            output_obj={
                "issue_hypotheses": result["issue_hypotheses"],
                "query_plan_count": len(result["query_plan"]),
            },
            rationale_summary="거절 통지서 기반 이슈 가설 및 검색 질의 계획 생성",
            fallback_used=fallback_used,
            latency_ms=latency_ms,
        )
    )
    return result


def _build_issue_planning_prompt(denial_text: str) -> str:
    return f"""당신은 보험 약관 검색 전략을 세우는 분석가입니다.
아래 거절 통지서 텍스트를 바탕으로 다음을 JSON으로 생성하세요.

1) issue_hypotheses: 거절 판단에 영향을 준 핵심 이슈 가설(최대 5개)
2) query_plan: 약관 검색용 질의 계획(최대 4개)
   - intent: 검색 목적
   - query_seed: 실제 검색에 넣을 핵심 문장
   - must_keywords: 결과에서 확인해야 할 핵심 키워드
   - priority: 1~5

[거절 통지서 텍스트]
{denial_text[:4000] if denial_text else "(없음)"}
"""


def _build_hyde_query_prompt(denial_text: str, intent: dict[str, object]) -> str:
    return f"""당신은 보험 약관 검색 쿼리를 만드는 도우미입니다.
아래 검색 의도와 거절 통지서 내용을 기반으로 약관 검색용 확장 쿼리 2개를 만들어 주세요.
각 쿼리는 실제 약관 문구를 찾는 데 유리하도록 구체적으로 작성하세요.

[검색 의도]
intent: {intent.get("intent", "")}
query_seed: {intent.get("query_seed", "")}
must_keywords: {intent.get("must_keywords", [])}

[거절 통지서]
{denial_text[:2500] if denial_text else "(없음)"}
"""


def _generate_hyde_queries(chat_client: Any, denial_text: str, intent: dict[str, object]) -> tuple[list[str], bool, int]:
    started = perf_counter()
    seed = str(intent.get("query_seed", "")).strip()
    fallback = HyDEQueryResponse(hyde_queries=[seed] if seed else [])
    prompt = _build_hyde_query_prompt(denial_text, intent)
    response, fallback_used = _invoke_structured_with_meta(chat_client, HyDEQueryResponse, prompt, fallback)
    queries = [str(item).strip() for item in (response.hyde_queries or []) if str(item).strip()]
    if not queries and seed:
        queries = [seed]

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query[:300])
    latency_ms = int((perf_counter() - started) * 1000)
    return deduped[:2], fallback_used, latency_ms


def retrieve_terms_node(state: dict, config: RunnableConfig) -> dict:
    """
    issue_planning 기반 다중 쿼리로 약관을 조회하고 집계한다.
    읽기: denial_statement_text, query_plan, policy_date
    쓰기: relevant_terms, retrieval_queries, retrieval_candidates
    """
    started = perf_counter()
    denial_text = str(state.get("denial_statement_text", "")).strip()
    if not denial_text:
        result = {
            "relevant_terms": "",
            "retrieval_queries": [],
            "retrieval_candidates": [],
        }
        result.update(
            _record_trace(
                state,
                step_id="retrieve_terms",
                input_obj={"denial_text_length": 0},
                output_obj={"retrieval_queries_count": 0, "retrieval_candidates_count": 0},
                rationale_summary="거절 통지서 텍스트 없음으로 약관 검색 생략",
                latency_ms=int((perf_counter() - started) * 1000),
            )
        )
        return result

    configurable = (config or {}).get("configurable", {})
    chat_client = _llm_from_config(config)
    policy_date = (
        str(state.get("policy_date") or "").strip()
        or str(configurable.get("policy_date") or "").strip()
        or "20200101"
    )
    policy_vectordb = configurable.get("policy_vectordb")

    _, fallback_query_plan = _default_issue_plan(denial_text)
    raw_query_plan = list(state.get("query_plan") or fallback_query_plan)
    query_plan = _normalize_query_plan(raw_query_plan) or fallback_query_plan

    broad_query = f"{denial_text[:260]} 지급거절 약관 근거 보장 요건 면책"
    retrieval_queries: list[str] = []
    seen_query_keys: set[str] = set()
    must_keywords_by_query: list[list[str]] = []
    hyde_fallback_count = 0
    hyde_call_count = 0
    hyde_latency_total_ms = 0

    for item in query_plan[:4]:
        hyde_queries, hyde_fallback_used, hyde_latency_ms = _generate_hyde_queries(chat_client, denial_text, item)
        hyde_call_count += 1
        hyde_latency_total_ms += hyde_latency_ms
        if hyde_fallback_used:
            hyde_fallback_count += 1
        if not hyde_queries:
            hyde_queries = [str(item.get("query_seed", "")).strip()]
        for query in hyde_queries:
            query = query.strip()
            if not query:
                continue
            key = query.lower()
            if key in seen_query_keys:
                continue
            seen_query_keys.add(key)
            retrieval_queries.append(query)
            must_keywords_by_query.append([str(kw).strip() for kw in item.get("must_keywords", []) if str(kw).strip()])

    if broad_query.strip():
        broad_key = broad_query.strip().lower()
        if broad_key not in seen_query_keys:
            seen_query_keys.add(broad_key)
            retrieval_queries.append(broad_query.strip())
            must_keywords_by_query.append([])

    retrieval_queries = retrieval_queries[:8]
    must_keywords_by_query = must_keywords_by_query[: len(retrieval_queries)]

    try:
        from tools.retrieve_terms import (
            ensure_vectordb_ready,
            format_candidates_to_terms,
            merge_candidates_rrf,
            retrieve_terms_candidates,
        )
    except Exception:
        result = {
            "relevant_terms": "",
            "retrieval_queries": retrieval_queries,
            "retrieval_candidates": [],
        }
        result.update(
            _record_trace(
                state,
                step_id="retrieve_terms",
                input_obj={
                    "query_plan_count": len(query_plan),
                    "policy_date": policy_date,
                },
                output_obj={
                    "retrieval_queries_count": len(retrieval_queries),
                    "retrieval_candidates_count": 0,
                },
                rationale_summary="약관 조회 도구 import 실패로 검색 결과 없음",
                fallback_used=True,
                latency_ms=int((perf_counter() - started) * 1000),
                meta={
                    "hyde_call_count": hyde_call_count,
                    "hyde_fallback_count": hyde_fallback_count,
                    "hyde_latency_total_ms": hyde_latency_total_ms,
                },
            )
        )
        return result

    try:
        ready_vectordb = ensure_vectordb_ready(policy_vectordb)
        candidates_by_query: list[list[dict[str, object]]] = []
        for query in retrieval_queries:
            candidates = retrieve_terms_candidates(
                query=query,
                policy_date=policy_date,
                k=5,
                vectordb=ready_vectordb,
            )
            candidates_by_query.append(candidates)

        ranked_candidates = merge_candidates_rrf(
            candidates_by_query,
            must_keywords_by_query,
            rrf_k=60,
            keyword_bonus=0.2,
            keyword_bonus_cap=0.8,
        )
        top_candidates = ranked_candidates[:5]
        relevant_terms = format_candidates_to_terms(top_candidates, top_n=5)
    except Exception:
        ranked_candidates = []
        relevant_terms = ""

    retrieval_candidates = [
        {
            "source_id": item.get("source_id", ""),
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "score": float(item.get("score", 0.0)),
            "matched_keywords": item.get("matched_keywords", []),
        }
        for item in ranked_candidates[:8]
    ]
    result = {
        "relevant_terms": relevant_terms or "",
        "retrieval_queries": retrieval_queries,
        "retrieval_candidates": retrieval_candidates,
    }
    flags: list[str] = []
    if hyde_fallback_count > 0:
        flags = ["fallback:hyde_query_generation"]
    result.update(
        _record_trace(
            state,
            step_id="retrieve_terms",
            input_obj={
                "query_plan_count": len(query_plan),
                "policy_date": policy_date,
            },
            output_obj={
                "retrieval_queries_count": len(retrieval_queries),
                "retrieval_candidates_count": len(retrieval_candidates),
                "top_source_ids": [str(item.get("source_id", "")) for item in retrieval_candidates[:5]],
            },
            rationale_summary="HyDE 확장 질의와 다중 검색 결과를 집계하여 약관 후보 생성",
            latency_ms=int((perf_counter() - started) * 1000),
            flags=flags,
            meta={
                "hyde_call_count": hyde_call_count,
                "hyde_fallback_count": hyde_fallback_count,
                "hyde_latency_total_ms": hyde_latency_total_ms,
            },
        )
    )
    return result


def final_planning_node(state: dict, config: RunnableConfig) -> dict:
    """
    약관 조회 결과를 반영해 최종 전략과 필요 서류를 확정한다.
    읽기: denial_statement_text, relevant_terms, issue_hypotheses
    쓰기: plan, required_document_ids, required_documents, final_plan_confidence
    """
    started = perf_counter()
    denial_text = str(state.get("denial_statement_text", "")).strip()
    relevant_terms = str(state.get("relevant_terms", "")).strip()
    issue_hypotheses = list(state.get("issue_hypotheses") or [])
    retrieval_candidates = list(state.get("retrieval_candidates") or [])

    fallback = FinalPlanningResponse(
        plan=(
            "현재 확보된 자료를 기준으로 보험사의 지급거절 사유를 약관과 대조해 핵심 쟁점을 정리했습니다. "
            "약관 근거와 추가 문서 근거를 함께 확인하며 필요한 보완서류를 우선 수집하는 것이 다음 단계입니다."
        ),
        required_document_ids=[
            "denial_notice",
            "medical_certificate",
        ],
        confidence="low" if not relevant_terms else "medium",
    )

    chat_client = _llm_from_config(config)
    prompt = _build_final_planning_prompt(
        denial_text=denial_text,
        relevant_terms=relevant_terms,
        issue_hypotheses=issue_hypotheses,
        retrieval_candidates=retrieval_candidates,
    )
    response, fallback_used = _invoke_structured_with_meta(chat_client, FinalPlanningResponse, prompt, fallback)
    required_document_ids = normalize_required_document_ids(
        [str(item).strip() for item in (response.required_document_ids or []) if str(item).strip()],
        max_items=3,
    )
    flags: list[str] = []
    if not required_document_ids:
        required_document_ids = list(fallback.required_document_ids)
        flags.append("fallback:required_document_ids")

    required_documents = document_display_names(required_document_ids)

    result = {
        "plan": str(response.plan).strip() or fallback.plan,
        "required_document_ids": required_document_ids,
        "required_documents": required_documents,
        "final_plan_confidence": response.confidence,
    }
    result.update(
        _record_trace(
            state,
            step_id="final_planning",
            input_obj={
                "denial_text_length": len(denial_text),
                "relevant_terms_length": len(relevant_terms),
                "issue_hypotheses_count": len(issue_hypotheses),
                "retrieval_candidates_count": len(retrieval_candidates),
            },
            output_obj={
                "required_document_ids": required_document_ids,
                "required_documents": required_documents,
                "final_plan_confidence": response.confidence,
            },
            rationale_summary="약관 검색 결과와 이슈 가설을 반영해 최종 계획 및 요청서류 선정",
            fallback_used=fallback_used,
            latency_ms=int((perf_counter() - started) * 1000),
            flags=flags,
        )
    )
    return result


def planning_node(state: dict, config: RunnableConfig) -> dict:
    """
    Backward-compatible wrapper.
    """
    return final_planning_node(state, config)


def request_additional_documents_node(state: dict, config: RunnableConfig) -> dict:
    """
    카탈로그 ID 기반 요청 서류를 interrupt로 전달하고, resume 경로를 additional_document_paths에 저장.
    읽기: required_document_ids, required_documents / 쓰기: additional_document_paths
    """
    required_ids = normalize_required_document_ids(
        [str(item).strip() for item in (state.get("required_document_ids") or []) if str(item).strip()],
        max_items=3,
    )
    required = list(state.get("required_documents") or [])
    if required_ids and not required:
        required = document_display_names(required_ids)

    if not required_ids and not required:
        return {"additional_document_paths": []}

    configurable = (config or {}).get("configurable", {})
    auto_resume = bool(configurable.get("test_auto_resume_documents", False))
    raw_mock_map = configurable.get("mock_document_map") or {}
    mock_map = raw_mock_map if isinstance(raw_mock_map, dict) else {}
    if auto_resume and required_ids:
        current_round = int(state.get("auto_resume_round", 0) or 0)
        max_rounds = int(configurable.get("test_auto_resume_max_rounds", 2) or 2)
        if current_round >= max_rounds:
            flags = [str(item).strip() for item in (state.get("quality_flags") or []) if str(item).strip()]
            if "auto_resume_exhausted" not in flags:
                flags.append("auto_resume_exhausted")
            return {
                "additional_document_paths": [],
                "auto_resume_round": current_round,
                "auto_resume_exhausted": True,
                "quality_flags": flags,
            }

        resolved_paths: list[str] = []
        missing: list[str] = []
        for doc_id in required_ids:
            mapped = str(mock_map.get(doc_id, "")).strip()
            if not mapped:
                missing.append(doc_id)
                continue
            resolved_paths.append(mapped)
        if not missing and resolved_paths:
            return {
                "additional_document_paths": resolved_paths,
                "auto_resume_round": current_round + 1,
                "auto_resume_exhausted": False,
            }

    request_items = document_request_items(required_ids)
    message_lines = [f"- {name}" for name in required] if required else [f"- {item['name']}" for item in request_items]

    payload = interrupt({
        "action": "request_additional_documents",
        "required_document_ids": required_ids,
        "required_document_items": request_items,
        "required_documents": required,
        "message": "다음 서류를 제출해 주세요:\n" + "\n".join(message_lines),
    })
    # resume 시 payload가 경로 리스트 또는 dict 등으로 올 수 있음
    if isinstance(payload, list):
        paths = [str(p).strip() for p in payload if p]
    elif isinstance(payload, dict) and "paths" in payload:
        paths = [str(p).strip() for p in payload["paths"] if p]
    elif isinstance(payload, dict) and "additional_document_paths" in payload:
        paths = [str(p).strip() for p in payload["additional_document_paths"] if p]
    else:
        paths = [str(payload).strip()] if payload else []
    return {"additional_document_paths": paths}




def _build_final_planning_prompt(
    *,
    denial_text: str,
    relevant_terms: str,
    issue_hypotheses: list[str],
    retrieval_candidates: list[dict],
) -> str:
    hypotheses_text = "\n".join(f"- {item}" for item in issue_hypotheses[:6]) or "- (없음)"
    catalog_text = build_catalog_prompt_text()
    candidate_lines = []
    for idx, item in enumerate(retrieval_candidates[:5], start=1):
        candidate_lines.append(
            f"[후보 {idx}] title={item.get('title', '')} / score={item.get('score', 0.0)} / snippet={item.get('snippet', '')}"
        )
    candidates_text = "\n".join(candidate_lines) if candidate_lines else "(없음)"

    return f"""당신은 보험금 지급 분쟁 대리·상담 경험이 있는 전문가입니다.
출력 시 거부 사유에 대한 약관·법적 근거를 전략에 반영하고, 서류는 실제 제출 가능한 구체적 명칭(퇴원요약서, 진단서, 소득증명원 등)으로 적어 주세요.

아래 [거부 명세서]는 문서 파싱(DP) 결과, [관련 보험 약관]은 RAG 검색 결과입니다.
약관이 제공되지 않은 경우에도 명세서 내용만으로 전략과 필요 서류를 제시해 주세요.

[거부 명세서]
{denial_text}

[관련 보험 약관]
{relevant_terms or "(아직 약관 DB가 연결되지 않았습니다.)"}

[이슈 가설]
{hypotheses_text}

[검색된 약관 후보 요약]
{candidates_text}

[추가 서류 카탈로그]
{catalog_text}

다음 두 가지를 구조화된 형식으로 작성해 주세요.

1) plan (전략/계획)
- 상황 요약: 피보험자, 보험 종목, 거부 사유, 금액 등 핵심 사실만 2~3문장으로 간결히 요약하세요.
- 전략/계획: 거부 사유에 대한 법·약관상 논거, 분쟁 조정·심사 청구 시 강조할 포인트, 필요 시 보완할 증거(서류)와의 연결을 구체적으로 3~5문장 이상 서술하세요.

2) required_document_ids (추가 필요 서류 ID 목록)
- 반드시 위 카탈로그의 id 중에서만 선택하세요.
- 임의의 새 서류명을 만들지 마세요.
- 최대 3개까지만 선택하세요.
- 전략적으로 꼭 필요한 문서만 선택하세요.

추가로 confidence( high / medium / low )를 함께 반환하세요.
"""

def parse_and_extract_node(state: dict, config: RunnableConfig) -> dict:
    """
    additional_document_paths 각 경로를 DP로 파싱한 뒤, 문서별로 LLM에 넣어 필요한 데이터·근거만 추출해 state에 저장.
    읽기: additional_document_paths, (선택) plan, required_documents / 쓰기: extracted_document_infos
    """
    started = perf_counter()
    paths = state.get("additional_document_paths") or []
    if not paths:
        result = {"extracted_document_infos": []}
        result.update(
            _record_trace(
                state,
                step_id="parse_and_extract",
                input_obj={"paths_count": 0},
                output_obj={"extracted_count": 0, "non_empty_evidence_count": 0},
                rationale_summary="추가 문서 경로가 없어 정보 추출 생략",
                latency_ms=int((perf_counter() - started) * 1000),
            )
        )
        return result

    configurable = (config or {}).get("configurable", {})
    chat_client = configurable.get("chat_client")
    plan = state.get("plan") or ""
    required = [str(item).strip() for item in (state.get("required_documents") or []) if str(item).strip()]

    extracted: list[dict] = []
    fallback_count = 0
    for idx, path in enumerate(paths, start=1):
        source_name = required[idx - 1] if idx - 1 < len(required) else f"문서 {idx}"
        try:
            raw_text = parse_document(path)
        except (FileNotFoundError, OSError):
            raw_text = ""
        if not raw_text:
            fallback_count += 1
            extracted.append(
                {
                    "source_name": source_name,
                    "key_data": "",
                    "evidence_or_grounds": "",
                    "helpful_notes": "(파싱 실패 또는 빈 문서)",
                }
            )
            continue
        fallback = ExtractedDocumentInfo(
            key_data=raw_text[:500],
            evidence_or_grounds="",
            helpful_notes="(LLM 추출 실패 fallback)",
        )
        prompt = _build_extract_prompt(raw_text, plan, required)
        response, fallback_used = _invoke_structured_with_meta(
            chat_client, ExtractedDocumentInfo, prompt, fallback
        )
        if fallback_used:
            fallback_count += 1
        extracted.append(
            {
                "source_name": source_name,
                **(response.model_dump() if hasattr(response, "model_dump") else fallback.model_dump()),
            }
        )
    non_empty_evidence_count = 0
    for item in extracted:
        if not isinstance(item, dict):
            continue
        key_data = str(item.get("key_data", "")).strip()
        evidence = str(item.get("evidence_or_grounds", "")).strip()
        if key_data or evidence:
            non_empty_evidence_count += 1

    result = {"extracted_document_infos": extracted}
    result.update(
        _record_trace(
            state,
            step_id="parse_and_extract",
            input_obj={
                "paths_count": len(paths),
                "required_documents_count": len(required),
            },
            output_obj={
                "extracted_count": len(extracted),
                "non_empty_evidence_count": non_empty_evidence_count,
            },
            rationale_summary="추가 문서 파싱 후 핵심 데이터/근거 추출",
            fallback_used=fallback_count > 0,
            latency_ms=int((perf_counter() - started) * 1000),
            meta={"fallback_count": fallback_count},
        )
    )
    return result


def _build_extract_prompt(doc_text: str, plan: str, required_documents: list) -> str:
    req_str = ", ".join(required_documents) if required_documents else "(없음)"
    return f"""당신은 문서 정보 추출기입니다.
아래 [문서 내용]에서만 근거를 찾아 핵심 정보를 추출하세요.

중요 규칙:
1) [참고: 분쟁 신청 계획], [요청했던 서류 목록]은 추출 우선순위 참고용입니다.
2) 위 참고 블록의 문구를 사실로 채택하거나 복사해 출력하면 안 됩니다.
3) 출력한 모든 내용은 반드시 [문서 내용]에서 직접 확인 가능해야 합니다.
4) [문서 내용]에 근거가 없으면 해당 필드는 빈 문자열로 두세요.
5) 추측/일반론/외부지식 사용 금지.

[참고: 분쟁 신청 계획]
{plan[:800] if plan else "(없음)"}

[요청했던 서류 목록]
{req_str}

[문서 내용]
{doc_text[:6000] if doc_text else "(빈 문서)"}
"""


def evaluate_sufficiency_node(state: dict, config: RunnableConfig) -> dict:
    """
    추출된 정보를 합쳐서 분쟁 신청을 위한 근거가 충분한지 판단.
    읽기: plan, required_documents, extracted_document_infos / 쓰기: evidence_sufficient
    """
    started = perf_counter()
    infos = state.get("extracted_document_infos") or []
    if not infos:
        result = {"evidence_sufficient": False}
        result.update(
            _record_trace(
                state,
                step_id="evaluate_sufficiency",
                input_obj={"extracted_document_count": 0},
                output_obj={"evidence_sufficient": False},
                rationale_summary="추출 문서 정보가 없어 근거 부족으로 판정",
                latency_ms=int((perf_counter() - started) * 1000),
            )
        )
        return result

    configurable = (config or {}).get("configurable", {})
    chat_client = configurable.get("chat_client")

    plan = state.get("plan") or ""
    required = state.get("required_documents") or []
    prompt = _build_sufficiency_prompt(plan, required, infos)
    flags: list[str] = []
    fallback = SufficiencyResponse(sufficient=False)
    response, fallback_used = _invoke_structured_with_meta(chat_client, SufficiencyResponse, prompt, fallback)
    sufficient = bool(response.sufficient)

    has_evidence = False
    for item in infos:
        if not isinstance(item, dict):
            continue
        if str(item.get("key_data", "")).strip() or str(item.get("evidence_or_grounds", "")).strip():
            has_evidence = True
            break
    if sufficient and not has_evidence:
        flags.append("inconsistency:sufficiency_without_evidence")

    result = {"evidence_sufficient": sufficient}
    result.update(
        _record_trace(
            state,
            step_id="evaluate_sufficiency",
            input_obj={
                "extracted_document_count": len(infos),
                "required_documents_count": len(required),
            },
            output_obj={"evidence_sufficient": sufficient},
            rationale_summary="추출된 문서 근거가 분쟁 신청에 충분한지 판정",
            fallback_used=fallback_used,
            latency_ms=int((perf_counter() - started) * 1000),
            flags=flags,
            meta={"has_non_empty_evidence": has_evidence},
        )
    )
    return result


def _build_sufficiency_prompt(plan: str, required_documents: list, extracted_infos: list[dict]) -> str:
    plan_text = plan.strip() if plan else "(없음)"
    req_str = "\n".join(f"- {r}" for r in (required_documents or [])) or "- (없음)"

    docs_str = ""
    for i, info in enumerate(extracted_infos, 1):
        docs_str += f"\n[문서 {i}]\n"
        docs_str += f"- 핵심 데이터: {str(info.get('key_data', '')).strip() or '(없음)'}\n"
        docs_str += f"- 근거: {str(info.get('evidence_or_grounds', '')).strip() or '(없음)'}\n"
        docs_str += f"- 기타: {str(info.get('helpful_notes', '')).strip() or '(없음)'}\n"

    return f"""당신은 보험 분쟁 신청 준비도를 점검하는 심사자입니다.
아래 정보를 반드시 모두 읽고, 지금 상태에서 분쟁 신청을 진행하기에 근거가 충분한지 판단하세요.

[분쟁 신청 계획]
{plan_text}

[요청했던 추가 서류]
{req_str}

[문서별 추출 정보]
{docs_str if docs_str else "(문서 정보 없음)"}

판정 기준:
1) 계획(plan)의 핵심 주장에 대응되는 사실/근거가 문서 추출 정보에 실제로 존재하는가
2) 요청했던 추가 서류 항목이 실질적으로 충족되었거나, 그에 준하는 근거가 있는가
3) 핵심 근거가 비어 있거나 모순/불명확하면 insufficient로 본다

출력 규칙:
- 충분하면 sufficient=true
- 부족하면 sufficient=false
- 반드시 불리언 하나만 판단하고, 추측으로 true를 주지 마세요.
"""


def _mask_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return text

    def _mask_token(token: str) -> str:
        token = token.strip()
        if not token:
            return token
        if re.fullmatch(r"[가-힣]{2,4}", token):
            return token[0] + ("*" * (len(token) - 1))
        if re.fullmatch(r"[A-Za-z][A-Za-z'.-]{1,}", token):
            return token[0] + ("*" * (len(token) - 1))
        return token

    parts = re.split(r"(\s+)", text)
    return "".join(_mask_token(part) if idx % 2 == 0 else part for idx, part in enumerate(parts))


def _mask_digits_keep_tail(text: str, *, keep_tail: int = 2) -> str:
    chars = list(str(text or ""))
    digit_positions = [idx for idx, ch in enumerate(chars) if ch.isdigit()]
    if not digit_positions:
        return "".join(chars)
    if keep_tail <= 0:
        keep_set: set[int] = set()
    else:
        keep_set = set(digit_positions[-keep_tail:])
    for idx in digit_positions:
        if idx not in keep_set:
            chars[idx] = "*"
    return "".join(chars)


def _mask_birth_date(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"\b(\d{4})[-./](\d{2})[-./](\d{2})\b", r"\1-**-**", text)
    text = re.sub(r"\b(\d{4})(\d{2})(\d{2})\b", r"\1****", text)
    return text


def _mask_email_in_text(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        local = match.group("local")
        domain = match.group("domain")
        masked_local = local[0] + ("*" * max(len(local) - 1, 1))
        domain_parts = domain.split(".")
        if domain_parts:
            head = domain_parts[0]
            domain_parts[0] = head[0] + ("*" * max(len(head) - 1, 1)) if head else "***"
        return f"{masked_local}@{'.'.join(domain_parts)}"

    return re.sub(
        r"(?P<local>[A-Za-z0-9._%+-]+)@(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        _repl,
        str(text or ""),
    )


def _mask_phone_in_text(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}-****-**{match.group(3)[-2:]}"

    return re.sub(r"\b(01[0-9]|0[2-9][0-9]?)-?(\d{3,4})-?(\d{4})\b", _repl, str(text or ""))


def _mask_pii_text(text: str) -> str:
    masked = str(text or "")
    if not masked:
        return masked

    # Label-aware masking first.
    label_pattern = re.compile(
        r"(?P<label>"
        r"(?:수신|성명|이름|환자명|피보험자|계약자|수익자|작성자|대상자|생년월일|주민등록번호|"
        r"계약번호|증권번호|전화번호|휴대전화|연락처|이메일|주소|계좌번호)\s*[:：]\s*)"
        r"(?P<value>[^\n]+)"
    )

    def _label_repl(match: re.Match[str]) -> str:
        label = match.group("label")
        value = match.group("value").strip()
        normalized_label = label.replace(" ", "")
        if any(key in normalized_label for key in ("성명", "이름", "환자명", "피보험자", "계약자", "수익자", "작성자", "대상자", "수신")):
            masked_value = _mask_name(value)
        elif "생년월일" in normalized_label:
            masked_value = _mask_birth_date(value)
        elif "주민등록번호" in normalized_label:
            masked_value = re.sub(r"(\d{6})[- ]?(\d{7})", r"\1-*******", value)
        elif any(key in normalized_label for key in ("전화번호", "휴대전화", "연락처")):
            masked_value = _mask_phone_in_text(value)
        elif "이메일" in normalized_label:
            masked_value = _mask_email_in_text(value)
        elif any(key in normalized_label for key in ("계약번호", "증권번호", "계좌번호")):
            masked_value = _mask_digits_keep_tail(value, keep_tail=2)
        elif "주소" in normalized_label:
            core = value[:6]
            masked_value = f"{core}***" if value else value
        else:
            masked_value = value
        return f"{label}{masked_value}"

    masked = label_pattern.sub(_label_repl, masked)

    # Unlabeled patterns.
    masked = re.sub(r"\b(\d{6})[- ]?([1-4]\d{6})\b", r"\1-*******", masked)
    masked = _mask_phone_in_text(masked)
    masked = _mask_email_in_text(masked)
    masked = re.sub(r"\b([가-힣]{2,4})(?=\s*고객님\b)", lambda m: _mask_name(m.group(1)), masked)
    return masked


def _to_plain_language(text: str) -> str:
    simplified = str(text or "").strip()
    if not simplified:
        return simplified
    replacements = [
        ("면책", "보상 제외(면책)"),
        ("보장 요건", "보험금을 받기 위한 조건"),
        ("자기부담금", "본인이 내야 하는 금액(자기부담금)"),
        ("비급여", "건강보험 미적용 항목(비급여)"),
        ("인과관계", "원인과 결과의 연결"),
        ("지급 불가", "보험금 지급이 어려움"),
        ("지급거절", "보험금 지급 거절"),
    ]
    for source, target in replacements:
        simplified = simplified.replace(source, target)
    simplified = re.sub(r"\s+", " ", simplified)
    return simplified


def _compose_user_friendly_explanation(
    *,
    user_situation: str,
    insurer_claim: str,
    conclusion_reason: str,
    policy_clauses: list[dict],
    document_evidence: list[dict],
    llm_plain_explanation: str = "",
) -> str:
    lines: list[str] = []
    lines.append("1) 지금 상황")
    lines.append(f"- {_to_plain_language(user_situation) or '(요약 정보 없음)'}")

    lines.append("2) 보험사 판단")
    lines.append(f"- {_to_plain_language(insurer_claim) or '(보험사 주장 정보 없음)'}")

    lines.append("3) 약관 근거")
    if policy_clauses:
        for idx, clause in enumerate(policy_clauses[:3], start=1):
            title = str(clause.get("title", "")).strip() or "(조항 제목 없음)"
            snippet = str(clause.get("snippet", "")).strip() or "(요약 없음)"
            lines.append(f"- [{idx}] {title}: {_to_plain_language(snippet)}")
    else:
        lines.append("- 약관 근거가 충분히 확인되지 않았습니다.")

    lines.append("4) 문서에서 확인된 사실")
    if document_evidence:
        for item in document_evidence[:3]:
            src = int(item.get("source_index", 0) or 0)
            source_name = str(item.get("source_name", "")).strip() or f"문서 {src}"
            key_data = str(item.get("key_data", "")).strip() or "(핵심 데이터 없음)"
            evidence = str(item.get("evidence", "")).strip() or "(근거 없음)"
            lines.append(
                f"- [{source_name}] 핵심={_to_plain_language(key_data)} / 근거={_to_plain_language(evidence)}"
            )
    else:
        lines.append("- 추가 문서 근거가 충분하지 않습니다.")

    lines.append("5) 한 줄 결론")
    lines.append(f"- {_to_plain_language(conclusion_reason) or '(결론 근거 없음)'}")

    lines.append("6) 다음 단계")
    lines.append("- 약관 조항과 제출 문서를 하나씩 맞춰 보면서 이의신청 포인트를 정리하세요.")

    if llm_plain_explanation.strip():
        lines.append("7) 쉬운 설명")
        lines.append(f"- {_to_plain_language(llm_plain_explanation.strip())}")

    lines.append("8) 안내")
    lines.append("- 이 설명은 AI가 만든 참고용 답변입니다.")
    lines.append("- 최종 판단과 결정의 책임은 사용자와 담당 전문가에게 있습니다.")

    return "\n".join(lines)


def _mask_decision_summary(summary: dict) -> dict:
    masked_summary = dict(summary or {})
    for key in ("user_situation", "insurer_claim", "conclusion_reason"):
        masked_summary[key] = _mask_pii_text(str(masked_summary.get(key, "")))

    policy_clauses = []
    for item in masked_summary.get("policy_clauses", []) or []:
        policy_clauses.append(
            {
                "title": _mask_pii_text(str(item.get("title", ""))),
                "snippet": _mask_pii_text(str(item.get("snippet", ""))),
            }
        )
    masked_summary["policy_clauses"] = policy_clauses

    document_evidence = []
    for item in masked_summary.get("document_evidence", []) or []:
        source_index = int(item.get("source_index", 1) or 1)
        source_name = str(item.get("source_name", "")).strip()
        document_evidence.append(
            {
                "source_index": source_index,
                "source_name": _mask_pii_text(source_name),
                "key_data": _mask_pii_text(str(item.get("key_data", ""))),
                "evidence": _mask_pii_text(str(item.get("evidence", ""))),
            }
        )
    masked_summary["document_evidence"] = document_evidence
    return masked_summary


def _extract_clause_blocks(relevant_terms: str, limit: int = 3) -> list[dict[str, str]]:
    text = (relevant_terms or "").strip()
    if not text:
        return []

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("[Section:"):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)

    items: list[dict[str, str]] = []
    for block in blocks[:limit]:
        title = block[0].strip()
        body_lines = [line.strip() for line in block[1:] if line.strip()]
        snippet = " ".join(body_lines)[:180] if body_lines else "(본문 없음)"
        items.append({"title": title, "snippet": snippet})
    return items


def _extract_document_evidence(
    infos: list[dict],
    *,
    required_documents: list[str] | None = None,
    limit: int = 3,
) -> list[dict[str, str | int]]:
    required = [str(item).strip() for item in (required_documents or []) if str(item).strip()]
    extracted: list[dict[str, str | int]] = []
    for idx, info in enumerate(infos[:limit], start=1):
        source_name = str(info.get("source_name", "")).strip()
        if not source_name and idx - 1 < len(required):
            source_name = required[idx - 1]
        key_data = str(info.get("key_data", "")).strip()
        evidence = str(info.get("evidence_or_grounds", "")).strip()
        helpful = str(info.get("helpful_notes", "")).strip()
        extracted.append(
            {
                "source_index": idx,
                "source_name": source_name or f"문서 {idx}",
                "key_data": key_data or helpful or "(핵심 데이터 없음)",
                "evidence": evidence or helpful or "(근거 문구 없음)",
            }
        )
    return extracted


def _build_decision_explanation_prompt(
    denial_text: str,
    relevant_terms: str,
    plan: str,
    required_documents: list[str],
    extracted_infos: list[dict],
) -> str:
    required_text = "\n".join(f"- {item}" for item in required_documents[:5]) or "- (없음)"
    evidence_lines = []
    for idx, info in enumerate(extracted_infos[:5], start=1):
        source_name = str(info.get("source_name", "")).strip() or (
            required_documents[idx - 1] if idx - 1 < len(required_documents) else f"문서 {idx}"
        )
        evidence_lines.append(
            f"[{source_name}] 핵심={str(info.get('key_data', '')).strip()} / "
            f"근거={str(info.get('evidence_or_grounds', '')).strip()} / "
            f"기타={str(info.get('helpful_notes', '')).strip()}"
        )
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "(없음)"

    return f"""당신은 보험 가입자가 이해하기 쉽게 현재 상황을 설명하는 어시스턴트입니다.
아래 정보를 바탕으로 '보험사가 왜 지급하지 않는 결론을 냈는지'를 쉬운 한국어로 설명하세요.

[거절 통지서 텍스트]
{denial_text[:3000] if denial_text else "(없음)"}

[약관 검색 결과]
{relevant_terms[:5000] if relevant_terms else "(없음)"}

[현재 전략 요약]
{plan[:1200] if plan else "(없음)"}

[요청 서류 목록]
{required_text}

[추가 문서에서 추출한 정보]
{evidence_text}

작성 규칙:
1) 보험사 주장(insurer_claim)과 사용자 상황(user_situation)을 먼저 분리해 적으세요.
2) 약관 근거(policy_clauses)는 최대 3개만 고르고, 각 항목은 제목+짧은 요약(snippet)으로 작성하세요.
3) 문서 근거(document_evidence)는 최대 3개만 고르고, source_index는 1부터 시작하세요.
   - source_name에는 문서 카탈로그 이름(예: 진료비 세부산정내역서/영수증)을 우선적으로 쓰세요.
4) plain_explanation은 한 문장 길이를 짧게 쓰고, 어려운 용어는 괄호로 쉬운 뜻을 붙여 설명하세요.
5) plain_explanation 안에 반드시 '약관 근거'를 명시하고, 실제 조항 제목을 1개 이상 인용하세요.
6) 이름/계약번호/전화번호/이메일 등 개인정보는 원문 그대로 쓰지 말고 마스킹 형태로 표현하세요.
7) 결론은 '현재 확보된 자료 기준'이라는 전제를 포함하세요.
8) 마지막에 '이 설명은 참고용이며 최종 결정 책임은 사람에게 있다'는 취지의 안내를 한 줄 포함하세요.
9) 근거가 부족하면 confidence를 low로 두세요.
"""


def _build_fallback_decision_explanation(state: dict) -> tuple[dict, str]:
    denial_text = str(state.get("denial_statement_text", "")).strip()
    relevant_terms = str(state.get("relevant_terms", "")).strip()
    plan = str(state.get("plan", "")).strip()
    required_documents = [str(item).strip() for item in (state.get("required_documents") or []) if str(item).strip()]
    extracted_infos = list(state.get("extracted_document_infos") or [])

    clauses = _extract_clause_blocks(relevant_terms, limit=3)
    doc_evidence = _extract_document_evidence(
        extracted_infos,
        required_documents=required_documents,
        limit=3,
    )

    user_situation = (
        "제출된 거절 통지서와 추가 문서를 기준으로 현재 청구 상황을 정리했습니다."
        if denial_text
        else "거절 통지서 텍스트가 충분하지 않아 제한된 정보로 현재 상황을 정리했습니다."
    )
    insurer_claim = "보험사는 약관상 보장 요건에 맞지 않거나 면책 사유에 해당한다고 판단한 것으로 보입니다."
    if denial_text:
        insurer_claim = f"보험사는 통지서 내용에 근거해 지급 거절을 통보했습니다. ({denial_text[:120]})"

    conclusion_reason = (
        "약관 조항 해석과 추가 문서 근거를 종합하면, 보험사는 현재 자료 기준으로 지급 불가 방향 결론을 낸 상태입니다."
    )

    summary = {
        "user_situation": user_situation,
        "insurer_claim": insurer_claim,
        "policy_clauses": clauses,
        "document_evidence": doc_evidence,
        "conclusion_reason": conclusion_reason,
        "confidence": "medium" if clauses or doc_evidence else "low",
    }
    masked_summary = _mask_decision_summary(summary)
    explanation = _compose_user_friendly_explanation(
        user_situation=str(masked_summary.get("user_situation", "")),
        insurer_claim=str(masked_summary.get("insurer_claim", "")),
        conclusion_reason=str(masked_summary.get("conclusion_reason", "")),
        policy_clauses=list(masked_summary.get("policy_clauses", []) or []),
        document_evidence=list(masked_summary.get("document_evidence", []) or []),
    )
    return masked_summary, _mask_pii_text(explanation)


def explain_decision_node(state: dict, config: RunnableConfig) -> dict:
    """
    evidence_sufficient=True일 때 사용자 이해용 설명문을 생성한다.
    읽기: denial_statement_text, relevant_terms, plan, extracted_document_infos
    쓰기: decision_summary, decision_explanation
    """
    if not state.get("evidence_sufficient") and not state.get("auto_resume_exhausted"):
        return {}

    fallback_summary, fallback_explanation = _build_fallback_decision_explanation(state)

    configurable = (config or {}).get("configurable", {})
    chat_client = configurable.get("chat_client")
    strict_mode = _env_bool("ONBOARDING_STRICT_LLM_MODE", False)
    if not chat_client:
        if strict_mode:
            raise RuntimeError("chat_client is required when ONBOARDING_STRICT_LLM_MODE=true")
        return {
            "decision_summary": fallback_summary,
            "decision_explanation": fallback_explanation,
        }

    denial_text = str(state.get("denial_statement_text", "")).strip()
    relevant_terms = str(state.get("relevant_terms", "")).strip()
    plan = str(state.get("plan", "")).strip()
    required_documents = list(state.get("required_documents") or [])
    extracted_infos = list(state.get("extracted_document_infos") or [])

    try:
        structured_llm = chat_client.with_structured_output(DecisionExplanationResponse)
        prompt = _build_decision_explanation_prompt(
            denial_text=denial_text,
            relevant_terms=relevant_terms,
            plan=plan,
            required_documents=required_documents,
            extracted_infos=extracted_infos,
        )
        response: DecisionExplanationResponse = structured_llm.invoke([HumanMessage(content=prompt)])

        policy_clauses = [
            ClauseReference(title=item.title, snippet=item.snippet).model_dump()
            for item in response.policy_clauses[:3]
        ]
        document_evidence = [
            EvidenceReference(
                source_index=item.source_index,
                key_data=item.key_data,
                evidence=item.evidence,
            ).model_dump()
            for item in response.document_evidence[:3]
        ]
        for item in document_evidence:
            source_index = int(item.get("source_index", 0) or 0)
            source_name = str(item.get("source_name", "")).strip()
            if source_name:
                continue
            if source_index > 0 and source_index - 1 < len(extracted_infos):
                source_name = str(extracted_infos[source_index - 1].get("source_name", "")).strip()
            if not source_name and source_index > 0 and source_index - 1 < len(required_documents):
                source_name = str(required_documents[source_index - 1]).strip()
            item["source_name"] = source_name or f"문서 {source_index or 1}"

        decision_summary_raw = {
            "user_situation": response.user_situation,
            "insurer_claim": response.insurer_claim,
            "policy_clauses": policy_clauses,
            "document_evidence": document_evidence,
            "conclusion_reason": response.conclusion_reason,
            "confidence": response.confidence,
        }
        decision_summary = _mask_decision_summary(decision_summary_raw)
        decision_explanation = _compose_user_friendly_explanation(
            user_situation=str(decision_summary.get("user_situation", "")),
            insurer_claim=str(decision_summary.get("insurer_claim", "")),
            conclusion_reason=str(decision_summary.get("conclusion_reason", "")),
            policy_clauses=list(decision_summary.get("policy_clauses", []) or []),
            document_evidence=list(decision_summary.get("document_evidence", []) or []),
            llm_plain_explanation=str(response.plain_explanation or "").strip(),
        ).strip()
        if not decision_explanation:
            decision_explanation = fallback_explanation
        decision_explanation = _mask_pii_text(decision_explanation)
        return {
            "decision_summary": decision_summary,
            "decision_explanation": decision_explanation,
        }
    except Exception:
        if strict_mode:
            raise
        return {
            "decision_summary": fallback_summary,
            "decision_explanation": fallback_explanation,
        }
