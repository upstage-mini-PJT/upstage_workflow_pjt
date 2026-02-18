"""
onboarding_agent용 그래프 노드 정의.

- temp_builder.py: 여기서 노드를 import 해서 테스트용 그래프 구성
- workflow/builder.py: 여기서 노드를 import 해서 메인 그래프에 add_node
"""

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from tools.document_parser import parse_document

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


def _invoke_structured_or_fallback(chat_client: Any, schema: Any, prompt: str, fallback: Any):
    if not chat_client:
        return fallback
    try:
        structured_llm = chat_client.with_structured_output(schema)
        return structured_llm.invoke([HumanMessage(content=prompt)])
    except Exception:
        return fallback


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
    denial_text = str(state.get("denial_statement_text", "")).strip()
    fallback_hypotheses, fallback_query_plan = _default_issue_plan(denial_text)
    fallback = IssuePlanningResponse(
        issue_hypotheses=fallback_hypotheses,
        query_plan=[QueryIntent(**item) for item in fallback_query_plan],
    )

    chat_client = _llm_from_config(config)
    prompt = _build_issue_planning_prompt(denial_text)
    response = _invoke_structured_or_fallback(chat_client, IssuePlanningResponse, prompt, fallback)

    issue_hypotheses = [str(item).strip() for item in (response.issue_hypotheses or []) if str(item).strip()]
    if not issue_hypotheses:
        issue_hypotheses = fallback_hypotheses

    query_plan = _normalize_query_plan(
        [item.model_dump() if hasattr(item, "model_dump") else item for item in (response.query_plan or [])]
    )
    if not query_plan:
        query_plan = fallback_query_plan

    return {
        "issue_hypotheses": issue_hypotheses[:6],
        "query_plan": query_plan[:6],
    }


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


def _generate_hyde_queries(chat_client: Any, denial_text: str, intent: dict[str, object]) -> list[str]:
    seed = str(intent.get("query_seed", "")).strip()
    fallback = HyDEQueryResponse(hyde_queries=[seed] if seed else [])
    prompt = _build_hyde_query_prompt(denial_text, intent)
    response = _invoke_structured_or_fallback(chat_client, HyDEQueryResponse, prompt, fallback)
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
    return deduped[:2]


def retrieve_terms_node(state: dict, config: RunnableConfig) -> dict:
    """
    issue_planning 기반 다중 쿼리로 약관을 조회하고 집계한다.
    읽기: denial_statement_text, query_plan, policy_date
    쓰기: relevant_terms, retrieval_queries, retrieval_candidates
    """
    denial_text = str(state.get("denial_statement_text", "")).strip()
    if not denial_text:
        return {
            "relevant_terms": "",
            "retrieval_queries": [],
            "retrieval_candidates": [],
        }

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

    for item in query_plan[:4]:
        hyde_queries = _generate_hyde_queries(chat_client, denial_text, item)
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
        return {
            "relevant_terms": "",
            "retrieval_queries": retrieval_queries,
            "retrieval_candidates": [],
        }

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
    return {
        "relevant_terms": relevant_terms or "",
        "retrieval_queries": retrieval_queries,
        "retrieval_candidates": retrieval_candidates,
    }


def final_planning_node(state: dict, config: RunnableConfig) -> dict:
    """
    약관 조회 결과를 반영해 최종 전략과 필요 서류를 확정한다.
    읽기: denial_statement_text, relevant_terms, issue_hypotheses
    쓰기: plan, required_documents, final_plan_confidence
    """
    denial_text = str(state.get("denial_statement_text", "")).strip()
    relevant_terms = str(state.get("relevant_terms", "")).strip()
    issue_hypotheses = list(state.get("issue_hypotheses") or [])
    retrieval_candidates = list(state.get("retrieval_candidates") or [])

    fallback = FinalPlanningResponse(
        plan=(
            "현재 확보된 자료를 기준으로 보험사의 지급거절 사유를 약관과 대조해 핵심 쟁점을 정리했습니다. "
            "약관 근거와 추가 문서 근거를 함께 확인하며 필요한 보완서류를 우선 수집하는 것이 다음 단계입니다."
        ),
        required_documents=[
            "지급거절 통지서 원문 (거절 사유 원문 확인)",
            "진단서/소견서 (질병·치료 사실 확인)",
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
    response = _invoke_structured_or_fallback(chat_client, FinalPlanningResponse, prompt, fallback)
    required_documents = [str(item).strip() for item in (response.required_documents or []) if str(item).strip()]
    if not required_documents:
        required_documents = list(fallback.required_documents)

    return {
        "plan": str(response.plan).strip() or fallback.plan,
        "required_documents": required_documents[:5],
        "final_plan_confidence": response.confidence,
    }


def planning_node(state: dict, config: RunnableConfig) -> dict:
    """
    Backward-compatible wrapper.
    """
    return final_planning_node(state, config)


def request_additional_documents_node(state: dict, config: RunnableConfig) -> dict:
    """
    required_documents로 interrupt 후, Command(resume)로 받은 경로를 additional_document_paths에 저장.
    읽기: required_documents / 쓰기: additional_document_paths
    """
    required = state.get("required_documents") or []
    if not required:
        return {"additional_document_paths": []}

    payload = interrupt({
        "action": "request_additional_documents",
        "required_documents": required,
        "message": "다음 서류를 제출해 주세요: " + ", ".join(required),
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

다음 두 가지를 구조화된 형식으로 작성해 주세요.

1) plan (전략/계획)
- 상황 요약: 피보험자, 보험 종목, 거부 사유, 금액 등 핵심 사실만 2~3문장으로 간결히 요약하세요.
- 전략/계획: 거부 사유에 대한 법·약관상 논거, 분쟁 조정·심사 청구 시 강조할 포인트, 필요 시 보완할 증거(서류)와의 연결을 구체적으로 3~5문장 이상 서술하세요.

2) required_documents (추가 필요 서류)
- 전략적인 분쟁 신청을 위해 "추가로" 제출이 필요한 서류만 나열하세요. 이미 거부 명세서에 포함된 자료는 제외합니다.
- 위 전략에서 필요하다고 판단한 서류만 최대 3개, 각 항목은 "서류명 (목적/키워드)" 형식으로 적고, 서류명은 퇴원요약서·진단서·소득증명원 등 실제 제출 가능한 구체적 명칭을 사용하세요.

추가로 confidence( high / medium / low )를 함께 반환하세요.
"""

def parse_and_extract_node(state: dict, config: RunnableConfig) -> dict:
    """
    additional_document_paths 각 경로를 DP로 파싱한 뒤, 문서별로 LLM에 넣어 필요한 데이터·근거만 추출해 state에 저장.
    읽기: additional_document_paths, (선택) plan, required_documents / 쓰기: extracted_document_infos
    """
    paths = state.get("additional_document_paths") or []
    if not paths:
        return {"extracted_document_infos": []}

    configurable = (config or {}).get("configurable", {})
    chat_client = configurable.get("chat_client")
    plan = state.get("plan") or ""
    required = state.get("required_documents") or []

    extracted: list[dict] = []
    for path in paths:
        try:
            raw_text = parse_document(path)
        except (FileNotFoundError, OSError):
            raw_text = ""
        if not raw_text:
            extracted.append(
                {"key_data": "", "evidence_or_grounds": "", "helpful_notes": "(파싱 실패 또는 빈 문서)"}
            )
            continue
        if not chat_client:
            extracted.append(
                {"key_data": raw_text[:500], "evidence_or_grounds": "", "helpful_notes": ""}
            )
            continue
        structured_llm = chat_client.with_structured_output(ExtractedDocumentInfo)
        prompt = _build_extract_prompt(raw_text, plan, required)
        response: ExtractedDocumentInfo = structured_llm.invoke([HumanMessage(content=prompt)])
        extracted.append(response.model_dump())
    return {"extracted_document_infos": extracted}


def _build_extract_prompt(doc_text: str, plan: str, required_documents: list) -> str:
    req_str = ", ".join(required_documents) if required_documents else "(없음)"
    return f"""아래는 추가 제출 서류의 문서 내용입니다. 분쟁 신청에 필요한 핵심 데이터, 근거가 되는 문구, 기타 도움이 되는 정보만 추출해 주세요.

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
    infos = state.get("extracted_document_infos") or []
    if not infos:
        return {"evidence_sufficient": False}

    configurable = (config or {}).get("configurable", {})
    chat_client = configurable.get("chat_client")
    if not chat_client:
        return {"evidence_sufficient": False}

    plan = state.get("plan") or ""
    required = state.get("required_documents") or []
    prompt = _build_sufficiency_prompt(plan, required, infos)
    structured_llm = chat_client.with_structured_output(SufficiencyResponse)
    response: SufficiencyResponse = structured_llm.invoke([HumanMessage(content=prompt)])
    return {"evidence_sufficient": response.sufficient}


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


def _extract_document_evidence(infos: list[dict], limit: int = 3) -> list[dict[str, str | int]]:
    extracted: list[dict[str, str | int]] = []
    for idx, info in enumerate(infos[:limit], start=1):
        key_data = str(info.get("key_data", "")).strip()
        evidence = str(info.get("evidence_or_grounds", "")).strip()
        helpful = str(info.get("helpful_notes", "")).strip()
        extracted.append(
            {
                "source_index": idx,
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
        evidence_lines.append(
            f"[문서 {idx}] 핵심={str(info.get('key_data', '')).strip()} / "
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
4) plain_explanation은 6~10문장으로, 쉬운 말로 작성하세요.
5) 결론은 '현재 확보된 자료 기준'이라는 전제를 포함하세요.
6) 근거가 부족하면 confidence를 low로 두세요.
"""


def _build_fallback_decision_explanation(state: dict) -> tuple[dict, str]:
    denial_text = str(state.get("denial_statement_text", "")).strip()
    relevant_terms = str(state.get("relevant_terms", "")).strip()
    plan = str(state.get("plan", "")).strip()
    extracted_infos = list(state.get("extracted_document_infos") or [])

    clauses = _extract_clause_blocks(relevant_terms, limit=3)
    doc_evidence = _extract_document_evidence(extracted_infos, limit=3)

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

    lines = [
        "현재까지 제출된 자료를 기준으로 상황을 쉽게 정리해드리겠습니다.",
        user_situation,
        insurer_claim,
    ]
    if clauses:
        lines.append(f"보험사는 약관 조항({clauses[0]['title']})을 근거로 지급 요건 미충족을 주장할 가능성이 큽니다.")
    if doc_evidence:
        lines.append(
            f"추가 문서에서도 {doc_evidence[0]['key_data']} 같은 정보가 확인되어 보험사 판단 근거로 사용될 수 있습니다."
        )
    lines.append(conclusion_reason)
    lines.append("즉, 현재 확보된 자료 기준으로는 보험사 쪽 결론이 지급하지 않는 방향으로 정리된 상태입니다.")

    summary = {
        "user_situation": user_situation,
        "insurer_claim": insurer_claim,
        "policy_clauses": clauses,
        "document_evidence": doc_evidence,
        "conclusion_reason": conclusion_reason,
        "confidence": "medium" if clauses or doc_evidence else "low",
    }
    return summary, " ".join(lines)


def explain_decision_node(state: dict, config: RunnableConfig) -> dict:
    """
    evidence_sufficient=True일 때 사용자 이해용 설명문을 생성한다.
    읽기: denial_statement_text, relevant_terms, plan, extracted_document_infos
    쓰기: decision_summary, decision_explanation
    """
    if not state.get("evidence_sufficient"):
        return {}

    fallback_summary, fallback_explanation = _build_fallback_decision_explanation(state)

    configurable = (config or {}).get("configurable", {})
    chat_client = configurable.get("chat_client")
    if not chat_client:
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

        decision_summary = {
            "user_situation": response.user_situation,
            "insurer_claim": response.insurer_claim,
            "policy_clauses": policy_clauses,
            "document_evidence": document_evidence,
            "conclusion_reason": response.conclusion_reason,
            "confidence": response.confidence,
        }
        decision_explanation = (response.plain_explanation or "").strip() or fallback_explanation
        return {
            "decision_summary": decision_summary,
            "decision_explanation": decision_explanation,
        }
    except Exception:
        return {
            "decision_summary": fallback_summary,
            "decision_explanation": fallback_explanation,
        }
