"""
onboarding_agent용 그래프 노드 정의.

- temp_builder.py: 여기서 노드를 import 해서 테스트용 그래프 구성
- workflow/builder.py: 여기서 노드를 import 해서 메인 그래프에 add_node
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from tools.document_parser import parse_document

from agents.onboarding_agent.schemas import (
    DecisionExplanationResponse,
    EvidenceReference,
    ClauseReference,
    PlanningResponse,
    ExtractedDocumentInfo,
    SufficiencyResponse,
)

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


def retrieve_terms_node(state: dict, config: RunnableConfig) -> dict:
    """
    denial_statement_text로 RAG 조회 후 relevant_terms만 state에 채운다.
    읽기: denial_statement_text, (선택) policy_date
    쓰기: relevant_terms
    """
    denial_text = (state.get("denial_statement_text") or "").strip()
    if not denial_text:
        return {"relevant_terms": ""}

    configurable = (config or {}).get("configurable", {})
    policy_date = (
        str(state.get("policy_date") or "").strip()
        or str(configurable.get("policy_date") or "").strip()
        or "20200101"
    )
    policy_vectordb = configurable.get("policy_vectordb")

    try:
        from tools.retrieve_terms import ensure_vectordb_ready, retrieve_terms_text
    except Exception:
        return {"relevant_terms": ""}

    try:
        ready_vectordb = ensure_vectordb_ready(policy_vectordb)
        relevant_terms = retrieve_terms_text(
            query=denial_text[:3000],
            policy_date=policy_date,
            k=5,
            vectordb=ready_vectordb,
        )
    except Exception:
        relevant_terms = ""

    return {"relevant_terms": relevant_terms or ""}


def planning_node(state: dict, config: RunnableConfig) -> dict:
    """
    denial_statement_text와 relevant_terms를 바탕으로 LLM으로 전략·필요 서류만 생성한다.
    DP/RAG 호출 없음. 읽기: denial_statement_text, relevant_terms / 쓰기: plan, required_documents
    """
    denial_text = state.get("denial_statement_text") or ""
    relevant_terms = state.get("relevant_terms") or ""

    configurable = (config or {}).get("configurable", {})
    chat_client = configurable.get("chat_client")
    if not chat_client:
        return {"plan": "", "required_documents": []}

    structured_llm = chat_client.with_structured_output(PlanningResponse)
    prompt = _build_planning_prompt(denial_text, relevant_terms)
    response: PlanningResponse = structured_llm.invoke([HumanMessage(content=prompt)])
    return {
        "plan": response.plan,
        "required_documents": response.required_documents,
    }


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




def _build_planning_prompt(denial_text: str, relevant_terms: str) -> str:
    return f"""당신은 보험금 지급 분쟁 대리·상담 경험이 있는 전문가입니다.
출력 시 거부 사유에 대한 약관·법적 근거를 전략에 반영하고, 서류는 실제 제출 가능한 구체적 명칭(퇴원요약서, 진단서, 소득증명원 등)으로 적어 주세요.

아래 [거부 명세서]는 문서 파싱(DP) 결과, [관련 보험 약관]은 RAG 검색 결과입니다.
약관이 제공되지 않은 경우에도 명세서 내용만으로 전략과 필요 서류를 제시해 주세요.

[거부 명세서]
{denial_text}

[관련 보험 약관]
{relevant_terms or "(아직 약관 DB가 연결되지 않았습니다.)"}

다음 두 가지를 구조화된 형식으로 작성해 주세요.

1) plan (전략/계획)
- 상황 요약: 피보험자, 보험 종목, 거부 사유, 금액 등 핵심 사실만 2~3문장으로 간결히 요약하세요.
- 전략/계획: 거부 사유에 대한 법·약관상 논거, 분쟁 조정·심사 청구 시 강조할 포인트, 필요 시 보완할 증거(서류)와의 연결을 구체적으로 3~5문장 이상 서술하세요.

2) required_documents (추가 필요 서류)
- 전략적인 분쟁 신청을 위해 "추가로" 제출이 필요한 서류만 나열하세요. 이미 거부 명세서에 포함된 자료는 제외합니다.
- 위 전략에서 필요하다고 판단한 서류만 최대 3개, 각 항목은 "서류명 (목적/키워드)" 형식으로 적고, 서류명은 퇴원요약서·진단서·소득증명원 등 실제 제출 가능한 구체적 명칭을 사용하세요.
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
