"""
onboarding_agent용 그래프 노드 정의.

- temp_builder.py: 여기서 노드를 import 해서 테스트용 그래프 구성
- workflow/builder.py: 여기서 노드를 import 해서 메인 그래프에 add_node
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from tools.document_parser import parse_document
from tools.rag_terms import fetch_relevant_insurance_terms

from agents.onboarding_agent.schemas import (
    PlanningResponse,
    ExtractedDocumentInfo,
    SufficiencyResponse
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
    읽기: denial_statement_text
    쓰기: relevant_terms
    """
    denial_text = (state.get("denial_statement_text") or "").strip()
    relevant_terms = fetch_relevant_insurance_terms(denial_text, top_k=5)
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
    req_str = "\n".join(f"- {r}" for r in (required_documents or [])) or "(없음)"
    docs_str = ""
    for i, info in enumerate(extracted_infos, 1):
        docs_str += f"\n[문서 {i}]\n"
        docs_str += f"핵심 데이터: {info.get('key_data', '')}\n"
        docs_str += f"근거: {info.get('evidence_or_grounds', '')}\n"
        docs_str += f"기타: {info.get('helpful_notes', '')}\n"
    return f"""분쟁 신청 계획과 요청했던 서류, 그리고 아래 추출된 문서별 정보를 종합했을 때, 분쟁 신청을 진행하기에 **근거가 충분한지** 판단해 주세요.
    충분하면 sufficient=True, 부족하면 sufficient=False로 답하세요.
    """
