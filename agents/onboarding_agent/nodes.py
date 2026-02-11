"""
onboarding_agent용 그래프 노드 정의.

- temp_builder.py: 여기서 노드를 import 해서 테스트용 그래프 구성
- workflow/builder.py: 여기서 노드를 import 해서 메인 그래프에 add_node
"""

from langchain_core.messages import HumanMessage

from tools.document_parser import parse_document
from tools.rag_terms import fetch_relevant_insurance_terms

from agents.onboarding_agent.schemas import PlanningResponse


def planning(state: dict, config: dict) -> dict:
    """
    사용자 입력 파일(거부 명세서)을 DP로 파싱하고, RAG로 관련 약관을 가져온 뒤
    분쟁신청 전략을 세우고, 추가 필요 서류 목록을 Pydantic 스키마로 받아 state에 넣는다.

    기대 state 입력: input_file_path (사용자에게 받은 파일 경로)
    출력 state: denial_statement_text, relevant_terms, plan, required_documents
    """
    file_path = (state.get("input_file_path") or "").strip()
    if not file_path:
        return {
            "denial_statement_text": "",
            "relevant_terms": "",
            "plan": "",
            "required_documents": [],
        }

    configurable = config.get("configurable", {})
    dp_client = configurable.get("dp_client")
    chat_client = configurable.get("chat_client")

    if not dp_client:
        return {
            "denial_statement_text": "",
            "relevant_terms": "",
            "plan": "",
            "required_documents": [],
        }

    # Upstage DP로 사용자 파일 파싱
    denial_text = parse_document(file_path, dp_client)

    # RAG: 관련 보험 약관 조회
    relevant_terms = fetch_relevant_insurance_terms(denial_text, top_k=5)

    if not chat_client:
        return {
            "denial_statement_text": denial_text,
            "relevant_terms": relevant_terms,
            "plan": "",
            "required_documents": [],
        }

    # LLM: Pydantic 스키마로 전략 + 필요 서류 목록 생성
    structured_llm = chat_client.with_structured_output(PlanningResponse)
    prompt = _build_planning_prompt(denial_text, relevant_terms)
    response: PlanningResponse = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "denial_statement_text": denial_text,
        "relevant_terms": relevant_terms,
        "plan": response.plan,
        "required_documents": response.required_documents,
    }


def _build_planning_prompt(denial_text: str, relevant_terms: str) -> str:
    return f"""당신은 보험 분쟁 신청을 돕는 전문가입니다.

아래는 보험사가 보험금 지급을 거부한 명세서 내용입니다(DP로 파싱됨).
그리고 RAG로 가져온 관련 보험 약관입니다.

[거부 명세서]
{denial_text}

[관련 보험 약관]
{relevant_terms or "(아직 약관 DB가 연결되지 않았습니다.)"}

다음 두 가지를 작성해 주세요.

1) 피보험자의 현재 상황을 요약하고, 분쟁신청을 위한 전략/계획을 구체적으로 서술하세요.
2) 분쟁 신청을 위해 추가로 제출이 필요한 서류 목록을 나열하세요.
"""
