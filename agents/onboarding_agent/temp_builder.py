"""
onboarding_agent 전용 테스트 그래프.
노드는 agents/onboarding_agent/nodes.py 에서 import.
"""

from agents.onboarding_agent.state import OnboardingState


from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from langchain_upstage import (
    ChatUpstage,
    UpstageUniversalInformationExtraction,
)
from langgraph.checkpoint.memory import MemorySaver
from .nodes import (
    parse_denial_node,
    retrieve_terms_node,
    planning_node,
    request_additional_documents_node,
    parse_and_extract_node,
    evaluate_sufficiency_node
)
from .state import OnboardingState
import uuid
# 상위 위치에 있는 env파일을 참조하기 위해 루트 조정(추후에 외부 그래프에서 실행시에는 필요없는 로직)
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  # project root
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)


# 1. 그래프 빌드: parse_denial → retrieve_terms → planning
builder = StateGraph(OnboardingState)
builder.add_node("parse_denial", parse_denial_node)
builder.add_node("retrieve_terms", retrieve_terms_node)
builder.add_node("planning", planning_node)
builder.add_node("request_additional_documents", request_additional_documents_node)
builder.add_node("parse_and_extract",parse_and_extract_node)
builder.add_node("evaluate_sufficiency_node", evaluate_sufficiency_node)


builder.add_edge(START, "parse_denial")
builder.add_edge("parse_denial", "retrieve_terms")
builder.add_edge("retrieve_terms", "planning")
builder.add_edge("planning", "request_additional_documents")
builder.add_edge("request_additional_documents", "parse_and_extract")
builder.add_edge("parse_and_extract", "evaluate_sufficiency_node")



def _evidence_sufficiency_path(state: dict) -> str:
    """evidence_sufficient가 True면 END, False면 request_additional_documents로 복귀."""
    if state.get("evidence_sufficient"):
        return "__end__"
    return "request_additional_documents"


builder.add_conditional_edges(
    "evaluate_sufficiency_node",
    _evidence_sufficiency_path,
    {"__end__": END, "request_additional_documents": "request_additional_documents"},
)

memory = MemorySaver()
onboarding_graph = builder.compile(checkpointer=memory)




# 2. 실행 로직 (전처리 + 실행)
if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    print(f"--- 🚀 테스트 시작 (Thread ID: {thread_id}) ---")

    # [Step 1: 외부 전처리] 그래프 실행 전, 유효한 파일 경로 받기
    valid_file_path = "/Users/chanwooyang/workspace/upstage_workflow_pjt/agents/onboarding_agent/korean_denial_mock.pdf"

    # [Step 2: 설정 준비]
    runnable_config: RunnableConfig = {
        "configurable": {
            "ie_client": UpstageUniversalInformationExtraction(),
            "chat_client": ChatUpstage(model="solar-pro2"),
            "thread_id": thread_id
        }
    }

    # [Step 3: 그래프 실행] interrupt 발생 시 resume 루프 until END
    print(f"\n--- 🤖 그래프 분석 시작 (파일: {valid_file_path}) ---")

    result = onboarding_graph.invoke(
        {"denial_file_path": valid_file_path},
        config=runnable_config
    )

    # 문서 요청 개수는 항상 3개이므로, resume 시 korean_denial_mock 경로 3개 리스트로 전달
    mock_document_list = [valid_file_path] * 3

    while result.get("__interrupt__"):
        interrupt_list = result["__interrupt__"]
        payload = interrupt_list[0].value if interrupt_list else {}
        required = payload.get("required_documents", []) if isinstance(payload, dict) else []
        print(f"  [interrupt] 요청 서류: {required} → resume with {len(mock_document_list)}개 mock 경로")
        result = onboarding_graph.invoke(
            Command(resume=mock_document_list),
            config=runnable_config
        )

    final_state = result
    print("\n--- ✅ 분석 완료 ---")
    plan = final_state.get("plan") or ""
    if plan:
        print(f"결과 Plan: {plan[:100]}{'...' if len(plan) > 100 else ''}")
    req_docs = final_state.get("required_documents")
    if req_docs is not None:
        print(f"필요 서류: {req_docs}")