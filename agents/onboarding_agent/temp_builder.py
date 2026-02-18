"""
onboarding_agent 전용 테스트 그래프.
노드는 agents/onboarding_agent/nodes.py 에서 import.
"""

import os
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
from tools.retrieve_terms import ensure_vectordb_ready, load_vectordb, vectordb_document_count
from .nodes import (
    parse_denial_node,
    issue_planning_node,
    retrieve_terms_node,
    final_planning_node,
    request_additional_documents_node,
    parse_and_extract_node,
    evaluate_sufficiency_node,
    explain_decision_node,
)
from .state import OnboardingState
import uuid
# 상위 위치에 있는 env파일을 참조하기 위해 루트 조정(추후에 외부 그래프에서 실행시에는 필요없는 로직)
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  # project root
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)


# 1. 그래프 빌드: parse_denial → issue_planning → retrieve_terms → final_planning
builder = StateGraph(OnboardingState)
builder.add_node("parse_denial", parse_denial_node)
builder.add_node("issue_planning", issue_planning_node)
builder.add_node("retrieve_terms", retrieve_terms_node)
builder.add_node("final_planning", final_planning_node)
builder.add_node("request_additional_documents", request_additional_documents_node)
builder.add_node("parse_and_extract",parse_and_extract_node)
builder.add_node("evaluate_sufficiency_node", evaluate_sufficiency_node)
builder.add_node("explain_decision", explain_decision_node)


builder.add_edge(START, "parse_denial")
builder.add_edge("parse_denial", "issue_planning")
builder.add_edge("issue_planning", "retrieve_terms")
builder.add_edge("retrieve_terms", "final_planning")
builder.add_edge("final_planning", "request_additional_documents")
builder.add_edge("request_additional_documents", "parse_and_extract")
builder.add_edge("parse_and_extract", "evaluate_sufficiency_node")



def _evidence_sufficiency_path(state: dict) -> str:
    """evidence_sufficient가 True면 explain_decision, False면 request_additional_documents로 복귀."""
    if state.get("evidence_sufficient"):
        return "explain_decision"
    return "request_additional_documents"


builder.add_conditional_edges(
    "evaluate_sufficiency_node",
    _evidence_sufficiency_path,
    {"explain_decision": "explain_decision", "request_additional_documents": "request_additional_documents"},
)
builder.add_edge("explain_decision", END)

memory = MemorySaver()
onboarding_graph = builder.compile(checkpointer=memory)




# 2. 실행 로직 (전처리 + 실행)
if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    print(f"--- 🚀 테스트 시작 (Thread ID: {thread_id}) ---")

    # [Step 1: 외부 전처리] 그래프 실행 전, 입력 문서 경로 받기
    valid_file_path = os.getenv("ONBOARDING_DENIAL_FILE", "").strip()
    if not valid_file_path:
        raise SystemExit(
            "환경변수 ONBOARDING_DENIAL_FILE에 지급거절 명세서 파일 경로를 설정하세요."
        )
    if not Path(valid_file_path).exists():
        raise SystemExit(f"입력 파일이 존재하지 않습니다: {valid_file_path}")

    # [Step 2: 벡터 DB 준비 (콜드 스타트 시 1회 인덱싱)]
    policy_vectordb = None
    try:
        policy_vectordb = ensure_vectordb_ready(load_vectordb())
        print(f"--- 📚 Vector DB ready (docs={vectordb_document_count(policy_vectordb)}) ---")
    except Exception as exc:
        print(f"--- ⚠️ Vector DB cold-start skipped: {exc} ---")

    # [Step 3: 설정 준비]
    configurable = {
        "ie_client": UpstageUniversalInformationExtraction(),
        "chat_client": ChatUpstage(model="solar-pro2"),
        "thread_id": thread_id,
    }
    if policy_vectordb is not None:
        configurable["policy_vectordb"] = policy_vectordb

    runnable_config: RunnableConfig = {
        "configurable": configurable
    }

    # [Step 4: 그래프 실행] interrupt 발생 시 resume 루프 until END
    print(f"\n--- 🤖 그래프 분석 시작 (파일: {valid_file_path}) ---")

    result = onboarding_graph.invoke(
        {"denial_file_path": valid_file_path},
        config=runnable_config
    )

    while result.get("__interrupt__"):
        interrupt_list = result["__interrupt__"]
        payload = interrupt_list[0].value if interrupt_list else {}
        required = payload.get("required_documents", []) if isinstance(payload, dict) else []
        mock_document_list = [valid_file_path] * max(len(required), 1)
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

    explanation = (final_state.get("decision_explanation") or "").strip()
    if explanation:
        print("\n--- 👤 사용자 설명문 ---")
        print(explanation)

    decision_summary = final_state.get("decision_summary") or {}
    if decision_summary:
        print("\n--- 📌 설명 요약 ---")
        print("사용자 상황:", decision_summary.get("user_situation", ""))
        print("보험사 주장:", decision_summary.get("insurer_claim", ""))
        print("결론 근거:", decision_summary.get("conclusion_reason", ""))
