"""
onboarding_agent 전용 테스트 그래프.
노드는 agents/onboarding_agent/nodes.py 에서 import.
"""

import argparse
import os
from pathlib import Path
import uuid
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_upstage import (
    ChatUpstage,
    UpstageUniversalInformationExtraction,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from tools.retrieve_terms import (
    ensure_vectordb_ready,
    list_policy_dates,
    load_vectordb,
    vectordb_document_count,
)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="onboarding_graph test runner")
    parser.add_argument(
        "--denial-file",
        default=os.getenv("ONBOARDING_DENIAL_FILE", "").strip(),
        help="지급거절 명세서 파일 경로",
    )
    parser.add_argument(
        "--join-date",
        default=os.getenv("ONBOARDING_JOIN_DATE", "").strip(),
        help="보험 가입일(YYYYMMDD). 예: 20250301",
    )
    return parser.parse_args()


def _validate_yyyymmdd(value: str, *, label: str) -> str:
    cleaned = str(value).strip()
    if len(cleaned) != 8 or not cleaned.isdigit():
        raise SystemExit(f"{label}는 YYYYMMDD 형식이어야 합니다. 입력값: {value}")
    return cleaned


def _select_policy_date(join_date: str, available_dates: list[str]) -> tuple[str, str]:
    if not available_dates:
        raise SystemExit("vector DB에 policy_date가 없어 가입일 기반 매핑을 수행할 수 없습니다.")

    sorted_dates = sorted({d for d in available_dates if len(d) == 8 and d.isdigit()})
    if not sorted_dates:
        raise SystemExit("vector DB의 policy_date 형식이 유효하지 않아 가입일 기반 매핑을 수행할 수 없습니다.")

    if join_date in sorted_dates:
        return join_date, "가입일과 동일한 policy_date가 존재하여 그대로 사용"

    prior_dates = [d for d in sorted_dates if d <= join_date]
    if prior_dates:
        chosen = prior_dates[-1]
        return chosen, f"가입일({join_date}) 기준 가장 가까운 이전 policy_date({chosen}) 선택"

    chosen = sorted_dates[0]
    return chosen, f"가입일({join_date})이 모든 약관 버전보다 이전이라 최소 policy_date({chosen}) 선택"




# 2. 실행 로직 (전처리 + 실행)
if __name__ == "__main__":
    args = _parse_args()
    thread_id = str(uuid.uuid4())
    print(f"--- 🚀 테스트 시작 (Thread ID: {thread_id}) ---")

    # [Step 1: 외부 전처리] 그래프 실행 전, 입력 문서 경로/가입일 받기
    valid_file_path = str(args.denial_file or "").strip()
    if not valid_file_path:
        raise SystemExit(
            "지급거절 명세서 파일 경로가 필요합니다. "
            "--denial-file 또는 ONBOARDING_DENIAL_FILE을 설정하세요."
        )
    if not Path(valid_file_path).exists():
        raise SystemExit(f"입력 파일이 존재하지 않습니다: {valid_file_path}")

    join_date = str(args.join_date or "").strip()

    # [Step 2: 벡터 DB 준비 (콜드 스타트 시 1회 인덱싱)]
    policy_vectordb = None
    available_policy_dates: list[str] = []
    try:
        policy_vectordb = ensure_vectordb_ready(load_vectordb())
        print(f"--- 📚 Vector DB ready (docs={vectordb_document_count(policy_vectordb)}) ---")
        available_policy_dates = list_policy_dates(policy_vectordb)
        if available_policy_dates:
            print(f"--- 🗂️ Available policy_date: {', '.join(available_policy_dates)} ---")
    except Exception as exc:
        print(f"--- ⚠️ Vector DB cold-start skipped: {exc} ---")

    # [Step 3: 가입일 기준 policy_date 선택]
    if not join_date:
        raise SystemExit(
            "가입일이 필요합니다. --join-date 또는 ONBOARDING_JOIN_DATE를 설정하세요."
        )
    normalized_join_date = _validate_yyyymmdd(
        join_date,
        label="--join-date / ONBOARDING_JOIN_DATE",
    )
    selected_policy_date, selection_reason = _select_policy_date(
        normalized_join_date,
        available_policy_dates,
    )
    print(f"--- 🧭 Selected policy_date: {selected_policy_date} ({selection_reason}) ---")

    # [Step 4: 설정 준비]
    configurable = {
        "ie_client": UpstageUniversalInformationExtraction(),
        "chat_client": ChatUpstage(model="solar-pro2"),
        "policy_date": selected_policy_date,
        "thread_id": thread_id,
    }
    if policy_vectordb is not None:
        configurable["policy_vectordb"] = policy_vectordb

    runnable_config: RunnableConfig = {
        "configurable": configurable
    }

    # [Step 5: 그래프 실행] interrupt 발생 시 resume 루프 until END
    print(
        f"\n--- 🤖 그래프 분석 시작 (파일: {valid_file_path}, "
        f"policy_date: {selected_policy_date}) ---"
    )

    result = onboarding_graph.invoke(
        {
            "denial_file_path": valid_file_path,
            "policy_date": selected_policy_date,
        },
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
