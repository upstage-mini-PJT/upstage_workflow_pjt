"""
retrieval_evaluator.py

Evaluates the quality of the retrieve_terms tool across three test cases.
Run this file to check how well the retriever works before integrating
into the full agent workflow.

Test cases cover:
    1. Exact fact retrieval  — 입원 공제금액
    2. Semantic similarity   — 미용 목적 수술
    3. Complex reasoning     — 보험 기간 종료 후 치료
"""

import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage

from agents.onboarding_agent.terms_retriever_agent import terms_retriever_agent

load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

# ============================================================================
# Test Cases
# ============================================================================
# ============================================================================
# Test Cases — realistic client data, no hardcoded queries
# ============================================================================

TEST_CASES = [
    {
        "id": 1,
        "category": "Exact Fact Retrieval",
        "policy_date": "20200101",
        "insurance_type": "노후실손의료비 상해(갱신형)",
        "denial_statement": "피보험자가 입원 치료를 받았으나 공제금액 및 보상비율 기준에 따라 일부 금액이 지급되지 않았습니다.",
        "expected": "입원당 공제금액 30만원, 급여 80% / 비급여 70% 보상비율"
    },
    {
        "id": 2,
        "category": "Semantic Similarity",
        "policy_date": "20200101",
        "insurance_type": "노후실손의료비 상해(갱신형)",
        "denial_statement": "피보험자의 청구 건은 외모 개선을 목적으로 한 쌍꺼풀 수술로 판단되어 보험금 지급이 거절되었습니다.",
        "expected": "외모개선 목적 수술은 보상 제외, 기능개선 목적은 예외적으로 보상 가능"
    },
    {
        "id": 3,
        "category": "Complex Reasoning",
        "policy_date": "20200101",
        "insurance_type": "노후실손의료비 질병(갱신형)",
        "denial_statement": "보험기간 종료 이후에 발생한 치료비에 대해서는 보상이 불가하다는 이유로 보험금 지급이 거절되었습니다.",
        "expected": "보험기간 종료일로부터 180일까지 계속 중인 치료 보상 가능"
    }
]


# ============================================================================
# Evaluation Function — uses terms_retriever_agent instead of direct retrieval
# ============================================================================

def evaluate_retrieval(test_case: dict, llm: ChatUpstage) -> dict:
    """
    Run terms_retriever_agent with realistic client data,
    then evaluate the quality of retrieved documents.
    """
    # Build client message exactly as it would appear in production
    client_message = (
        f"Policy date: {test_case['policy_date']}\n"
        f"Insurance type: {test_case['insurance_type']}\n"
        f"Denial statement: {test_case['denial_statement']}"
    )

    # Run agent — it generates its own query based on client data
    agent_result = terms_retriever_agent.invoke({
        "messages": [{
            "role": "user",
            "content": client_message
        }]
    })

    retrieved_docs = agent_result["messages"][-1].content
    # Truncate to avoid context length issues in eval prompt
    MAX_CHARS = 3000
    retrieved_docs_for_eval = retrieved_docs[:MAX_CHARS] + "\n... [truncated]" if len(
        retrieved_docs) > MAX_CHARS else retrieved_docs

    # Format context for evaluation using truncated version
    context = "\n\n".join([
        f"[문서 {i + 1}]\n{doc}"
        for i, doc in enumerate(retrieved_docs_for_eval.split("\n\n"))
    ])

    eval_prompt = f"""당신은 보험 약관 검색 품질 평가 전문가입니다.

고객 정보:
- 보험 종류: {test_case['insurance_type']}
- 부지급 사유: {test_case['denial_statement']}
기대되는 답변 방향: {test_case['expected']}

검색된 문서들:
{context}

다음 3가지 기준으로 평가해주세요:

1. 관련성 (Relevance): 검색된 문서가 부지급 사유에 답하는데 관련이 있는가? (1-5점)
   - 5점: 모든 문서가 직접 관련됨
   - 3점: 일부 문서만 관련됨
   - 1점: 대부분의 문서가 무관함

2. 완전성 (Completeness): 부지급 사유에 완전히 답하기에 충분한 정보가 있는가? (1-5점)
   - 5점: 완전히 답할 수 있음
   - 3점: 부분적으로 답할 수 있음
   - 1점: 정보 부족

3. 명확성 (Clarity): 문서들이 명확하고 일관되는가? 혼란스럽거나 상충되는 정보가 없는가? (1-5점)
   - 5점: 매우 명확하고 일관됨
   - 3점: 일부 혼란스러운 부분 있음
   - 1점: 혼란스럽고 상충됨

반드시 아래 형식으로만 출력하세요:
관련성: [점수]/5 - [이유 한 문장]
완전성: [점수]/5 - [이유 한 문장]
명확성: [점수]/5 - [이유 한 문장]
종합: [한 문장 요약]
반드시 문제점과 개선 가능한 부분을 찾아 지적하세요.
완벽한 검색 결과는 드문 경우이므로, 3점 이하의 점수도 적극적으로 부여하세요.
"""

    response = llm.invoke(eval_prompt)
    return {
        "client_message": client_message,
        "retrieved_docs": retrieved_docs,
        "evaluation": response.content
    }

# ============================================================================
# Terminal Output Helpers
# ============================================================================

def print_header(text: str, char: str = "=", width: int = 80):
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width)

def print_section(text: str, width: int = 80):
    print("\n" + "-" * width)
    print(f"  {text}")
    print("-" * width)

def print_test_case(test_case: dict, result: dict):
    print_header(
        f"TEST {test_case['id']}: {test_case['category']}",
        char="█",
        width=80
    )

    print(f"\n  Insurance Type : {test_case['insurance_type']}")
    print(f"  Policy Date    : {test_case['policy_date']}")
    print(f"  Denial         : {test_case['denial_statement']}")
    print(f"  Expected       : {test_case['expected']}")

    print_section("Client Message Sent to Agent")
    print(f"  {result['client_message']}")

    print_section("Retrieved Documents")
    print(result["retrieved_docs"])

    print_section("Evaluation")
    for line in result["evaluation"].strip().split("\n"):
        print(f"  {line}")


def print_summary(results: list):
    print_header("SUMMARY", char="*", width=80)

    for i, (test_case, result) in enumerate(results):
        print(f"\n  Test {test_case['id']} — {test_case['category']}")
        print(f"  Denial: {test_case['denial_statement']}")  # ← was test_case['query']

        for line in result["evaluation"].strip().split("\n"):
            if any(key in line for key in ["관련성:", "완전성:", "명확성:", "종합:"]):
                print(f"    {line.strip()}")

    print("\n" + "*" * 80)


# ============================================================================
# Main
# ============================================================================

def main():
    llm = ChatUpstage(
        api_key=UPSTAGE_API_KEY,
        model=LLM_MODEL
    )

    print_header("RETRIEVAL EVALUATOR — Insurance Policy RAG", char="=", width=80)
    print(f"  Running {len(TEST_CASES)} test cases...")

    results = []
    for test_case in TEST_CASES:
        print(f"\n  Processing test {test_case['id']}: {test_case['category']}...")
        result = evaluate_retrieval(test_case, llm)
        print_test_case(test_case, result)
        results.append((test_case, result))

    print_summary(results)


if __name__ == "__main__":
    main()