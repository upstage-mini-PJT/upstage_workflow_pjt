import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain.agents import create_agent
from tools.retrieve_terms import retrieve_terms

load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

llm = ChatUpstage(
    api_key=UPSTAGE_API_KEY,
    model=LLM_MODEL
)

terms_retriever_agent = create_agent(
    model=llm,
    tools=[retrieve_terms],
    system_prompt=(
        "당신은 보험 약관 검색 어시스턴트입니다. "
        "반드시 retrieve_terms 도구를 한 번만 호출하세요. "
        "도구 호출 후 반환된 텍스트를 그대로 출력하세요. "
        "요약, 설명, 예시, 추가 텍스트를 절대 추가하지 마세요. "
        "도구의 출력 결과가 곧 최종 답변입니다."
    )
)

if __name__ == "__main__":
    result = terms_retriever_agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "Policy date: 20200101\n"
                "Insurance type: 노후실손의료비 상해(갱신형)\n"
                "Denial statement: 피보험자의 청구 건은 미용 목적의 수술로 판단되어 "
                "보험금 지급이 거절되었습니다."
            )
        }]
    })

    relevant_terms = result["messages"][-1].content
    print(relevant_terms)