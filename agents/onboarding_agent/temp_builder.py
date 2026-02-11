"""
onboarding_agent 전용 테스트 그래프.
노드는 agents/onboarding_agent/nodes.py 에서 import.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_upstage import (
    ChatUpstage,
    UpstageDocumentParseLoader,
    UpstageUniversalInformationExtraction,
)

from .nodes import dummy_node

# 상위 위치에 있는 env파일을 참조하기 위해 루트 조정(추후에 외부 그래프에서 실행시에는 필요없는 로직)
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  # project root
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# api 클라이언트 관리 (그래프 전에 미리 인스턴스 생성해서 그래프 내에서 싱글톤으로 관리)
config = {
    "configurable": {
        "ie_client": UpstageUniversalInformationExtraction(api_key=os.getenv("UPSTAGE_API_KEY")),
        "chat_client": ChatUpstage(api_key=os.getenv("UPSTAGE_API_KEY")),
    }
}

# 1. 그래프 빌드
builder = StateGraph( )
builder.add_node("analyze", dummy_node)
builder.set_entry_point("analyze")
builder.add_edge("analyze", END)
graph = builder.compile()




# 테스트 시:
# graph.invoke(inputs={"pdf_text": "PDF에서 추출된 아주 긴 텍스트..."}, config=config)



