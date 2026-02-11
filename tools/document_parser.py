"""
Upstage Document Parse(DP)를 사용해 파일을 텍스트로 파싱.
에이전트/노드에서 config의 dp_client를 넘겨 사용.
"""

import os
from pathlib import Path

from langchain_upstage import UpstageDocumentParseLoader


def parse_document(file_path: str, dp_client) -> str:
    """
    파일 경로를 받아 Upstage DP로 파싱한 뒤 전체 텍스트를 반환.

    Args:
        file_path: 파싱할 파일 경로 (로컬 PDF 등).
        dp_client: config['configurable']['dp_client'].
                   UpstageDocumentParseLoader(api_key=...) 인스턴스이거나
                   api_key를 갖는 객체. 파일마다 로더를 새로 만들어 호출함.

    Returns:
        파싱된 평문 텍스트.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    api_key = getattr(dp_client, "api_key", None) or os.environ.get("UPSTAGE_API_KEY")
    loader = UpstageDocumentParseLoader(str(path), split="page", api_key=api_key)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)
