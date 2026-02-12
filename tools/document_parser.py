"""
Upstage Document Parse(DP)를 사용해 파일을 텍스트로 파싱.
에이전트/노드에서 파일 경로만 넘겨 호출. (추후 dp_client 주입 시 시그니처 확장 가능)
"""

from pathlib import Path
from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage.document_parse_parsers import OutputFormat


def parse_document(file_path: str) -> str:
    """
    파일 경로를 받아 Upstage DP로 파싱한 뒤 전체 텍스트를 반환.

    Args:
        file_path: 파싱할 파일 경로 (로컬 PDF 등).

    Returns:
        파싱된 평문 텍스트.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    
    loader = UpstageDocumentParseLoader(str(path), ocr = 'auto', output_format = "markdown")
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)
