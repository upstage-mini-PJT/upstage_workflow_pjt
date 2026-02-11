"""
RAG: 거부 명세서/질의를 바탕으로 관련 보험 약관을 가져오는 툴.

추후 벡터 DB·검색 엔진 연동 시 이 함수만 교체하면 됨.
"""


def fetch_relevant_insurance_terms(query: str, *, top_k: int = 5) -> str:
    """
    질의(예: 거부 명세서 요약)와 관련된 보험 약관 조항을 반환.

    Args:
        query: 검색 질의 (거부 사유, 담보명 등).
        top_k: 가져올 문서 수.

    Returns:
        관련 약관 텍스트 (연결된 문자열).
    """
    # TODO: 벡터 DB / 검색 API 연동
    # 예: retriever.invoke(query)[:top_k] → 조합해 반환
    return ""
