"""
onboarding_agent 그래프용 state 스키마.

플로우: 사용자 입력 파일 → planning(parse_document로 파싱, RAG로 약관 조회, 전략 수립, 필요 서류 목록) → 다음 노드에서 서류 수집·처리
"""

from typing import TypedDict


class OnboardingState(TypedDict, total=False):
    """planning 및 이후 노드에서 사용하는 state."""

    # 입력 (사용자에게 받은 파일 경로)
    denial_file_path: str  # 보험금 지급 거부 명세서 등 사용자 업로드 파일 경로
    policy_date: str  # 약관 버전(YYYYMMDD). 없으면 노드 내부 기본값 사용

    # planning 노드 내부에서 parse_document 결과로 채움 후 사용
    denial_statement_text: str  # DP로 파싱된 거부 명세서 텍스트

    # issue planning 노드 출력
    issue_hypotheses: list[str]
    query_plan: list[dict]
    retrieval_queries: list[str]
    retrieval_candidates: list[dict]

    # planning 노드 출력
    relevant_terms: str  # RAG로 가져온 관련 보험 약관
    plan: str  # 분쟁신청을 위한 전략/계획
    required_document_ids: list[str]  # 카탈로그 기준 추가 요청 서류 ID
    required_documents: list[str]  # 추가로 필요한 서류 목록 (다음 노드에서 수집)
    final_plan_confidence: str

    # request_additional_documents 노드: interrupt 후 resume으로 받은 경로
    additional_document_paths: list[str]

    # parse_and_extract 노드: 문서별 추출 정보 (additional_document_paths와 순서 대응)
    extracted_document_infos: list[dict]
    # evaluate_sufficiency 노드: True면 설명 생성 단계, False면 request_additional로 복귀
    evidence_sufficient: bool

    # explain_decision 노드 출력
    decision_summary: dict
    decision_explanation: str
