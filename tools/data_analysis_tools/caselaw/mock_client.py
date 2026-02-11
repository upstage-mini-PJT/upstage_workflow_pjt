"""
Mock CaseLaw Client

실제 판례 API 대신 Mock 데이터베이스(JSON)를 검색하여 유사한 사례를 반환합니다.
테스트 및 개발 단계에서 활용됩니다.
"""

import json
import os
import sys
from typing import List, Dict, Any, Set

# 상위 디렉토리 import 경로
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tools.data_analysis_tools.caselaw.caselaw_types import CaseLawDoc, SearchQuery


class MockCaseLawClient:
    """Mock 판례 검색 클라이언트"""
    
    def __init__(self, mock_db_path: str = None):
        """
        초기화
        
        Args:
            mock_db_path: Mock 데이터베이스 JSON 파일 경로 (None이면 기본 경로)
        """
        if mock_db_path is None:
            # upstage_workflow_pjt/ 디렉토리 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # tools/data_analysis_tools/caselaw -> upstage_workflow_pjt
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            
            mock_db_path = os.path.join(
                project_root, 
                "data/data_analysis_data/mock_cases.json"
            )
        
        self.mock_db_path = os.path.normpath(mock_db_path)
        self.cases = self._load_cases()
    
    def search(self, query: SearchQuery) -> List[CaseLawDoc]:
        """
        Mock DB에서 유사한 판례 검색 (가중치 기반 고도화)
        
        - 제목(Title): 높은 가중치
        - 키워드(Keywords): 중간 가중치
        - 요약/설명: 일반 가중치
        - 부분 일치 지원
        """
        
        results = []
        query_keywords = self._preprocess_keywords(query.keywords)
        
        if not query_keywords:
            return []

        for case_data in self.cases:
            score = 0.0
            matched_set = set()
            
            # 가중치 영역 정의
            weights = {
                "title": (case_data.get("title", ""), 5.0),
                "keywords": (" ".join(case_data.get("keywords", [])), 3.0),
                "summary": (case_data.get("summary", ""), 1.0),
                "full_text": (case_data.get("full_text", "") or "", 0.5)
            }
            
            for q_word in query_keywords:
                for field, (text, weight) in weights.items():
                    text_lower = text.lower()
                    
                    # 1. 완전 일치 (단어 경계 확인)
                    if f" {q_word} " in f" {text_lower} ":
                        score += weight
                        matched_set.add(q_word)
                    # 2. 부분 일치 (포함 관계)
                    elif q_word in text_lower:
                        score += weight * 0.5  # 부분 일치는 가중치의 절반
                        matched_set.add(q_word)

            # 점수 정규화 (Overlap Coefficient 개념 도입)
            # 쿼리 키워드 개수 대비 매칭 정도를 반영
            if score > 0:
                # 0~1 사이로 정규화 (최대 점수 제한)
                normalized_score = min(score / (len(query_keywords) * 5.0), 1.0)
                
                doc = CaseLawDoc(
                    case_id=case_data["case_id"],
                    title=case_data["title"],
                    court=case_data["court"],
                    date=case_data["date"],
                    summary=case_data["summary"],
                    full_text=case_data.get("full_text"),
                    case_type=case_data.get("case_type"),
                    result=case_data.get("result"),
                    relevance_score=float(normalized_score),
                    matched_keywords=list(matched_set),
                    url=case_data.get("url")
                )
                results.append(doc)
        
        # 정렬: 점수 내림차순 -> 날짜 내림차순
        results.sort(key=lambda x: (x.relevance_score, x.date), reverse=True)
        
        return results[:query.top_k]

    def _load_cases(self) -> List[Dict[str, Any]]:
        """Mock JSON 파일 로드"""
        if not os.path.exists(self.mock_db_path):
            print(f"Warning: Mock DB not found at {self.mock_db_path}")
            return []
            
        try:
            with open(self.mock_db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading mock DB: {e}")
            return []
    
    def _preprocess_keywords(self, keywords: List[str]) -> List[str]:
        """키워드 리스트 전처리"""
        return [k.strip().lower() for k in keywords if len(k.strip()) > 1]
    
    def _preprocess_text(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출 (이제 가중치 로직에서 직접 처리하므로 보조용으로만 유지)"""
        words = text.split()
        return [w.strip(".,()[]'\"").lower() for w in words if len(w) > 1]


# ============================================================================
# 테스트 코드
# ============================================================================
if __name__ == "__main__":
    from tools.data_analysis_tools.caselaw.caselaw_types import SearchQuery
    
    client = MockCaseLawClient()
    
    # 테스트 쿼리: 하지정맥류
    query = SearchQuery(
        keywords=["하지정맥류", "수술", "실손", "입원"],
        top_k=3
    )
    
    print(f"Query: {query.keywords}")
    results = client.search(query)
    
    print(f"\nFound {len(results)} results:")
    for result in results:
        print(f"[{result.relevance_score:.2f}] {result.title} ({result.case_id})")
        print(f"   Matched: {result.matched_keywords}")
