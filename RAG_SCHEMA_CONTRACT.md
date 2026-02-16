# RAG Schema Contract (Draft v1)

## 목적
Step3에서 팀 간 연동을 위해 아래 3개 구간의 데이터 계약을 먼저 고정합니다.

1. 판례/사례/웹 문서 정규화 및 벡터 적재 (`IngestionDoc`, `VectorChunk`)
2. 판례/사례/웹 검색 결과 반환 (`RAGRetrievalResult`)
3. 유저 데이터와 RAG 결과를 비교 분석 (`ComparativeAnalysisInput`, `ComparativeAnalysisOutput`)

---

## 0) 공통 트리 노드 스키마
```json
{
  "TreeNode": {
    "node_id": "string",
    "node_type": "ROOT|GROUP|DOCUMENT|CHUNK|ANALYSIS|STRATEGY|RESEARCH|SCORE_STEP",
    "parent_id": "string|null",
    "title": "string",
    "payload": {},
    "children": ["TreeNode"]
  }
}
```

### 설명
- 모든 단계에서 flat 구조와 함께 트리(`TreeNode`)를 병행 제공
- `payload`에 단계별 확장 정보를 저장

---

## 1) Ingestion 문서 스키마
```json
{
  "IngestionDoc": {
    "doc_id": "string",
    "source_type": "CASELAW|DISPUTE|WEB",
    "title": "string",
    "body": "string",
    "published_at": "YYYY-MM-DD",
    "jurisdiction": "string|null",
    "tags": ["string"],
    "url": "string|null",
    "meta": {}
  }
}
```

### 필드 설명
- `doc_id`: 문서 고유 ID (원천 시스템 ID 또는 내부 규칙 ID)
- `source_type`: 문서 출처 구분 (`CASELAW`, `DISPUTE`, `WEB`)
- `title`: 문서 제목
- `body`: 문서 원문/정규화 본문
- `published_at`: 문서 기준 날짜
- `jurisdiction`: 법원/기관/관할 정보
- `tags`: 검색/분류용 키워드
- `url`: 원문 링크
- `meta`: 원천별 확장 필드
- `published_at` 파싱 실패 정책: `1970-01-01`로 대체하고 `meta.published_at_parse_failed=true`, `meta.published_at_raw` 기록

---

## 2) 벡터 청크 스키마
```json
{
  "VectorChunk": {
    "chunk_id": "string",
    "doc_id": "string",
    "source_type": "CASELAW|DISPUTE|WEB",
    "chunk_text": "string",
    "chunk_index": 0,
    "embedding_model": "string",
    "embedding_dim": 0,
    "token_count": 0,
    "metadata": {
      "title": "string",
      "published_at": "YYYY-MM-DD|null",
      "url": "string|null",
      "tags": ["string"]
    }
  }
}
```

### 필드 설명
- `chunk_id`: 청크 고유 ID
- `doc_id`: 원본 문서 ID
- `source_type`: 문서 출처 타입
- `chunk_text`: 임베딩 대상 텍스트
- `chunk_index`: 문서 내 순번
- `embedding_model`: 사용 임베딩 모델명
- `embedding_dim`: 벡터 차원
- `token_count`: 청크 토큰 수
- `metadata`: 검색/출력에서 활용할 최소 메타

---

## 3) 검색 결과 스키마 (RAG)
```json
{
  "RAGRetrievalResult": {
    "query_id": "string",
    "query": "string",
    "filters": {},
    "items": [
      {
        "source_type": "CASELAW|DISPUTE|WEB",
        "doc_id": "string",
        "chunk_id": "string",
        "title": "string",
        "snippet": "string",
        "score": 0.0,
        "rerank_score": 0.0,
        "url": "string|null",
        "published_at": "YYYY-MM-DD|null",
        "provenance": {
          "index_name": "string",
          "retrieved_at": "ISO-8601",
          "retrieval_method": "vector|keyword|hybrid"
        }
      }
    ],
    "tree": "TreeNode",
    "stats": {
      "candidate_count": 0,
      "returned_count": 0,
      "latency_ms": 0
    }
  }
}
```

### 필드 설명
- `items[*].score`: 1차 검색 점수
- `items[*].rerank_score`: 재정렬 점수
- `provenance`: 추적/재현성 보장을 위한 검색 메타
- `stats`: 운영 모니터링용 지표

---

## 4) 비교 분석 입력 스키마
```json
{
  "ComparativeAnalysisInput": {
    "structured_case": {},
    "structured_case_tree": "TreeNode",
    "rag_result": "RAGRetrievalResult",
    "analysis_options": {
      "risk_level": "CONSERVATIVE|BALANCED|AGGRESSIVE",
      "output_style": "USER_READABLE|EXPERT_LIKE"
    }
  }
}
```

### 설명
- `structured_case`: Step2에서 생성된 케이스 구조화 결과
- `rag_result`: RAG 검색/재정렬 결과
- `analysis_options`: 전략 성향/출력 스타일 옵션

---

## 5) 비교 분석 출력 스키마
```json
{
  "ComparativeAnalysisOutput": {
    "analysis_report": {},
    "analysis_tree": "TreeNode",
    "strategy_plan": {},
    "strategy_tree": "TreeNode",
    "research_pack": {
      "caselaw_results": [],
      "web_sources": []
    },
    "research_tree": "TreeNode",
    "scoring_trace": {
      "precedent_score": 0,
      "case_adjustment": 0,
      "total_score": 0,
      "guardrails_applied": ["string"],
      "tree": "TreeNode"
    }
  }
}
```

### 설명
- `analysis_report`, `strategy_plan`, `research_pack`: 기존 Step3 출력과 호환
- `scoring_trace`: 2단계 점수(판례 1차 + 사례 보정) 추적 정보

---

## 6) 운영 가드레일 권장
- 사례 보정치(`case_adjustment`)는 `-15 ~ +15` 범위 제한
- `rationale` 또는 `cited_case_ids` 없으면 보정 무효화
- `cited_case_ids`는 실제 검색 결과(`RAGRetrievalResult.items`)의 `doc_id/case_id`와 일치해야 함
- 최종 점수는 `0~100` clamp

---

## 7) 협업 시 우선 확정 항목
1. `source_type` enum 값
2. `doc_id/chunk_id` 생성 규칙
3. `score/rerank_score` 스케일
4. `provenance` 최소 필드
5. `scoring_trace` 필수 여부

---

## 8) 규칙 고정값 (v1)
- `doc_id`: `{source_type}:{raw_id}`
- `chunk_id`: `{doc_id}#c{chunk_index}`
- `score`, `rerank_score`: `0.0 ~ 1.0`
- `provenance` 필수 필드: `index_name`, `retrieved_at`, `retrieval_method`
- `case_adjustment`: `-15 ~ +15`
- `total_score`: `0 ~ 100`
