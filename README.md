# 보험금 부지급 대응 에이전틱 워크플로우 - Step3 Data Analysis

보험금 부지급 통지를 받은 사용자를 위한 AI 기반 분석 및 재심의 전략 수립 시스템의 **Step3 (Data Analysis)** 구현 프로젝트입니다.

## 📋 프로젝트 개요

### 전체 워크플로우

1. **Step1 - Onboarding**: 사용자 정보, 보험 정보, 약관 PDF, 부지급 통지서 수집
2. **Step2 - Situation Explanation & Structuring**: 상황 설명 및 사실/쟁점/약관 구조화
3. **Step3 - Data Analysis** ⭐ **(이 프로젝트)**: 판례/사례 기반 분석 및 재심의 전략 수립
4. **Step4 - Re-review Document Drafting**: 재심의 요청 문서 생성

### Step3의 역할

- 보험사 부지급 사유를 **법적/실무적으로 분석**
- 유사 판례 및 분쟁 사례를 근거로 **재심의 가능성 판단**
- 사용자가 다음 액션을 결정할 수 있도록 **정량 + 정성 정보 제공**

## 🎯 출력 구조

Step3는 3개의 JSON 출력을 생성합니다:

### 1. 분석 리포트 (`analysis_report`)

```json
{
  "issue_tree": [...],           // 쟁점 트리 분석
  "gap_analysis": {...},         // 부족한 증빙 식별
  "recommended_actions": [...],  // 권장 액션
  "success_probability": {       // 재심의 성공 확률
    "score_0_to_100": 62,
    "band": "MEDIUM",
    "drivers_positive": [...],
    "drivers_negative": [...],
    "assumptions": [...]
  }
}
```

### 2. 재심의 전략 계획 (`strategy_plan`)

```json
{
  "strategy_overview": "...",    // 전략 개요
  "primary_claims": [...],       // 핵심 주장
  "step_by_step_plan": [...],    // 단계별 실행 계획
  "recommended_channel": "..."   // 제출 채널
}
```

### 3. 리서치 결과 (`research_pack`)

```json
{
  "caselaw_results": [...],      // 판례 검색 결과
  "web_sources": [...]           // 웹 소스 (사례/기사)
}
```

## 🗂️ 프로젝트 구조

```
.
├── README.md                    # 이 파일
├── IMPLEMENTATION_PLAN.md       # 상세 구현 계획
├── PLAN_SUMMARY.md              # 구현 플랜 요약
├── data_analysis.md             # Step3 구현 가이드
│
├── core/                        # 핵심 로직
│   ├── schemas/                 # 데이터 스키마
│   │   ├── analysis.py          # ✅ Step3 출력 스키마
│   │   ├── case_context.py      # ✅ Step2 입력 스키마
│   │   └── provenance.py        # ✅ 근거 추적
│   └── scoring/                 # 점수 산정 엔진
│       ├── features.py          # 🔨 Feature 추출
│       └── rubric.py            # 🔨 YAML 기반 점수 산정
│
├── agents/                      # 에이전트 로직
│   └── data_analysis_agent/
│       ├── pipeline.py          # 🔨 5단계 파이프라인
│       ├── agent.py             # 🔨 메인 실행 로직
│       ├── adapters.py          # 🔨 입출력 변환
│       └── prompts/             # 🔨 LLM 프롬프트
│
├── tools/                       # 도구
│   └── data_analysis_tools/
│       └── caselaw/             # 판례 검색 (팀원 담당)
│           ├── types.py         # ✅ 데이터 타입 정의
│           ├── client.py        # 👥 판례 API 호출
│           ├── query_builder.py # 👥 검색 쿼리 생성
│           ├── normalizer.py    # 👥 결과 정규화
│           └── ranker.py        # 👥 Relevance scoring
│
├── data/                        # 데이터 & 룰셋
│   └── data_analysis_data/
│       └── rubrics/             # 🔨 점수 산정 규칙
│           ├── issue_mapping.yml
│           └── success_probability_v1.yml
│
├── evaluations/                 # 평가 시스템
│   └── data_analysis/
│       ├── eval.py              # 🔨 평가 스크립트
│       ├── metrics.py           # 🔨 평가 메트릭
│       └── dataset.jsonl        # 🔨 테스트 데이터셋
│
└── main.py                      # 메인 실행 파일

범례:
✅ 완료
🔨 구현 필요
👥 팀원 담당
```

## 🚀 시작하기

### 환경 설정

```bash
# Python 3.13 권장 (Chroma 안정 실행)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 의존성 설치 (pyproject.toml 기준)
pip install -e .
```

### 실행

```bash
# 파이프라인 실행 예시 (langgraph 필요)
python -c "from agents.data_analysis_agent.pipeline import run_pipeline; print(run_pipeline({}))"
```

## 🔗 RAG v1 실행 순서

1. Raw 문서를 `normalize_ingestion_doc(s)`로 `IngestionDoc`으로 정규화
2. `chunk_ingestion_doc(s)`로 `VectorChunk` 생성
3. `embed_chunks()`로 임베딩 생성
4. `InMemoryVectorIndexStore` 또는 `ChromaVectorIndexStore`로 인덱스 적재
5. `VectorRetriever.retrieve()`로 `RAGRetrievalResult` 생성
6. `rerank_retrieval_result()`로 재정렬 점수 적용
7. `compute_comparative_scoring()`으로 `scoring_trace` 계산

## ⚙️ Chroma 사용 설정

```bash
export RAG_VECTOR_BACKEND=chroma
export RAG_CHROMA_DIR=.chroma_db
export RAG_INDEX_NAME=step3_rag_index
```

- 기본값은 `inmemory`
- Chroma 실행 시 Python 3.14는 비호환 이슈가 있어 3.13 권장

## 📚 KCA 분쟁사례 데이터 연동

- 반입 경로:
  - `data/data_analysis_data/converted_cases/kca_finance_insurance_cases_all.json`
- 보험 필터 결과:
  - `data/data_analysis_data/converted_cases/kca_finance_insurance_cases_filtered.json`
- 검색 소스 연동 기본값:
  - `RAG_INCLUDE_KCA=true`

```bash
# KCA 소스 끄기
export RAG_INCLUDE_KCA=false

# KCA 데이터 파일 경로 커스텀
export RAG_KCA_DATA_PATH=data/data_analysis_data/converted_cases/kca_finance_insurance_cases_filtered.json
```

## 🧠 HyDE / Reverse HyDE 설정

```bash
# 검색 모드: plain | hyde | reverse_hyde | hybrid_hyde
export RAG_RETRIEVAL_MODE=hybrid_hyde

# HyDE 생성 on/off (민감정보 마스킹 포함)
export RAG_ENABLE_HYDE=true
```

- `plain`: 원문 쿼리만 사용
- `hyde`: 가설 문서(질문 확장) 기반 검색
- `reverse_hyde`: 1차 검색 결과를 바탕으로 재질의 생성 후 검색
- `hybrid_hyde`: `plain + hyde + reverse_hyde` 융합

모드 비교 리포트 생성:

```bash
PYTHONPATH=. uv run python evaluations/data_analysis/run_hyde_mode_comparison.py
```

생성 파일:
- `evaluations/data_analysis/hyde_mode_comparison.json`
- `evaluations/data_analysis/hyde_mode_comparison.md`

## 🧪 테스트 실행

```bash
PYTHONPATH=. uv run pytest -q tests/test_chroma_integration.py tests/test_rag_contract.py tests/test_vector_retriever.py tests/test_comparative_scoring.py tests/test_rag_e2e.py
```

## 🧩 RAG 모듈 구조

`tools/data_analysis_tools/rag/normalizer.py`  
`tools/data_analysis_tools/rag/chunker.py`  
`tools/data_analysis_tools/rag/embedder.py`  
`tools/data_analysis_tools/rag/index_store.py`  
`tools/data_analysis_tools/rag/vector_retriever.py`  
`tools/data_analysis_tools/rag/reranker.py`  
`tools/data_analysis_tools/rag/search_client.py`  
`tools/data_analysis_tools/rag/comparative_scoring.py`

## 🛠️ 운영 체크리스트

- 임베딩 모델명 또는 차원(`embedding_dim`)이 바뀌면 기존 인덱스를 폐기하고 전량 재임베딩
- `chunk_size_tokens`/`chunk_overlap_tokens` 변경 시 인덱스 재생성
- `source_type`, `doc_id/chunk_id` 규칙 변경 시 기존 데이터 마이그레이션 계획 수립
- 점수 가드레일(`-15~+15`, `0~100`) 변경 시 비교분석 회귀 테스트 재실행
- 웹 소스 신뢰도(`publisher_grade`) 정책 변경 시 기존 WEB 문서 메타 재계산

## 📐 구현 단계

### ✅ Phase 1: 스키마 정의 (완료)

- [x] 출력 스키마 (`core/schemas/analysis.py`)
- [x] 입력 스키마 (`core/schemas/case_context.py`)
- [x] 근거 추적 (`core/schemas/provenance.py`)
- [x] 팀원 계약 (`tools/data_analysis_tools/caselaw/types.py`)

### 🔨 Phase 2: Scoring 엔진 (진행중)

- [ ] Feature 추출 로직
- [ ] YAML 룰셋 작성
- [ ] 점수 산정 엔진

### 🔨 Phase 3: 파이프라인

- [ ] Mock Retrieval 구현
- [ ] 7단계 파이프라인 구현
- [ ] 단위 테스트

### 🔨 Phase 4: LLM 통합

- [ ] 프롬프트 작성
- [ ] LLM 호출 로직
- [ ] 출력 검증

### 🔨 Phase 5: 평가 시스템

- [ ] 테스트 데이터셋
- [ ] 평가 메트릭
- [ ] 평가 스크립트

### 🔨 Phase 6: 통합

- [ ] 팀원 작업 통합
- [ ] End-to-End 테스트
- [ ] 최적화

## 🤝 팀 역할 분담

### 우리 팀

- 핵심 분석 로직 (`core/`, `agents/`)
- Scoring 엔진 (`core/scoring/`)
- 평가 시스템 (`evaluations/`)
- LLM 통합 (`agents/data_analysis_agent/prompts/`)

### 팀원

- 판례/사례 검색 (`tools/data_analysis_tools/caselaw/`)
- API 연동
- 데이터 정규화 및 랭킹

## 📖 문서

- `RAG_SCHEMA_CONTRACT.md`: Step3 RAG 계약(노드 구조 포함)
- `data_analysis.md`: Step3 구현 가이드 (원본 요구사항)
- `IMPLEMENTATION_PLAN.md`: 상세 구현 계획
- `PLAN_SUMMARY.md`: 구현 플랜 요약

## ✅ Definition of Done

Step3 구현 완료 기준:

1. ✅ `agent.py.run(structured_case)` 호출 가능
2. ✅ 결과가 `Step3Output` 스키마 완전 충족
3. ✅ 모든 주장에 provenance 존재
4. ✅ `evaluations/data_analysis/eval.py` 통과
5. ✅ Step4에서 추가 가공 없이 사용 가능

## 📞 문의

프로젝트 관련 문의사항은 이슈를 등록해주세요.

---

**Last Updated**: 2026-02-16  
**Status**: RAG v1 모듈/테스트 브랜치 진행중
