# Step3 Data Analysis Agent - 최종 구현 플랜

## 📋 개요

**작성일**: 2026-02-11  
**담당**: 우리 팀 (판례 Retrieval은 팀원 담당)  
**목표**: 보험금 부지급 대응 시스템의 Step3 (Data Analysis) 완성

---

## 🎯 출력 구조 (확정)

Step3는 **3개의 JSON 출력**을 생성:

### 1. **분석 리포트** (`analysis_report`)

```json
{
  "issue_tree": [...],           // 쟁점 트리
  "gap_analysis": {...},         // 증빙 갭 분석
  "recommended_actions": [...],  // 권장 액션
  "success_probability": {...}   // 성공 확률 (0-100점)
}
```

### 2. **재심의 전략 계획** (`strategy_plan`)

```json
{
  "strategy_overview": "...",    // 전략 개요
  "primary_claims": [...],       // 핵심 주장
  "step_by_step_plan": [...],    // 단계별 실행 계획
  "recommended_channel": "..."   // 제출 채널
}
```

### 3. **리서치 결과** (`research_pack`)

```json
{
  "caselaw_results": [...],      // 판례 검색 결과
  "web_sources": [...]           // 웹 소스 (사례/기사)
}
```

---

## 🗂️ 구현 범위

### ✅ **우리가 구현할 영역**

```
core/schemas/                    ✅ 완료
├── analysis.py                  ✅ Step3Output, AnalysisReport, StrategyPlan, ResearchPack
├── case_context.py              ✅ StructuredCase (입력)
└── provenance.py                ✅ ProvenanceSource

tools/data_analysis_tools/caselaw/
└── types.py                     ✅ CaseLawDoc, WebCaseDoc, SearchQuery (팀원 계약)

core/scoring/                    🔨 구현 필요
├── features.py                  # Feature 추출 로직
└── rubric.py                    # 점수 산정 엔진 (YAML 기반)

agents/data_analysis_agent/      🔨 구현 필요
├── pipeline.py                  # 5단계 파이프라인
├── agent.py                     # 메인 실행 로직
├── adapters.py                  # 입출력 변환
└── prompts/                     # LLM 프롬프트
    ├── issue_tree.md
    ├── strategy.md
    ├── recommendation.md
    └── system.md

data/data_analysis_data/         🔨 구현 필요
└── rubrics/
    ├── issue_mapping.yml        # 부지급 사유 → 쟁점 매핑
    └── success_probability_v1.yml  # 점수 산정 규칙

evaluations/data_analysis/       🔨 구현 필요
├── eval.py                      # 평가 스크립트
├── metrics.py                   # 평가 메트릭
└── dataset.jsonl                # 테스트 데이터셋
```

### 🤝 **팀원이 구현할 영역**

```
tools/data_analysis_tools/caselaw/
├── client.py                    # 판례 API 호출
├── query_builder.py             # SearchQuery 생성
├── normalizer.py                # API 결과 → CaseLawDoc 변환
└── ranker.py                    # Relevance scoring
```

---

## 📐 구현 단계

### **Phase 1: 스키마 & 계약** ✅ **완료**

- [x] `core/schemas/analysis.py` - 출력 스키마
- [x] `core/schemas/case_context.py` - 입력 스키마
- [x] `core/schemas/provenance.py` - 근거 추적
- [x] `tools/data_analysis_tools/caselaw/types.py` - 팀원 계약

**다음 단계**: 팀원에게 `types.py` 공유 및 피드백 수렴

---

### **Phase 2: Scoring 엔진 구현** 🔨

#### 2.1 Feature 정의 (`core/scoring/features.py`)

```python
def extract_features(case: StructuredCase, caselaws: List[CaseLawDoc]) -> dict:
    """
    케이스에서 점수 산정용 feature 추출

    Features:
    - has_similar_precedent: 유사 판례 존재 여부
    - precedent_win_rate: 유사 판례 승소율
    - evidence_completeness: 증빙 완전성 (0-1)
    - procedural_compliance: 절차 준수 여부
    - policy_clarity: 약관 명확성 (0-1)
    - timeline_consistency: 타임라인 일관성
    - medical_necessity_support: 의학적 필요성 근거 강도
    - claim_amount_reasonableness: 청구 금액 합리성
    """
```

#### 2.2 YAML 룰셋 작성 (`data/data_analysis_data/rubrics/success_probability_v1.yml`)

```yaml
# 성공 확률 산정 규칙
base_score: 50

rules:
  # 긍정 요인
  - condition: has_similar_precedent == true
    score_delta: +25
    driver: "유사 판례 존재"

  - condition: precedent_win_rate > 0.7
    score_delta: +15
    driver: "유사 판례 승소율 높음"

  - condition: evidence_completeness > 0.8
    score_delta: +20
    driver: "증빙 자료 충분"

  - condition: medical_necessity_support == "strong"
    score_delta: +15
    driver: "의학적 필요성 근거 강력"

  # 부정 요인
  - condition: procedural_compliance == false
    score_delta: -30
    driver: "절차적 하자"

  - condition: evidence_completeness < 0.3
    score_delta: -25
    driver: "증빙 자료 부족"

  - condition: timeline_consistency < 0.5
    score_delta: -15
    driver: "타임라인 불일치"

# 밴드 구분
bands:
  LOW: [0, 39]
  MEDIUM: [40, 69]
  HIGH: [70, 100]

# 가정 사항 템플릿
assumptions:
  - "추가 증빙 자료가 확보된다는 가정"
  - "판례의 사실관계가 본 케이스와 유사하다는 가정"
  - "약관 해석이 일관되게 적용된다는 가정"
```

#### 2.3 Rubric 엔진 (`core/scoring/rubric.py`)

```python
def calculate_probability(features: dict, rubric_path: str) -> SuccessProbability:
    """
    YAML 룰셋 기반 점수 산정

    Returns:
        SuccessProbability with:
        - score_0_to_100
        - band (LOW/MEDIUM/HIGH)
        - drivers_positive
        - drivers_negative
        - assumptions
    """
```

**📝 Action Items:**

- [ ] `features.py` 구현 (10-15개 feature)
- [ ] YAML 룰셋 초안 작성
- [ ] `rubric.py` 엔진 구현
- [ ] 단위 테스트 작성

---

### **Phase 3: Mock Retrieval & 파이프라인** 🔨

#### 3.1 Mock 데이터 준비

```python
# tools/data_analysis_tools/caselaw/client.py
MOCK_CASELAW_DATA = [
    CaseLawDoc(
        case_id="2019다123456",
        title="실손보험 의학적 필요성 분쟁",
        court="서울중앙지방법원",
        date="2019-05-15",
        summary="응급 상황에서의 치료는 의학적 필요성이 인정됨",
        relevance_score=0.85,
        ...
    ),
    # 3-5개 샘플
]

class MockCaseLawClient:
    def search(self, query: SearchQuery) -> List[CaseLawDoc]:
        return MOCK_CASELAW_DATA[:query.top_k]
```

#### 3.2 파이프라인 구현 (`agents/data_analysis_agent/pipeline.py`)

```python
class AnalysisPipeline:
    def __init__(self, caselaw_client):
        self.client = caselaw_client

    def run(self, case: StructuredCase) -> Step3Output:
        # 1. Retrieval
        query = self._build_query(case)
        caselaws = self.client.search(query)
        web_sources = []  # 나중에 추가

        # 2. Issue Analysis
        issue_tree = self._analyze_issues(case, caselaws)

        # 3. Gap Analysis
        gap_analysis = self._identify_gaps(case)

        # 4. Scoring
        features = extract_features(case, caselaws)
        probability = calculate_probability(features)

        # 5. Strategy Planning
        strategy = self._plan_strategy(case, issue_tree, probability)

        # 6. Recommendations
        actions = self._generate_actions(gap_analysis, probability)

        # 7. Package Evidence
        research_pack = self._package_research(caselaws, web_sources)

        return Step3Output(
            meta=Meta(...),
            analysis_report=AnalysisReport(...),
            strategy_plan=strategy,
            research_pack=research_pack
        )
```

**📝 Action Items:**

- [ ] Mock 데이터 3-5개 작성
- [ ] 파이프라인 7단계 구현
- [ ] 각 단계별 단위 테스트

---

### **Phase 4: LLM 통합 (설명/요약용)** 🔨

#### 4.1 프롬프트 작성

**`agents/data_analysis_agent/prompts/issue_tree.md`**

```markdown
# 쟁점 분석 프롬프트

당신은 보험 분쟁 분석 전문가입니다.

## 입력

- 보험사 부지급 사유: {denial_reasons}
- 관련 약관 조항: {policy_clauses}
- 유사 판례: {caselaws}

## 출력

쟁점을 트리 구조로 분석하세요:

1. **핵심 쟁점**: 가장 중요한 법적/실무적 쟁점
2. **하위 쟁점**: 핵심 쟁점을 뒷받침하는 세부 쟁점
3. **약관 해석 포인트**: 각 쟁점별 약관 해석 방향

⚠️ 주의:

- 모든 주장에는 판례/약관 근거를 명시하세요
- 단정적 표현 금지 ("~일 것이다" → "~로 해석될 여지가 있다")
```

**`agents/data_analysis_agent/prompts/strategy.md`**

```markdown
# 재심의 전략 수립 프롬프트

## 입력

- 쟁점 분석: {issue_tree}
- 성공 확률: {probability}
- 증빙 갭: {gap_analysis}

## 출력

재심의 전략을 다음 형식으로 작성하세요:

### 1. 전략 개요 (한 문단)

- 핵심 논리
- 기대 효과

### 2. 핵심 주장 (2-3개)

각 주장별:

- 주장 내용
- 근거 (판례/약관)
- 예상 반론 및 재반박

### 3. 실행 계획

- 단계별 액션
- 담당자 (USER/SYSTEM)
- 소요 일수
```

#### 4.2 LLM 호출 로직

```python
# agents/data_analysis_agent/agent.py
def generate_issue_explanation(case, caselaws) -> str:
    """LLM으로 쟁점 설명 생성 (판단은 규칙 기반)"""
    prompt = load_prompt("issue_tree.md").format(
        denial_reasons=case.denial_reasons,
        policy_clauses=case.policy_clauses,
        caselaws=caselaws
    )
    response = llm.generate(prompt)
    return response
```

**📝 Action Items:**

- [ ] 4개 프롬프트 작성
- [ ] LLM 호출 래퍼 구현
- [ ] 출력 검증 로직 (hallucination 방지)

---

### **Phase 5: 평가 시스템** 🔨

#### 5.1 테스트 데이터셋 (`evaluations/data_analysis/dataset.jsonl`)

```jsonl
{"case_id": "TEST-001", "input": {...}, "expected_band": "HIGH", "expected_drivers": ["유사 판례 존재"]}
{"case_id": "TEST-002", "input": {...}, "expected_band": "MEDIUM", "expected_drivers": ["증빙 부족"]}
```

#### 5.2 평가 메트릭 (`evaluations/data_analysis/metrics.py`)

```python
def evaluate_stability(results: List[Step3Output]) -> float:
    """동일 입력에 대한 점수 변동 폭 (< 5점)"""

def evaluate_provenance_coverage(output: Step3Output) -> float:
    """모든 주장에 근거가 있는지 (100% 목표)"""

def evaluate_band_accuracy(output, expected) -> bool:
    """예상 밴드와 일치하는지"""
```

**📝 Action Items:**

- [ ] 테스트 케이스 5-10개 작성
- [ ] 평가 스크립트 구현
- [ ] 통과 기준 설정

---

### **Phase 6: 통합 & 최적화** 🔨

#### 6.1 팀원 작업 통합

```python
# 팀원 작업 완료 후
from tools.data_analysis_tools.caselaw.client import RealCaseLawClient

pipeline = AnalysisPipeline(
    caselaw_client=RealCaseLawClient()  # Mock → Real
)
```

#### 6.2 End-to-End 테스트

- Step2 출력 → Step3 입력 검증
- Step3 출력 → Step4 입력 검증

**📝 Action Items:**

- [ ] 실제 API 연동 테스트
- [ ] 성능 최적화
- [ ] 에러 핸들링

---

## 🚀 우선순위 타임라인

### **Week 1: 기반 구축** (현재)

- [x] Day 1: 스키마 정의 ✅
- [ ] Day 2-3: Scoring 엔진 구현
- [ ] Day 4-5: Mock 데이터 & 파이프라인

### **Week 2: 파이프라인 완성**

- [ ] Day 6-8: 파이프라인 7단계 구현
- [ ] Day 9-10: 단위 테스트

### **Week 3: LLM & 평가**

- [ ] Day 11-13: 프롬프트 & LLM 통합
- [ ] Day 14-15: 평가 시스템

### **Week 4: 통합**

- [ ] Day 16-18: 팀원 작업 통합
- [ ] Day 19-20: End-to-End 테스트

---

## 🤝 팀원 협업 체크리스트

### **지금 팀원에게 전달할 것**

- [ ] `tools/data_analysis_tools/caselaw/types.py` 공유
- [ ] `CaseLawDoc`, `WebCaseDoc`, `SearchQuery` 스키마 확인 요청
- [ ] Mock 데이터 샘플 3-5개 요청

### **팀원으로부터 받을 것**

- [ ] 스키마 피드백
- [ ] API 엔드포인트 정보
- [ ] 예상 완료 일정

---

## ✅ Definition of Done

Step3 구현 완료 기준:

1. ✅ `agent.py.run(structured_case)` 호출 가능
2. ✅ 결과가 `Step3Output` 스키마 완전 충족
3. ✅ 모든 주장에 provenance 존재
4. ✅ `evaluations/data_analysis/eval.py` 통과
   - 점수 변동 폭 < 5점
   - Provenance 커버리지 100%
   - Band accuracy > 80%
5. ✅ Step4에서 추가 가공 없이 사용 가능

---

## 📞 다음 단계

**즉시 시작 가능한 작업**:

1. Scoring 엔진 구현 (`core/scoring/`)
2. YAML 룰셋 작성 (`data/data_analysis_data/rubrics/`)
3. Mock 데이터 준비

**팀원과 협의 필요**:

1. `types.py` 스키마 검토
2. Mock 데이터 샘플 공유
3. 통합 일정 조율
