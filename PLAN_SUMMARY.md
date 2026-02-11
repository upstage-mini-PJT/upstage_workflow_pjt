# Step3 구현 플랜 요약

## 🎯 핵심 요약

**팀원**: 판례/사례 데이터 가져오기 (`tools/data_analysis_tools/caselaw/`)  
**우리**: 나머지 모든 분석 로직 구현

---

## ✅ 완료된 작업 (2026-02-11)

### 1. 스키마 정의 완료

- ✅ `core/schemas/analysis.py` - Step3 출력 스키마 (3개 JSON)
- ✅ `core/schemas/case_context.py` - Step2 입력 스키마
- ✅ `core/schemas/provenance.py` - 근거 추적
- ✅ `tools/data_analysis_tools/caselaw/types.py` - 팀원과의 계약

### 2. 출력 구조 확정

```
Step3Output {
  meta: {...}
  analysis_report: {
    issue_tree: [...]           // 쟁점 분석
    gap_analysis: {...}         // 증빙 갭
    recommended_actions: [...]  // 권장 액션
    success_probability: {...}  // 성공 확률 (0-100)
  }
  strategy_plan: {
    strategy_overview: "..."    // 전략 개요
    primary_claims: [...]       // 핵심 주장
    step_by_step_plan: [...]    // 실행 계획
    recommended_channel: "..."  // 제출 채널
  }
  research_pack: {
    caselaw_results: [...]      // 판례
    web_sources: [...]          // 웹 소스
  }
}
```

---

## 🔨 다음 구현 단계

### Phase 2: Scoring 엔진 (우선순위 1)

```
core/scoring/
├── features.py              # 10-15개 feature 추출
└── rubric.py                # YAML 기반 점수 산정

data/data_analysis_data/rubrics/
└── success_probability_v1.yml  # 점수 산정 규칙
```

**핵심 로직**:

- Feature 추출 (유사 판례 존재, 증빙 완전성, 절차 준수 등)
- YAML 룰셋 기반 점수 산정 (base 50점 + 규칙 적용)
- 설명 가능성 (drivers_positive/negative, assumptions)

### Phase 3: 파이프라인 (우선순위 2)

```
agents/data_analysis_agent/
├── pipeline.py              # 7단계 파이프라인
├── agent.py                 # 메인 실행
└── adapters.py              # 입출력 변환
```

**7단계 파이프라인**:

1. Retrieval (Mock 데이터 사용)
2. Issue Analysis
3. Gap Analysis
4. Scoring
5. Strategy Planning
6. Recommendations
7. Evidence Packaging

### Phase 4: LLM 통합 (우선순위 3)

```
agents/data_analysis_agent/prompts/
├── issue_tree.md            # 쟁점 분석 프롬프트
├── strategy.md              # 전략 수립 프롬프트
├── recommendation.md        # 권장 액션 프롬프트
└── system.md                # 시스템 프롬프트
```

**원칙**: LLM은 설명/요약용, 판단은 규칙 기반

### Phase 5: 평가 시스템 (우선순위 4)

```
evaluations/data_analysis/
├── dataset.jsonl            # 테스트 케이스 5-10개
├── eval.py                  # 평가 스크립트
└── metrics.py               # 메트릭 (stability, provenance, accuracy)
```

---

## 🤝 팀원 협업

### 팀원에게 전달할 것

1. `tools/data_analysis_tools/caselaw/types.py` 스키마 검토 요청
2. Mock 데이터 샘플 3-5개 요청
3. API 완성 예상 일정 확인

### 우리가 독립적으로 진행

- Mock 데이터로 전체 파이프라인 구현
- 팀원 작업 완료 후 Mock → Real 교체만 하면 됨

---

## 📅 타임라인

- **Week 1**: Scoring 엔진 + Mock 파이프라인
- **Week 2**: 파이프라인 완성 + 단위 테스트
- **Week 3**: LLM 통합 + 평가 시스템
- **Week 4**: 팀원 작업 통합 + E2E 테스트

---

## 🚀 즉시 시작 가능한 작업

1. **Scoring 엔진 구현** (`core/scoring/features.py`, `rubric.py`)
2. **YAML 룰셋 작성** (`data/data_analysis_data/rubrics/success_probability_v1.yml`)
3. **Mock 데이터 준비** (판례 3-5개 샘플)

---

## 📖 상세 문서

전체 구현 계획은 `IMPLEMENTATION_PLAN.md` 참조
