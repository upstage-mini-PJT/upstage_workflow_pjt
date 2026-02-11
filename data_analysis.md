# STEP3 – Data Analysis Agent Implementation Guide

## 0. 목적 (IMPORTANT)

이 문서는 **보험금 부지급 대응 에이전틱 워크플로우 중 Step3(Data Analysis)** 를 구현하기 위한 가이드이다.

- 전체 시스템 흐름은 이해 대상
- **구현 및 수정 범위는 `data_analysis` 영역으로 제한**
- Step1 / Step2 / Step4 및 Orchestrator는 수정 대상 아님
- Step3의 산출물은 이후 Step4(재심의 문서 생성)가 그대로 사용함

---

## 1. 전체 워크플로우 개요 (Context Only)

### 보험금 부지급 대응 에이전틱 플로우

1. **Step1 – Onboarding**

- 유저 정보, 보험 정보, 약관 PDF, 부지급 통지서 수집
- 결과: 원천 데이터 패키지

2. **Step2 – Situation Explanation & Structuring**

- 사용자의 상황을 쉬운 언어로 설명
- 사실 / 쟁점 / 약관 조항을 구조화
- 결과: `StructuredCase`

3. **Step3 – Data Analysis (YOU IMPLEMENT THIS)**

- 판례 / 사례 / 법률 데이터 기반 분석
- 재심의 전략 수립
- 재심의 성공 가능성 산정
- 결과: `AnalysisResult`

4. **Step4 – Re-review Document Drafting**

- Step3 결과를 바탕으로 재심의 요청 문서 생성

---

## 2. Step3(Data Analysis)의 역할 정의

### Step3의 핵심 책임

Step3은 **“판단” 단계**다.

- 보험사의 부지급 사유를 **법적 / 실무적으로 분석**
- 유사 판례 및 분쟁 사례를 근거로 **재심의 가능성 판단**
- 사용자가 다음 액션을 결정할 수 있도록 **정량 + 정성 정보 제공**

⚠️ 주의:

- 법률 자문을 단정적으로 제공하지 않음
- 모든 주장에는 근거(provenance)가 필요함

---

## 3. Step3 입력 / 출력 계약

### Input (from Step2)

- `StructuredCase`
- 사실 타임라인
- 보험사 부지급 사유 (정규화된 형태)
- 관련 약관 조항
- 열린 질문(open questions)

### Output (to Step4)

- `AnalysisResult`
- Issue Tree (쟁점 구조)
- Gap Analysis (부족 증빙)
- Recommended Actions
- Success Probability
- Evidence Pack (판례 / 사례 근거)

모든 Output 스키마는 `core/schemas/analysis.py`에 정의되어 있음.

---

## 4. 구현 범위 (STRICT)

### 수정 / 구현 허용 디렉토리

`agents/data_analysis_agent/
tools/data_analysis_tools/
core/scoring/
evaluations/data_analysis/
data/data_analysis_data/`

### 수정하면 안 되는 영역

- Step1 / Step2 / Step4 관련 코드
- 전체 Orchestrator / API Router
- 다른 agent 디렉토리

---

## 5. 현재 디렉토리 구조 (Relevant)

`
.
├── README.md
├── agents
│ ├── **init**.py
│ └── data_analysis_agent
│ ├── adapters.py
│ ├── agent.py
│ ├── pipeline.py
│ └── prompts
│ ├── issue_tree.md
│ ├── probability.md
│ ├── recommendation.md
│ └── system.md
├── config
│ ├── **init**.py
│ └── data_analysis
│ └── settings.py
├── core
│ ├── **init**.py
│ ├── schemas
│ │ ├── analysis.py
│ │ ├── case_context.py
│ │ └── provenance.py
│ └── scoring
│ ├── features.py
│ └── rubric.py
├── data
│ ├── **init**.py
│ ├── data_analysis_data
│ │ ├── caselaw_sample
│ │ │ ├── **init**.py
│ │ │ └── dispute_cases
│ │ │ └── **init**.py
│ │ ├── mock_cases
│ │ │ └── **init**.py
│ │ └── rubrics
│ │ ├── **init**.py
│ │ ├── issue_mapping.yml
│ │ └── success_probability_v1.yml
│ └── onboarding
│ ├── **init**.py
│ └── step1.md
├── evaluations
│ ├── **init**.py
│ └── data_analysis
│ ├── dataset.jsonl
│ ├── eval.py
│ ├── metrics.py
│ └── reports
├── main.py
├── pyproject.toml
├── tools
│ ├── **init**.py
│ └── data_analysis_tools
│ └── caselaw
│ ├── client.py
│ ├── normalizer.py
│ ├── query_builder.py
│ ├── ranker.py
│ └── types.py
└── uv.lock

---

## 6. Step3 내부 처리 파이프라인 (REQUIRED)

`agents/data_analysis_agent/pipeline.py`는 아래 순서를 따른다.

### 1) Retrieval

- `query_builder.py`로 검색 쿼리 생성
- `caselaw/client.py`를 통해 판례 / 사례 API 호출
- `normalizer.py`로 표준 문서 타입으로 변환
- `ranker.py`로 relevance 정렬

### 2) Issue Analysis

- 보험사 부지급 사유 → 쟁점 트리(issue tree) 생성
- 약관 조항과의 충돌 / 해석 포인트 식별

### 3) Gap Analysis

- 현재 자료로 부족한 증빙 식별
- 추가 확보 시 효과가 큰 증빙 우선순위화

### 4) Success Probability Scoring

- `core/scoring/features.py`에서 feature 추출
- `core/scoring/rubric.py`에서 점수 산정
- 결과는 explainable 해야 함
- positive drivers
- negative drivers
- assumptions

### 5) Evidence Packaging

- 판례 / 사례를 `EvidencePack`으로 묶음
- 각 판단 결과에 provenance 연결

---

## 7. Success Probability 규칙

- 점수 범위: 0 ~ 100
- Band:
- LOW (0–39)
- MEDIUM (40–69)
- HIGH (70–100)

### 점수 산정 기준

- 규칙 기반 (YAML): `data/data_analysis_data/rubrics/success_probability_v1.yml`
- 확률은 **절대적 예측이 아닌 참고 지표**
- 반드시 가정(assumptions)을 명시

---

## 8. Prompts 사용 규칙

- Prompt는 **설명 / 요약 / 전략 서술용**
- 판단 로직 자체는 LLM에 의존하지 말 것
- LLM 출력은 항상 구조화된 스키마에 맞게 변환

---

## 9. Evaluation (MANDATORY)

- `evaluations/data_analysis/`는 반드시 유지
- 최소 요구 사항:
- 동일 입력에 대해 점수 변동 폭이 과도하지 않을 것
- 모든 결론에 provenance가 존재할 것
- expected band와 크게 벗어나지 않을 것

---

## 10. 최종 목표 (Definition of Done)

Step3 구현 완료 기준:

- `agent.py.run(structured_case)` 호출 가능
- 결과가 `AnalysisResult` 스키마를 완전히 충족
- Step4에서 추가 가공 없이 사용 가능
- eval 스크립트 통과

---

## 11. 절대 하지 말 것

- 법률 자문을 단정적으로 서술
- 근거 없는 성공률 제시
- Step4 문서 포맷을 가정하고 출력 구조 변경
- Step3 외 영역 코드 수정

---

## TL;DR

> Step3은 “판례 / 사례 기반 판단 엔진”이다.
> 너의 임무는 **정확하고 설명 가능한 판단 결과를 구조화해서 반환하는 것**이다.
