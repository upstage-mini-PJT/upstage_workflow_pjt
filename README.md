# upstage-workflow-pjt

## 프로젝트 소개
이 프로젝트는 보험금 지급거절 통지서를 받은 사용자가 재심의를 준비할 수 있도록 돕는 AI 워크플로우입니다.
온보딩 단계(Step1/2)에서 거절 사유, 약관 근거, 제출 증빙을 정리하고, 분석 단계(Step3)에서 유사 근거 기반 전략과 실행 가이드를 생성합니다.
최종 결과는 사용자 친화형 안내와 구조화된 `analysis_result`로 제공되어, 이후 검토/연계 단계에서 바로 활용할 수 있습니다.

## 무엇을 제공하나요?
- 지급거절 통지서 해석 및 핵심 쟁점 정리
- 추가 제출 서류 요청/수집/검토 루프 처리
- 판례/분쟁사례 기반 재심의 전략 액션 제안
- 사용자용 단계별 실행 안내(`User Guidance`) 생성
- 재사용 가능한 구조화 결과(`analysis_result`) 저장 및 전달

## 한눈에 보는 처리 흐름
```text
[입력] 지급거절 명세서 파일 + 가입일
  -> Step1/2 온보딩: 거절 사유 해석, 추가서류 요청/검토, 요약 생성
  -> Step3 분석: 유사 근거 탐색, 쟁점/갭 분석, 재심의 액션 생성
  -> [출력] analysis_result + User Guidance + 실행 결과 JSON 저장
```

## 현재 범위(간단)
- Step1/2 -> Step3 통합 실행을 지원합니다.
- 결과는 `analysis_result` 형태로 저장되며 후속 단계 연계가 가능합니다.
- 테스트/검증은 현재 mock 데이터 기반 시나리오 중심으로 운영합니다.

## 통합 실행 (Step1/2 -> Step3)

```bash
uv run python -m main
```

기본 동작:
- `auto-resume-mock=true` 기본값으로 Step1 interrupt를 manifest 기반 자동 resume 처리
- 온보딩(`decision_summary`, `decision_explanation`) 완료 확인 후에만 Step3 실행

실행 중 입력(기본 모드):
- 지급거절 명세서 파일 경로
- 가입일(`YYYYMMDD`)

수동 resume 모드:
```bash
uv run python -m main --auto-resume-mock false
```
이 경우 interrupt마다 추가 서류 파일 경로를 직접 입력합니다.

manifest 경로 지정:
```bash
uv run python -m main --mock-manifest data/mock_documents/hyundai_senior_silson_case/manifest.json
```

결과:
- 콘솔에 Step2 5줄 요약 출력
  - `사용자 상황`
  - `보험사 주장`
  - `결론 근거`
  - `증빙 상태`
  - `세부 쟁점 안내`
- 콘솔에 Step3 요약 출력
- 콘솔에 사용자 가이드 4섹션 고정 출력
  - `1. 한눈 요약`
  - `2. 지금 할 일`
  - `3. 근거 설명`
  - `4. 안내`
- `outputs/run_YYYYMMDD_HHMMSS.json` 저장
- 저장 JSON에는 `onboarding_state`, `structured_case`, `analysis_result` 포함

`analysis_result` 주요 필드:
- 기존 5키 유지: `issue_tree`, `gap_analysis`, `recommended_actions`, `success_probability`, `evidence_pack`
- 추가(optional): `user_guidance`
  - `plain_summary`: Step3 판단 중심 한눈 요약
  - `issue_brief`: 쟁점별 `title + why_it_matters + needed_evidence`
  - `next_steps`: `recommended_actions`와 동일 개수의 단계별 행동
  - `evidence_guide`: 근거 토큰(`CASELAW:...`)의 의미/제목/관련성 설명
  - `disclaimer`: 참고용 안내 문구

`user_guidance` 생성 규칙:
- Step2 `decision_explanation`의 넘버링 섹션(`1)`, `1.`, `1 )` 등 포맷 변형 포함)을 우선 파싱해 반영
- 번호-구분자 변형(`1 -`, 번호 단독줄 + 다음줄 제목)도 지원
- 섹션 제목 alias(`보험사 주장`, `결론`, `후속 조치`, `유의사항`)도 인식
- 섹션 파싱이 실패하면 `decision_summary` 필드로 자동 폴백
- `next_steps`는 단계별 `무엇/방법/이유/준비물/완료기준` 구조로 생성
- Step2는 간단 요약만 표시하고, 쟁점 수/쟁점 상세는 Step3(`issue_tree`) 기준으로만 표시
- Step3 근거는 도메인 필터(`domain_filter.yml`)를 우선 적용하고 부족 시 soft deny 제한 fallback을 사용
- Step3 액션은 `반박서/증빙보강/제출체크` 카테고리로 정규화 후 중복 병합
- dedupe 이후에도 `recommended_actions` 최소 2개를 보장
- Step3 출력 액션 수와 `user_guidance.next_steps` 개수를 동일하게 유지
- 통합 실행 출력/저장 JSON은 방어적 PII 마스킹을 적용
- User Guidance 핵심 쟁점 라벨은 `쟁점 포인트`로 표기

## 개별 실행

Step1/2(온보딩) 단독:

```bash
uv run python -m agents.onboarding_agent.temp_builder
```

Step3 평가 단독:

```bash
uv run python -m evaluations.data_analysis.eval
```

## 환경 변수(최소)

- `UPSTAGE_API_KEY`

선택:
- `ONBOARDING_CHAT_MODEL` (기본 `solar-pro2`)
- `ONBOARDING_CHAT_TIMEOUT` (기본 `45`)
- `ONBOARDING_CHAT_MAX_RETRIES` (기본 `1`)
- `ONBOARDING_AUTO_RESUME_MOCK` (기본 `true`)
- `ONBOARDING_MOCK_MANIFEST` (기본 `data/mock_documents/hyundai_senior_silson_case/manifest.json`)
- `STEP3_RISK_LEVEL` (`CONSERVATIVE|BALANCED|AGGRESSIVE`)
- `STEP3_OUTPUT_STYLE` (`USER_READABLE|EXPERT_LIKE`)
