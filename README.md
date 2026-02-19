# upstage-workflow-pjt

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
- 콘솔에 Step3 요약 출력
- 콘솔에 Step2 요약(`decision_summary`, `decision_explanation` 일부) 출력
- `outputs/run_YYYYMMDD_HHMMSS.json` 저장
- 저장 JSON에는 `onboarding_state`, `structured_case`, `analysis_result` 포함

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
