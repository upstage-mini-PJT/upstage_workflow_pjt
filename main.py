from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_upstage import ChatUpstage, UpstageUniversalInformationExtraction
from langgraph.types import Command

from agents.data_analysis_agent.agent import run as run_data_analysis
from agents.onboarding_agent.temp_builder import onboarding_graph
from core.schemas.case_context import StructuredCase, normalize_structured_case
from tools.retrieve_terms import ensure_vectordb_ready, list_policy_dates, load_vectordb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step1/2 + Step3 통합 실행기")
    parser.add_argument("--denial-file", default="", help="지급거절 명세서 파일 경로")
    parser.add_argument("--join-date", default="", help="보험 가입일(YYYYMMDD)")
    parser.add_argument("--output", default="", help="결과 JSON 저장 경로")
    parser.add_argument(
        "--auto-resume-mock",
        default=str(os.getenv("ONBOARDING_AUTO_RESUME_MOCK", "true")).strip(),
        help="interrupt 시 manifest 자동 매핑 사용 여부(true/false, 기본 true)",
    )
    parser.add_argument(
        "--mock-manifest",
        default=str(
            os.getenv(
                "ONBOARDING_MOCK_MANIFEST",
                "data/mock_documents/hyundai_senior_silson_case/manifest.json",
            )
        ).strip(),
        help="자동 resume용 manifest 경로",
    )
    return parser.parse_args()


def _require_env() -> None:
    if not os.getenv("UPSTAGE_API_KEY"):
        raise SystemExit("UPSTAGE_API_KEY가 필요합니다. .env를 확인하세요.")


def _validate_yyyymmdd(value: str, *, label: str) -> str:
    cleaned = str(value).strip()
    if len(cleaned) != 8 or not cleaned.isdigit():
        raise SystemExit(f"{label}는 YYYYMMDD 형식이어야 합니다. 입력값: {value}")
    return cleaned


def _select_policy_date(join_date: str, available_dates: list[str]) -> tuple[str, str]:
    if not available_dates:
        raise SystemExit("vector DB에 policy_date가 없어 가입일 기반 매핑을 수행할 수 없습니다.")

    sorted_dates = sorted({d for d in available_dates if len(d) == 8 and d.isdigit()})
    if not sorted_dates:
        raise SystemExit("vector DB의 policy_date 형식이 유효하지 않습니다.")

    if join_date in sorted_dates:
        return join_date, "가입일과 동일한 policy_date 사용"

    prior_dates = [d for d in sorted_dates if d <= join_date]
    if prior_dates:
        chosen = prior_dates[-1]
        return chosen, f"가입일({join_date}) 기준 가장 가까운 이전 policy_date({chosen}) 선택"

    chosen = sorted_dates[0]
    return chosen, f"가입일({join_date})이 모든 버전보다 이전이라 최소 policy_date({chosen}) 선택"


def _input_existing_file(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if Path(value).exists():
            return value
        print(f"파일이 존재하지 않습니다: {value}")


def _resolve_denial_file(arg_value: str) -> str:
    value = str(arg_value or "").strip()
    if value and Path(value).exists():
        return value
    if value and not Path(value).exists() and not sys.stdin.isatty():
        raise SystemExit(f"입력 파일이 존재하지 않습니다: {value}")
    if not sys.stdin.isatty():
        raise SystemExit("--denial-file 인자가 필요합니다.")
    return _input_existing_file("지급거절 명세서 파일 경로를 입력하세요: ")


def _resolve_join_date(arg_value: str) -> str:
    value = str(arg_value or "").strip()
    if value:
        return _validate_yyyymmdd(value, label="가입일")
    if not sys.stdin.isatty():
        raise SystemExit("--join-date 인자가 필요합니다.")
    while True:
        entered = input("보험 가입일(YYYYMMDD)을 입력하세요: ").strip()
        if len(entered) == 8 and entered.isdigit():
            return entered
        print("가입일 형식이 올바르지 않습니다. 예: 20250301")


def _parse_bool(value: str, *, name: str) -> bool:
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise SystemExit(f"{name} 값이 유효하지 않습니다: {value} (true/false)")


def _build_chat_client() -> ChatUpstage:
    model_name = str(os.getenv("ONBOARDING_CHAT_MODEL", "solar-pro2")).strip() or "solar-pro2"
    try:
        timeout_sec = float(str(os.getenv("ONBOARDING_CHAT_TIMEOUT", "45")).strip())
    except ValueError:
        timeout_sec = 45.0
    try:
        max_retries = int(str(os.getenv("ONBOARDING_CHAT_MAX_RETRIES", "1")).strip())
    except ValueError:
        max_retries = 1

    return ChatUpstage(model=model_name, timeout=max(timeout_sec, 5.0), max_retries=max(max_retries, 0))


def _load_mock_manifest(manifest_path: str) -> dict[str, str]:
    path = Path(str(manifest_path).strip())
    if not path.exists():
        raise SystemExit(f"mock manifest 파일이 존재하지 않습니다: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"mock manifest 파싱 실패: {path} ({exc})") from exc

    raw_map = payload.get("catalog_to_file", {})
    if not isinstance(raw_map, dict):
        raise SystemExit(f"mock manifest 형식 오류: catalog_to_file이 dict가 아닙니다: {path}")

    resolved: dict[str, str] = {}
    for key, value in raw_map.items():
        doc_id = str(key).strip()
        doc_path = str(value).strip()
        if not doc_id or not doc_path:
            continue
        resolved[doc_id] = doc_path
    return resolved


def _resolve_mock_resume_paths(required_ids: list[str], mock_map: dict[str, str]) -> list[str]:
    if not required_ids:
        raise SystemExit("required_document_ids가 비어 있어 자동 resume를 진행할 수 없습니다.")

    missing: list[str] = []
    paths: list[str] = []
    for doc_id in required_ids:
        mapped = str(mock_map.get(doc_id, "")).strip()
        if not mapped:
            missing.append(doc_id)
            continue
        if not Path(mapped).exists():
            raise SystemExit(f"manifest에 매핑된 파일이 존재하지 않습니다: {doc_id} -> {mapped}")
        paths.append(mapped)

    if missing:
        raise SystemExit(f"manifest 미매핑 doc_id: {missing}")

    return paths


def _normalize_policy_clauses(decision_summary: dict[str, Any]) -> list[str]:
    clauses: list[str] = []
    raw = decision_summary.get("policy_clauses", [])
    if not isinstance(raw, list):
        return clauses

    for item in raw:
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            merged = " - ".join([part for part in [title, snippet] if part])
            if merged:
                clauses.append(merged)
        else:
            text = str(item).strip()
            if text:
                clauses.append(text)
    return clauses


def _normalize_evidence_summary(state: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    extracted = state.get("extracted_document_infos", [])
    if not isinstance(extracted, list):
        return result

    for idx, info in enumerate(extracted, start=1):
        if not isinstance(info, dict):
            continue
        key_data = str(info.get("key_data", "")).strip()
        evidence = str(info.get("evidence_or_grounds", "")).strip()
        notes = str(info.get("helpful_notes", "")).strip()
        summary = " / ".join([part for part in [key_data, evidence, notes] if part])
        result.append(
            {
                "title": f"추가서류 {idx}",
                "summary": summary or "추출 정보 없음",
                "document_type": "additional_document",
                "source": "onboarding.extracted_document_infos",
            }
        )
    return result


def _build_structured_case(onboarding_state: dict[str, Any]) -> StructuredCase:
    decision_summary = onboarding_state.get("decision_summary", {})
    if not isinstance(decision_summary, dict):
        decision_summary = {}

    denial_reasons: list[str] = []
    insurer_claim = str(decision_summary.get("insurer_claim", "")).strip()
    conclusion_reason = str(decision_summary.get("conclusion_reason", "")).strip()
    if insurer_claim:
        denial_reasons.append(insurer_claim)
    if conclusion_reason and conclusion_reason != insurer_claim:
        denial_reasons.append(conclusion_reason)

    timeline: list[dict[str, str]] = []
    user_situation = str(decision_summary.get("user_situation", "")).strip()
    if user_situation:
        timeline.append(
            {
                "date": "",
                "description": user_situation,
                "actor": "insured",
                "source": "onboarding.decision_summary",
            }
        )

    structured_case = {
        "user_info": {
            "policy_date": str(onboarding_state.get("policy_date", "")).strip(),
            "final_plan_confidence": str(onboarding_state.get("final_plan_confidence", "")).strip(),
        },
        "denial_summary": str(onboarding_state.get("plan", "")).strip()
        or str(onboarding_state.get("denial_statement_text", "")).strip()[:800],
        "denial_reasons": denial_reasons,
        "policy_clauses": _normalize_policy_clauses(decision_summary),
        "timeline": timeline,
        "evidence_summary": _normalize_evidence_summary(onboarding_state),
        "open_questions": [str(x).strip() for x in onboarding_state.get("required_documents", []) if str(x).strip()],
    }
    return normalize_structured_case(structured_case)


def _run_onboarding_step(
    denial_file_path: str,
    selected_policy_date: str,
    thread_id: str,
    *,
    auto_resume_mock: bool,
    mock_manifest_path: str,
) -> dict[str, Any]:
    policy_vectordb = ensure_vectordb_ready(load_vectordb())
    mock_map = _load_mock_manifest(mock_manifest_path) if auto_resume_mock else {}

    config: RunnableConfig = {
        "configurable": {
            "ie_client": UpstageUniversalInformationExtraction(),
            "chat_client": _build_chat_client(),
            "policy_vectordb": policy_vectordb,
            "policy_date": selected_policy_date,
            "thread_id": thread_id,
        }
    }

    result = onboarding_graph.invoke(
        {
            "denial_file_path": denial_file_path,
            "policy_date": selected_policy_date,
        },
        config=config,
    )

    interrupt_count = 0
    max_interrupts = 5
    while result.get("__interrupt__"):
        interrupt_count += 1
        if interrupt_count > max_interrupts:
            payload = result["__interrupt__"][0].value if result.get("__interrupt__") else {}
            evidence_sufficient = result.get("evidence_sufficient")
            raise SystemExit(
                f"interrupt 반복 횟수 초과({max_interrupts}). "
                f"last_payload={payload}, evidence_sufficient={evidence_sufficient}"
            )

        payload = result["__interrupt__"][0].value if result.get("__interrupt__") else {}
        required_ids = payload.get("required_document_ids", []) if isinstance(payload, dict) else []
        required_items = payload.get("required_document_items", []) if isinstance(payload, dict) else []
        required_docs = payload.get("required_documents", []) if isinstance(payload, dict) else []
        normalized_required_ids = [str(item).strip() for item in required_ids if str(item).strip()]

        if not normalized_required_ids:
            raise SystemExit(
                "interrupt payload에 required_document_ids가 없습니다. "
                f"required_documents={required_docs}"
            )

        print("\n[추가 서류 요청]")
        for idx, item in enumerate(required_items, start=1):
            doc_id = str(item.get("id", "")).strip()
            name = str(item.get("name", "")).strip()
            print(f"{idx}. {doc_id} - {name}")

        if auto_resume_mock:
            resume_paths = _resolve_mock_resume_paths(normalized_required_ids, mock_map)
            print(f"[auto-resume] {normalized_required_ids} -> {resume_paths}")
        else:
            resume_paths = []
            for doc_id in normalized_required_ids:
                resume_paths.append(_input_existing_file(f"- {doc_id} 파일 경로: "))

        result = onboarding_graph.invoke(Command(resume=resume_paths), config=config)

    return dict(result)


def _analysis_options_from_env() -> dict[str, str]:
    risk_level = str(os.getenv("STEP3_RISK_LEVEL", "BALANCED")).strip().upper() or "BALANCED"
    output_style = str(os.getenv("STEP3_OUTPUT_STYLE", "USER_READABLE")).strip().upper() or "USER_READABLE"
    return {"risk_level": risk_level, "output_style": output_style}


def _save_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"run_{timestamp}.json"


def _ensure_onboarding_complete(onboarding_state: dict[str, Any]) -> None:
    decision_summary = onboarding_state.get("decision_summary")
    decision_explanation = str(onboarding_state.get("decision_explanation", "")).strip()
    if not isinstance(decision_summary, dict) or not decision_summary or not decision_explanation:
        print("[error] 온보딩 결과가 불완전합니다. Step3를 실행하지 않습니다.")
        print(f"- has_decision_summary: {isinstance(decision_summary, dict) and bool(decision_summary)}")
        print(f"- has_decision_explanation: {bool(decision_explanation)}")
        print(f"- available_keys: {sorted(onboarding_state.keys())}")
        raise SystemExit(1)


def main() -> None:
    load_dotenv()
    _require_env()

    args = _parse_args()
    auto_resume_mock = _parse_bool(args.auto_resume_mock, name="--auto-resume-mock")
    denial_file_path = _resolve_denial_file(args.denial_file)
    join_date = _resolve_join_date(args.join_date)

    policy_vectordb = ensure_vectordb_ready(load_vectordb())
    available_policy_dates = list_policy_dates(policy_vectordb)
    selected_policy_date, reason = _select_policy_date(join_date, available_policy_dates)

    thread_id = str(uuid.uuid4())
    print(f"[session] thread_id={thread_id}")
    print(f"[policy] selected_policy_date={selected_policy_date} ({reason})")

    onboarding_state = _run_onboarding_step(
        denial_file_path,
        selected_policy_date,
        thread_id,
        auto_resume_mock=auto_resume_mock,
        mock_manifest_path=args.mock_manifest,
    )
    _ensure_onboarding_complete(onboarding_state)

    decision_summary = onboarding_state.get("decision_summary", {})
    decision_explanation = str(onboarding_state.get("decision_explanation", "")).strip()
    print("\n=== Step2 Result Summary ===")
    print(f"- 사용자 상황: {str(decision_summary.get('user_situation', '')).strip()[:120]}")
    print(f"- 보험사 주장: {str(decision_summary.get('insurer_claim', '')).strip()[:120]}")
    print(f"- 결론 근거: {str(decision_summary.get('conclusion_reason', '')).strip()[:120]}")
    print(f"- 설명문: {decision_explanation[:180]}")

    structured_case = _build_structured_case(onboarding_state)

    analysis_result = run_data_analysis(
        structured_case,
        rag_result=None,
        analysis_options=_analysis_options_from_env(),
    )

    actions = analysis_result.get("recommended_actions", [])
    band = (analysis_result.get("success_probability") or {}).get("band", "UNKNOWN")
    evidence_count = len(analysis_result.get("evidence_pack", []))

    print("\n=== Step3 Result Summary ===")
    print(f"- success_probability.band: {band}")
    print(f"- recommended_actions: {len(actions)}")
    print(f"- evidence_pack: {evidence_count}")
    for idx, action in enumerate(actions[:3], start=1):
        title = str(action.get("title", "")).strip()
        detail = str(action.get("detail", "")).strip()
        print(f"  {idx}. {title} :: {detail[:120]}")

    user_guidance = analysis_result.get("user_guidance", {})
    if isinstance(user_guidance, dict) and user_guidance:
        print("\n=== User Guidance ===")
        plain_summary = str(user_guidance.get("plain_summary", "")).strip()
        if plain_summary:
            print(f"- 요약: {plain_summary}")

        next_steps = user_guidance.get("next_steps", [])
        if isinstance(next_steps, list) and next_steps:
            print("- 바로 할 일:")
            for step in next_steps[:3]:
                print(f"  {str(step).strip()}")

        evidence_guide = user_guidance.get("evidence_guide", [])
        if isinstance(evidence_guide, list) and evidence_guide:
            print("- 근거 설명:")
            for item in evidence_guide[:3]:
                if not isinstance(item, dict):
                    continue
                ref_token = str(item.get("ref_token", "")).strip()
                source_label = str(item.get("source_label", "")).strip()
                title = str(item.get("title", "")).strip()
                print(f"  - {ref_token} -> {source_label} / {title}")

    output_path = Path(args.output).resolve() if args.output else _default_output_path().resolve()
    global_state = {
        "thread_id": thread_id,
        "denial_file_path": denial_file_path,
        "join_date": join_date,
        "policy_date": selected_policy_date,
        "decision_summary": onboarding_state.get("decision_summary", {}),
        "decision_explanation": onboarding_state.get("decision_explanation", ""),
        "onboarding_state": onboarding_state,
        "structured_case": structured_case,
        "analysis_result": analysis_result,
    }
    _save_output(output_path, global_state)
    print(f"\n[done] saved: {output_path}")


if __name__ == "__main__":
    main()
