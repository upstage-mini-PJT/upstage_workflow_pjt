from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
import sys
import uuid
from typing import Any, cast

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
            "decision_explanation": str(onboarding_state.get("decision_explanation", "")).strip(),
            "decision_summary_user_situation": str(decision_summary.get("user_situation", "")).strip(),
            "decision_summary_insurer_claim": str(decision_summary.get("insurer_claim", "")).strip(),
            "decision_summary_conclusion_reason": str(decision_summary.get("conclusion_reason", "")).strip(),
            "required_documents": [str(x).strip() for x in onboarding_state.get("required_documents", []) if str(x).strip()],
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


def _analysis_options_from_env(thread_id: str) -> dict[str, Any]:
    risk_level = str(os.getenv("STEP3_RISK_LEVEL", "BALANCED")).strip().upper() or "BALANCED"
    output_style = str(os.getenv("STEP3_OUTPUT_STYLE", "USER_READABLE")).strip().upper() or "USER_READABLE"
    trace_tags_raw = str(os.getenv("STEP3_TRACE_TAGS", "")).strip()
    trace_tags = [token.strip() for token in trace_tags_raw.split(",") if token.strip()] if trace_tags_raw else []
    options: dict[str, Any] = {
        "risk_level": risk_level,
        "output_style": output_style,
        "thread_id": thread_id,
        "entrypoint": "main",
    }
    if trace_tags:
        options["trace_tags"] = trace_tags
    return options


def _mask_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return text

    def _mask_token(token: str) -> str:
        token = token.strip()
        if not token:
            return token
        if re.fullmatch(r"[가-힣]{2,4}", token):
            return token[0] + ("*" * (len(token) - 1))
        if re.fullmatch(r"[A-Za-z][A-Za-z'.-]{1,}", token):
            return token[0] + ("*" * (len(token) - 1))
        return token

    parts = re.split(r"(\s+)", text)
    return "".join(_mask_token(part) if idx % 2 == 0 else part for idx, part in enumerate(parts))


def _mask_digits_keep_tail(text: str, *, keep_tail: int = 2) -> str:
    chars = list(str(text or ""))
    digit_positions = [idx for idx, ch in enumerate(chars) if ch.isdigit()]
    if not digit_positions:
        return "".join(chars)
    keep_set = set(digit_positions[-max(0, keep_tail) :]) if keep_tail > 0 else set()
    for idx in digit_positions:
        if idx not in keep_set:
            chars[idx] = "*"
    return "".join(chars)


def _mask_alnum_keep_tail(text: str, *, keep_tail: int = 2) -> str:
    chars = list(str(text or ""))
    positions = [idx for idx, ch in enumerate(chars) if ch.isalnum()]
    if not positions:
        return "".join(chars)
    keep_set = set(positions[-max(0, keep_tail) :]) if keep_tail > 0 else set()
    for idx in positions:
        if idx not in keep_set:
            chars[idx] = "*"
    return "".join(chars)


def _mask_email_in_text(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        local = match.group("local")
        domain = match.group("domain")
        masked_local = local[0] + ("*" * max(len(local) - 1, 1))
        domain_parts = domain.split(".")
        if domain_parts:
            head = domain_parts[0]
            domain_parts[0] = head[0] + ("*" * max(len(head) - 1, 1)) if head else "***"
        return f"{masked_local}@{'.'.join(domain_parts)}"

    return re.sub(
        r"(?P<local>[A-Za-z0-9._%+-]+)@(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        _repl,
        str(text or ""),
    )


def _mask_phone_in_text(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}-****-**{match.group(3)[-2:]}"

    return re.sub(r"\b(01[0-9]|0[2-9][0-9]?)-?(\d{3,4})-?(\d{4})\b", _repl, str(text or ""))


def _mask_pii_text(text: str) -> str:
    masked = str(text or "")
    if not masked:
        return masked

    label_pattern = re.compile(
        r"(?P<label>"
        r"(?:수신|성명|이름|환자명|피보험자|계약자|수익자|작성자|대상자|생년월일|주민등록번호|"
        r"계약번호|증권번호|전화번호|휴대전화|연락처|이메일|주소|계좌번호)\s*[:：]\s*)"
        r"(?P<value>[^\n]+)"
    )

    def _label_repl(match: re.Match[str]) -> str:
        label = match.group("label")
        value = match.group("value").strip()
        normalized_label = label.replace(" ", "")
        if any(key in normalized_label for key in ("성명", "이름", "환자명", "피보험자", "계약자", "수익자", "작성자", "대상자", "수신")):
            masked_value = _mask_name(value)
        elif "주민등록번호" in normalized_label:
            masked_value = re.sub(r"(\d{6})[- ]?(\d{7})", r"\1-*******", value)
        elif any(key in normalized_label for key in ("전화번호", "휴대전화", "연락처")):
            masked_value = _mask_phone_in_text(value)
        elif "이메일" in normalized_label:
            masked_value = _mask_email_in_text(value)
        elif any(key in normalized_label for key in ("계약번호", "증권번호", "계좌번호")):
            masked_value = _mask_alnum_keep_tail(value, keep_tail=2)
        elif "생년월일" in normalized_label:
            masked_value = re.sub(r"\b(\d{4})[-./](\d{2})[-./](\d{2})\b", r"\1-**-**", value)
            masked_value = re.sub(r"\b(\d{4})(\d{2})(\d{2})\b", r"\1****", masked_value)
        elif "주소" in normalized_label:
            core = value[:6]
            masked_value = f"{core}***" if value else value
        else:
            masked_value = value
        return f"{label}{masked_value}"

    masked = label_pattern.sub(_label_repl, masked)
    masked = re.sub(
        r"(계약번호|증권번호|계좌번호)\s*[:：]\s*([A-Za-z0-9-]+)",
        lambda m: f"{m.group(1)}: {_mask_alnum_keep_tail(m.group(2), keep_tail=2)}",
        masked,
    )
    masked = re.sub(r"\b(\d{6})[- ]?([1-4]\d{6})\b", r"\1-*******", masked)
    masked = _mask_phone_in_text(masked)
    masked = _mask_email_in_text(masked)
    masked = re.sub(r"\b([가-힣]{2,4})(?=\s*고객님\b)", lambda m: _mask_name(m.group(1)), masked)
    return masked


def _mask_payload_recursive(value: Any) -> Any:
    if isinstance(value, str):
        return _mask_pii_text(value)
    if isinstance(value, list):
        return [_mask_payload_recursive(item) for item in value]
    if isinstance(value, dict):
        return {key: _mask_payload_recursive(item) for key, item in value.items()}
    return value


def _save_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _mask_payload_recursive(payload)
    path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"run_{timestamp}.json"


def _to_one_line(text: str, max_chars: int = 120) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return "정보 없음"
    return normalized


def _render_step2_brief_5lines(onboarding_state: dict[str, Any]) -> str:
    decision_summary = onboarding_state.get("decision_summary", {})
    if not isinstance(decision_summary, dict):
        decision_summary = {}

    user_situation = _to_one_line(str(decision_summary.get("user_situation", "")).strip(), max_chars=140)
    insurer_claim = _to_one_line(str(decision_summary.get("insurer_claim", "")).strip(), max_chars=140)
    conclusion_reason = _to_one_line(str(decision_summary.get("conclusion_reason", "")).strip(), max_chars=140)

    extracted_infos = onboarding_state.get("extracted_document_infos", [])
    evidence_docs = len(extracted_infos) if isinstance(extracted_infos, list) else 0
    evidence_sufficient = bool(onboarding_state.get("evidence_sufficient", False))
    evidence_status = "충분" if evidence_sufficient else "보강 필요"

    lines = [
        f"- 사용자 상황: {user_situation}",
        f"- 보험사 주장: {insurer_claim}",
        f"- 결론 근거: {conclusion_reason}",
        f"- 증빙 상태: {evidence_status} (추출 문서 {evidence_docs}건)",
        "- 세부 쟁점 안내: Step3 '한눈 요약 > 핵심 쟁점'을 확인하세요.",
    ]
    return "\n".join(lines)


def _render_user_guidance_4sections(user_guidance: dict[str, Any]) -> str:
    def _norm_lines(text: str) -> list[str]:
        return [line.strip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]

    lines: list[str] = []
    lines.append("1. 한눈 요약")
    plain_summary = str(user_guidance.get("plain_summary", "")).strip()
    summary_lines = _norm_lines(plain_summary)
    if summary_lines:
        for row in summary_lines:
            lines.append(row if row.startswith("- ") else f"- {row}")
    else:
        lines.append("- 요약 정보를 생성하지 못했습니다.")

    issue_brief = user_guidance.get("issue_brief", [])
    if isinstance(issue_brief, list) and issue_brief:
        lines.append("- 핵심 쟁점:")
        for item in issue_brief[:3]:
            if not isinstance(item, dict):
                continue
            issue_id = str(item.get("issue_id", "")).strip() or "ISSUE-UNKNOWN"
            title = str(item.get("title", "")).strip() or "쟁점 제목 정보 없음"
            why = str(item.get("why_it_matters", "")).strip() or "핵심 판단 기준과 직접 연결됩니다."
            needed = str(item.get("needed_evidence", "")).strip() or "진단서/진료기록"
            lines.append(f"  - {issue_id} | {title}")
            lines.append(f"    쟁점 포인트: {why}")
            lines.append(f"    준비 증빙: {needed}")

    lines.append("")
    lines.append("2. 지금 할 일")
    next_steps = user_guidance.get("next_steps", [])
    rendered_steps = 0
    if isinstance(next_steps, list):
        for step in next_steps[:5]:
            step_lines = _norm_lines(step)
            if not step_lines:
                continue
            lines.append(step_lines[0])
            for extra in step_lines[1:]:
                lines.append(f"   {extra}")
            lines.append("")
            rendered_steps += 1
    if lines and lines[-1] == "":
        lines.pop()
    if rendered_steps == 0:
        lines.append("- 바로 실행 가능한 단계가 아직 생성되지 않았습니다.")

    lines.append("")
    lines.append("3. 근거 설명")
    evidence_guide = user_guidance.get("evidence_guide", [])
    rendered_evidence = 0
    if isinstance(evidence_guide, list):
        for item in evidence_guide[:5]:
            if not isinstance(item, dict):
                continue
            ref_token = str(item.get("ref_token", "")).strip() or "CASELAW:UNKNOWN"
            source_label = str(item.get("source_label", "")).strip() or "판례"
            title = str(item.get("title", "")).strip() or "제목 정보 없음"
            why_relevant = str(item.get("why_relevant", "")).strip()
            lines.append(f"- {ref_token} | {source_label} | {title}")
            if why_relevant:
                lines.append(f"  {why_relevant}")
            rendered_evidence += 1
    if rendered_evidence == 0:
        lines.append("- 근거 정보가 충분하지 않아 기본 전략으로 진행합니다.")

    lines.append("")
    lines.append("4. 안내")
    disclaimer = str(user_guidance.get("disclaimer", "")).strip()
    if disclaimer:
        for row in _norm_lines(disclaimer):
            lines.append(row if row.startswith("- ") else f"- {row}")
    else:
        lines.append("- 본 결과는 참고용입니다. 최종 판단은 전문가와 함께 진행하세요.")

    return "\n".join(lines).strip()


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
    masked_onboarding_state = cast(dict[str, Any], _mask_payload_recursive(onboarding_state))

    print("\n=== Step2 Result Summary ===")
    print(_render_step2_brief_5lines(masked_onboarding_state))

    structured_case = _build_structured_case(onboarding_state)

    analysis_result = run_data_analysis(
        structured_case,
        rag_result=None,
        analysis_options=_analysis_options_from_env(thread_id),
    )
    masked_analysis_result = cast(dict[str, Any], _mask_payload_recursive(analysis_result))

    actions = masked_analysis_result.get("recommended_actions", [])
    band = (masked_analysis_result.get("success_probability") or {}).get("band", "UNKNOWN")
    evidence_count = len(masked_analysis_result.get("evidence_pack", []))

    print("\n=== Step3 Result Summary ===")
    print(f"- success_probability.band: {band}")
    print(f"- recommended_actions: {len(actions)}")
    print(f"- evidence_pack: {evidence_count}")
    for idx, action in enumerate(actions, start=1):
        title = str(action.get("title", "")).strip()
        detail = str(action.get("detail", "")).strip()
        print(f"  {idx}. {title} :: {detail}")

    user_guidance = masked_analysis_result.get("user_guidance", {})
    if isinstance(user_guidance, dict) and user_guidance:
        print("\n=== User Guidance ===")
        print(_render_user_guidance_4sections(user_guidance))

    masked_structured_case = cast(dict[str, Any], _mask_payload_recursive(structured_case))
    output_path = Path(args.output).resolve() if args.output else _default_output_path().resolve()
    global_state = {
        "thread_id": thread_id,
        "denial_file_path": denial_file_path,
        "join_date": join_date,
        "policy_date": selected_policy_date,
        "decision_summary": masked_onboarding_state.get("decision_summary", {}),
        "decision_explanation": masked_onboarding_state.get("decision_explanation", ""),
        "onboarding_state": masked_onboarding_state,
        "structured_case": masked_structured_case,
        "analysis_result": masked_analysis_result,
    }
    _save_output(output_path, global_state)
    print(f"\n[done] saved: {output_path}")


if __name__ == "__main__":
    main()
