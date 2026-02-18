from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class HyDEGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class HyDEConfig:
    max_chars: int = 700
    include_open_questions: bool = True


def generate_hypothetical_doc(
    query: str,
    structured_case: dict,
    generator: HyDEGenerator | None = None,
    config: HyDEConfig | None = None,
) -> str:
    cfg = config or HyDEConfig()
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")

    prompt = _build_hyde_prompt(clean_query, structured_case, cfg)
    if generator is None:
        return _truncate(_build_local_hypothesis(clean_query, structured_case, cfg), cfg.max_chars)

    try:
        generated = generator.generate(prompt).strip()
    except Exception:
        generated = ""

    if not generated:
        return _truncate(_build_local_hypothesis(clean_query, structured_case, cfg), cfg.max_chars)
    return _truncate(generated, cfg.max_chars)


def _build_hyde_prompt(query: str, structured_case: dict, cfg: HyDEConfig) -> str:
    reasons = ", ".join([str(x) for x in structured_case.get("denial_reasons", [])])
    clauses = ", ".join([str(x) for x in structured_case.get("policy_clauses", [])])
    summary = str(structured_case.get("denial_summary", ""))
    open_questions = ", ".join([str(x) for x in structured_case.get("open_questions", [])])

    base = (
        "아래 사건을 근거로 판례 검색용 가설 문서를 한국어로 4~6문장 작성하세요. "
        "사실관계, 쟁점, 예상 반박 논리를 포함하고 불필요한 장식은 제외하세요.\n"
        f"질문: {query}\n"
        f"사건요약: {summary}\n"
        f"부지급사유: {reasons}\n"
        f"약관쟁점: {clauses}"
    )
    if cfg.include_open_questions and open_questions:
        base += f"\n미해결질문: {open_questions}"
    return base


def _build_local_hypothesis(query: str, structured_case: dict, cfg: HyDEConfig) -> str:
    summary = str(structured_case.get("denial_summary", "")).strip()
    reasons = [str(x).strip() for x in structured_case.get("denial_reasons", []) if str(x).strip()]
    clauses = [str(x).strip() for x in structured_case.get("policy_clauses", []) if str(x).strip()]
    evidence_rows = structured_case.get("evidence_summary", [])
    evidence_titles = [str(x.get("title", "")).strip() for x in evidence_rows if str(x.get("title", "")).strip()]

    lines: list[str] = [
        f"검색 질문은 '{query}'이며, 본 사안은 보험금 부지급 판단의 정당성을 다투는 사건이다.",
    ]
    if summary:
        lines.append(f"사건 개요는 다음과 같다: {summary}")
    if reasons:
        lines.append(f"주요 쟁점은 {', '.join(reasons)}이며 보험사의 입증 범위를 중점적으로 검토한다.")
    if clauses:
        lines.append(f"약관 해석 쟁점은 {', '.join(clauses)} 조항의 적용 범위와 문언 해석이다.")
    if evidence_titles:
        lines.append(f"핵심 증빙으로는 {', '.join(evidence_titles[:3])} 등이 존재하며 사실관계 보강의 근거가 된다.")

    if cfg.include_open_questions:
        open_questions = [str(x).strip() for x in structured_case.get("open_questions", []) if str(x).strip()]
        if open_questions:
            lines.append(f"추가 검토 포인트는 {', '.join(open_questions[:2])}이며 유사 판례의 판단기준과 비교가 필요하다.")

    lines.append("따라서 유사 판례 탐색 시 면책조항 엄격해석, 인과관계 판단, 입증책임 배분 기준을 우선 검색한다.")
    return " ".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
