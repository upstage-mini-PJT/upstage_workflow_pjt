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
    reverse_max_items: int = 5


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


def generate_reverse_hypothesis(
    query: str,
    retrieved_items: list[dict],
    generator: HyDEGenerator | None = None,
    config: HyDEConfig | None = None,
) -> str:
    cfg = config or HyDEConfig()
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")

    top_items = retrieved_items[: max(1, cfg.reverse_max_items)]
    prompt = _build_reverse_prompt(clean_query, top_items)

    if generator is None:
        return _truncate(_build_local_reverse_hypothesis(clean_query, top_items), cfg.max_chars)

    try:
        generated = generator.generate(prompt).strip()
    except Exception:
        generated = ""
    if not generated:
        return _truncate(_build_local_reverse_hypothesis(clean_query, top_items), cfg.max_chars)
    return _truncate(generated, cfg.max_chars)


def _build_reverse_prompt(query: str, items: list[dict]) -> str:
    snippets: list[str] = []
    for idx, item in enumerate(items, start=1):
        title = str(item.get("title", ""))
        snippet = str(item.get("snippet", ""))
        snippets.append(f"[{idx}] title={title} / snippet={snippet}")

    return (
        "다음 검색결과를 바탕으로 재검색용 가설문서를 4~6문장으로 작성하세요. "
        "중복 표현을 줄이고 핵심 법리/쟁점을 명확히 하세요.\n"
        f"질문: {query}\n"
        f"검색결과:\n{chr(10).join(snippets)}"
    )


def _build_local_reverse_hypothesis(query: str, items: list[dict]) -> str:
    if not items:
        return (
            f"검색 질문 '{query}'에 대해 판례와 분쟁사례의 핵심 판단기준을 재정리한다. "
            "면책조항 해석, 인과관계 판단, 입증책임 분배 기준을 중심으로 유사 사례를 재검색한다."
        )

    titles = [str(item.get("title", "")).strip() for item in items if str(item.get("title", "")).strip()]
    snippets = [str(item.get("snippet", "")).strip() for item in items if str(item.get("snippet", "")).strip()]
    joined_titles = ", ".join(titles[:3]) if titles else "상위 검색결과"
    short_snippet = " ".join(snippets[:2])[:220]

    return (
        f"질문 '{query}'의 재검색을 위해 상위 결과({joined_titles})를 핵심 근거로 재정리한다. "
        f"주요 요지는 다음과 같다: {short_snippet}. "
        "재검색 시 약관 문언 해석, 부지급 사유별 입증 기준, 사실관계 유사성의 판단 요소를 우선 반영한다."
    )
