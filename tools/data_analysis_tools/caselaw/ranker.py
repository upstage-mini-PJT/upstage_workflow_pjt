from __future__ import annotations

import re

from core.schemas.case_context import StructuredCase
from tools.data_analysis_tools.caselaw.types import CaseLawDoc, RankedCaseLawDoc


WIN_HINTS = ("승소", "인용", "취소", "일부 인용", "조정 성립")


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.split(r"\W+", text.lower()) if tok}


def rank_cases(
    cases: list[CaseLawDoc],
    structured_case: StructuredCase,
    top_k: int = 5,
) -> list[RankedCaseLawDoc]:
    target_text = " ".join(
        structured_case.get("denial_reasons", [])
        + structured_case.get("policy_clauses", [])
        + [structured_case.get("denial_summary", "")]
    )
    target_tokens = _tokenize(target_text)

    ranked: list[RankedCaseLawDoc] = []
    for doc in cases:
        text = " ".join(
            [
                doc.get("title", ""),
                doc.get("summary", ""),
                doc.get("holding", ""),
                doc.get("result", ""),
                " ".join(doc.get("keywords", [])),
            ]
        )
        doc_tokens = _tokenize(text)
        matched_tokens = list(target_tokens & doc_tokens)
        overlap = len(matched_tokens)
        denominator = max(len(target_tokens), 1)

        base = overlap / denominator
        result_text = doc.get("result", "")
        win_boost = 0.08 if any(hint in result_text for hint in WIN_HINTS) else 0.0
        score = round(min(1.0, base + win_boost), 4)

        ranked.append(
            RankedCaseLawDoc(
                doc_id=doc.get("doc_id", ""),
                title=doc.get("title", ""),
                summary=doc.get("summary", ""),
                holding=doc.get("holding", ""),
                result=doc.get("result", ""),
                keywords=doc.get("keywords", []),
                source=doc.get("source", ""),
                source_type=doc.get("source_type", "caselaw"),
                provenance=doc.get("provenance", []),
                relevance_score=score,
                matched_keywords=matched_tokens,
            )
        )

    ranked.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    return ranked[:top_k]
