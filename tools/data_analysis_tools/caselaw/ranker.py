from __future__ import annotations

import re

from core.schemas.case_context import StructuredCase
from tools.data_analysis_tools.caselaw.types import CaseLawDoc, RankedCaseLawDoc


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
                " ".join(doc.get("keywords", [])),
            ]
        )
        doc_tokens = _tokenize(text)
        overlap = len(target_tokens & doc_tokens)
        denominator = max(len(target_tokens), 1)
        score = round(overlap / denominator, 4)

        ranked.append(
            RankedCaseLawDoc(
                doc_id=doc.get("doc_id", ""),
                title=doc.get("title", ""),
                summary=doc.get("summary", ""),
                holding=doc.get("holding", ""),
                keywords=doc.get("keywords", []),
                source=doc.get("source", ""),
                provenance=doc.get("provenance", []),
                relevance_score=score,
            )
        )

    ranked.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    return ranked[:top_k]
