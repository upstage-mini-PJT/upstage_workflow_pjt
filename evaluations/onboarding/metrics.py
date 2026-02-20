from __future__ import annotations

from itertools import combinations
from typing import Any


def _group_by_scenario(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id", "")).strip() or "unknown"
        grouped.setdefault(scenario_id, []).append(row)
    return grouped


def decision_exact_match_rate(rows: list[dict[str, Any]]) -> float:
    grouped = _group_by_scenario(rows)
    total = 0
    matched = 0
    for runs in grouped.values():
        signatures = [str(item.get("decision_signature", "")).strip() for item in runs if str(item.get("decision_signature", "")).strip()]
        if not signatures:
            continue
        total += len(signatures)
        baseline = signatures[0]
        matched += sum(1 for sig in signatures if sig == baseline)
    if total <= 0:
        return 0.0
    return matched / total


def required_docs_exact_match_rate(rows: list[dict[str, Any]]) -> float:
    grouped = _group_by_scenario(rows)
    total = 0
    matched = 0
    for runs in grouped.values():
        normalized = [
            tuple(sorted({str(item).strip() for item in (run.get("required_document_ids") or []) if str(item).strip()}))
            for run in runs
        ]
        if not normalized:
            continue
        total += len(normalized)
        baseline = normalized[0]
        matched += sum(1 for value in normalized if value == baseline)
    if total <= 0:
        return 0.0
    return matched / total


def sufficiency_flip_rate(rows: list[dict[str, Any]]) -> float:
    grouped = _group_by_scenario(rows)
    evaluated = 0
    flips = 0
    for runs in grouped.values():
        values = [run.get("evidence_sufficient") for run in runs if isinstance(run.get("evidence_sufficient"), bool)]
        if not values:
            continue
        evaluated += 1
        if any(values) and not all(values):
            flips += 1
    if evaluated <= 0:
        return 0.0
    return flips / evaluated


def confidence_match_rate(rows: list[dict[str, Any]]) -> float:
    grouped = _group_by_scenario(rows)
    total = 0
    matched = 0
    for runs in grouped.values():
        values = [str(run.get("final_plan_confidence", "")).strip().lower() for run in runs]
        values = [value for value in values if value]
        if not values:
            continue
        total += len(values)
        baseline = values[0]
        matched += sum(1 for value in values if value == baseline)
    if total <= 0:
        return 0.0
    return matched / total


def retrieval_topk_jaccard_mean(rows: list[dict[str, Any]]) -> float:
    grouped = _group_by_scenario(rows)
    scores: list[float] = []
    for runs in grouped.values():
        sources = [set(run.get("retrieval_top_source_ids") or []) for run in runs]
        if len(sources) < 2:
            continue
        for a, b in combinations(sources, 2):
            union = a | b
            if not union:
                scores.append(1.0)
                continue
            scores.append(len(a & b) / len(union))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

