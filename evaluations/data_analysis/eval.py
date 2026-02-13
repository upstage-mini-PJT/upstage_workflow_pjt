from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.data_analysis_agent.agent import run
from evaluations.data_analysis.metrics import band_validity, provenance_coverage, schema_completeness


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def evaluate(dataset_path: str = "evaluations/data_analysis/dataset.jsonl") -> dict[str, Any]:
    dataset = _load_dataset(Path(dataset_path))
    if not dataset:
        return {
            "count": 0,
            "schema_completeness_avg": 0.0,
            "provenance_coverage_avg": 0.0,
            "band_validity_rate": 0.0,
        }

    schema_scores: list[float] = []
    provenance_scores: list[float] = []
    valid_bands = 0

    for row in dataset:
        structured_case = row.get("structured_case", {})
        result = run(structured_case)
        schema_scores.append(schema_completeness(result))
        provenance_scores.append(provenance_coverage(result))
        if band_validity(result):
            valid_bands += 1

    output = {
        "count": len(dataset),
        "schema_completeness_avg": round(sum(schema_scores) / len(schema_scores), 4),
        "provenance_coverage_avg": round(sum(provenance_scores) / len(provenance_scores), 4),
        "band_validity_rate": round(valid_bands / len(dataset), 4),
    }
    return output


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
