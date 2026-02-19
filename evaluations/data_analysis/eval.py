from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agents.data_analysis_agent.agent import run
from evaluations.data_analysis.metrics import band_validity, provenance_coverage, schema_completeness

load_dotenv(dotenv_path=".env")


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
            "expected_band_accuracy": 0.0,
        }

    schema_scores: list[float] = []
    provenance_scores: list[float] = []
    valid_bands = 0
    expected_hits = 0

    for row in dataset:
        result = run(
            row.get("structured_case", {}),
            analysis_options={
                "entrypoint": "eval",
                "trace_tags": ["step3", "eval"],
                "trace_metadata": {"dataset_path": str(dataset_path)},
            },
        )
        schema_scores.append(schema_completeness(result))
        provenance_scores.append(provenance_coverage(result))

        if band_validity(result):
            valid_bands += 1

        expected_band = row.get("expected_band")
        if expected_band and result.get("success_probability", {}).get("band") == expected_band:
            expected_hits += 1

    return {
        "count": len(dataset),
        "schema_completeness_avg": round(sum(schema_scores) / len(schema_scores), 4),
        "provenance_coverage_avg": round(sum(provenance_scores) / len(provenance_scores), 4),
        "band_validity_rate": round(valid_bands / len(dataset), 4),
        "expected_band_accuracy": round(expected_hits / len(dataset), 4),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
