from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="drift_report에서 컨텍스트 비교 표 생성")
    parser.add_argument(
        "--input",
        default="evaluations/onboarding/live_no_fallback/drift_report.json",
        help="run_repro_eval이 생성한 drift_report.json 경로",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluations/onboarding/live_no_fallback",
        help="표 출력 디렉토리",
    )
    parser.add_argument(
        "--prefix",
        default="runs_context_table",
        help="출력 파일 접두어(.csv/.md)",
    )
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _safe_ratio(a: str, b: str) -> float:
    left = _normalize_text(a)
    right = _normalize_text(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _issue_text(row: dict[str, Any]) -> str:
    hypotheses = [str(item).strip() for item in (row.get("issue_hypotheses") or []) if str(item).strip()]
    seeds = [str(item).strip() for item in (row.get("issue_query_seeds") or []) if str(item).strip()]
    chunks: list[str] = []
    if hypotheses:
        chunks.append("hypotheses: " + " | ".join(hypotheses))
    if seeds:
        chunks.append("query_seeds: " + " | ".join(seeds))
    return " || ".join(chunks)


def _extract_text(row: dict[str, Any]) -> str:
    pieces: list[str] = []
    for info in row.get("extracted_document_infos") or []:
        if not isinstance(info, dict):
            continue
        source = str(info.get("source_name", "")).strip() or "(unknown)"
        key_data = str(info.get("key_data", "")).strip()
        evidence = str(info.get("evidence_or_grounds", "")).strip()
        helpful = str(info.get("helpful_notes", "")).strip()
        pieces.append(f"[{source}] key={key_data} / ev={evidence} / note={helpful}")
    return " || ".join(pieces)


def _extract_preview(row: dict[str, Any], *, max_items: int = 3, max_len: int = 500) -> str:
    parts: list[str] = []
    for info in list(row.get("extracted_document_infos") or [])[:max_items]:
        if not isinstance(info, dict):
            continue
        source = str(info.get("source_name", "")).strip() or "(unknown)"
        key_data = _normalize_text(str(info.get("key_data", "")))
        evidence = _normalize_text(str(info.get("evidence_or_grounds", "")))
        parts.append(f"[{source}] key={key_data} / ev={evidence}")
    joined = " || ".join(parts)
    return joined[:max_len]


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 존재하지 않습니다: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    runs = list(payload.get("runs") or [])
    if not runs:
        raise SystemExit(f"runs가 비어 있습니다: {input_path}")

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        scenario_id = str(row.get("scenario_id", "")).strip() or "unknown"
        by_scenario.setdefault(scenario_id, []).append(row)

    baselines: dict[str, dict[str, str]] = {}
    for scenario_id, items in by_scenario.items():
        ordered = sorted(items, key=lambda x: int(x.get("repeat_index", 0) or 0))
        baseline = ordered[0]
        baselines[scenario_id] = {
            "issue": _issue_text(baseline),
            "plan": _normalize_text(str(baseline.get("final_plan_text", ""))),
            "extract": _extract_text(baseline),
        }

    rows: list[dict[str, Any]] = []
    for scenario_id in sorted(by_scenario):
        for row in sorted(by_scenario[scenario_id], key=lambda x: int(x.get("repeat_index", 0) or 0)):
            issue_text = _issue_text(row)
            plan_text = _normalize_text(str(row.get("final_plan_text", "")))
            extract_text = _extract_text(row)
            base = baselines[scenario_id]

            output = {
                "scenario_id": scenario_id,
                "repeat_index": int(row.get("repeat_index", 0) or 0),
                "issue_hypotheses": " | ".join(row.get("issue_hypotheses") or []),
                "issue_query_seeds": " | ".join(row.get("issue_query_seeds") or []),
                "final_plan_preview": _normalize_text(str(row.get("final_plan_preview", "")))[:500],
                "required_document_ids": ", ".join(row.get("required_document_ids") or []),
                "parse_extract_preview": _extract_preview(row),
                "evidence_sufficient": row.get("evidence_sufficient"),
                "final_plan_confidence": row.get("final_plan_confidence"),
                "quality_flags": ", ".join(row.get("quality_flags") or []),
                "issue_sim_to_baseline": round(_safe_ratio(base["issue"], issue_text), 4),
                "plan_sim_to_baseline": round(_safe_ratio(base["plan"], plan_text), 4),
                "extract_sim_to_baseline": round(_safe_ratio(base["extract"], extract_text), 4),
            }
            rows.append(output)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.prefix}.csv"
    md_path = out_dir / f"{args.prefix}.md"

    fieldnames = [
        "scenario_id",
        "repeat_index",
        "issue_hypotheses",
        "issue_query_seeds",
        "final_plan_preview",
        "required_document_ids",
        "parse_extract_preview",
        "evidence_sufficient",
        "final_plan_confidence",
        "quality_flags",
        "issue_sim_to_baseline",
        "plan_sim_to_baseline",
        "extract_sim_to_baseline",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Runs Context Table",
        "",
        f"- total rows: {len(rows)}",
        "",
        "| scenario_id | repeat_index | issue_hypotheses | issue_query_seeds | final_plan_preview | required_document_ids | parse_extract_preview | evidence_sufficient | final_plan_confidence | quality_flags | issue_sim_to_baseline | plan_sim_to_baseline | extract_sim_to_baseline |",
        "|---|---:|---|---|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(row["scenario_id"]),
                    _markdown_escape(row["repeat_index"]),
                    _markdown_escape(row["issue_hypotheses"]),
                    _markdown_escape(row["issue_query_seeds"]),
                    _markdown_escape(row["final_plan_preview"]),
                    _markdown_escape(row["required_document_ids"]),
                    _markdown_escape(row["parse_extract_preview"]),
                    _markdown_escape(row["evidence_sufficient"]),
                    _markdown_escape(row["final_plan_confidence"]),
                    _markdown_escape(row["quality_flags"]),
                    _markdown_escape(row["issue_sim_to_baseline"]),
                    _markdown_escape(row["plan_sim_to_baseline"]),
                    _markdown_escape(row["extract_sim_to_baseline"]),
                ]
            )
            + " |"
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()

