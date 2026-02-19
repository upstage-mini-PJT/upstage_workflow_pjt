from __future__ import annotations

import json
from pathlib import Path

from tools.data_analysis_tools.caselaw.kca_loader import (
    DEFAULT_KCA_PATH,
    filter_insurance_cases,
    load_kca_raw_cases,
    load_normalized_kca_disputes,
)


def main() -> None:
    out_dir = Path("data/data_analysis_data/converted_cases")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_kca_raw_cases(DEFAULT_KCA_PATH)
    filtered_rows = filter_insurance_cases(raw_rows)
    normalized_rows = load_normalized_kca_disputes(DEFAULT_KCA_PATH, insurance_only=True)

    filtered_path = out_dir / "kca_finance_insurance_cases_filtered.json"
    normalized_path = out_dir / "kca_finance_insurance_cases_filtered_normalized.json"
    stats_path = out_dir / "kca_finance_insurance_cases_filtered_stats.json"

    filtered_path.write_text(json.dumps(filtered_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    normalized_path.write_text(json.dumps(normalized_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "source_file": str(DEFAULT_KCA_PATH),
        "total_rows": len(raw_rows),
        "filtered_rows": len(filtered_rows),
        "normalized_rows": len(normalized_rows),
        "filter_ratio": round((len(filtered_rows) / len(raw_rows)), 4) if raw_rows else 0.0,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
