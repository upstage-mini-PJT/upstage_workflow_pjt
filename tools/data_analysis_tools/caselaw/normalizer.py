from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.schemas.provenance import Provenance
from tools.data_analysis_tools.caselaw.types import CaseLawDoc


def normalize_cases(raw_cases: list[dict[str, Any]]) -> list[CaseLawDoc]:
    normalized: list[CaseLawDoc] = []
    now = datetime.now(tz=timezone.utc).isoformat()

    for raw in raw_cases:
        source_type = str(raw.get("source_type", "CASELAW")).upper()
        provenance = Provenance(
            source_type="DISPUTE" if source_type == "DISPUTE" else "CASELAW",
            source_id=str(raw.get("doc_id", "unknown")),
            title=str(raw.get("title", "")),
            snippet=str(raw.get("summary", ""))[:280],
            retrieved_at=str(raw.get("retrieved_at", now)),
            metadata={"source": raw.get("source", "unknown")},
        )
        normalized.append(
            CaseLawDoc(
                doc_id=str(raw.get("doc_id", "unknown")),
                title=str(raw.get("title", "")),
                summary=str(raw.get("summary", "")),
                holding=str(raw.get("holding", "")),
                result=str(raw.get("result", "")),
                keywords=[str(x) for x in raw.get("keywords", [])],
                source=str(raw.get("source", "unknown")),
                source_type="DISPUTE" if source_type == "DISPUTE" else "CASELAW",
                provenance=[provenance],
            )
        )
    return normalized
