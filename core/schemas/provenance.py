from __future__ import annotations

from typing import Any, Literal, TypedDict


class Provenance(TypedDict, total=False):
    source_type: Literal["caselaw", "dispute_case", "web", "internal_rule"]
    source_id: str
    title: str
    snippet: str
    retrieved_at: str
    metadata: dict[str, Any]
