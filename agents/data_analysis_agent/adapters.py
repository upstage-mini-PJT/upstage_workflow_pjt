from __future__ import annotations

from core.schemas.analysis import EvidencePackItem
from tools.data_analysis_tools.caselaw.types import RankedCaseLawDoc


def ranked_case_to_evidence_item(doc: RankedCaseLawDoc, issue_id: str) -> EvidencePackItem:
    return EvidencePackItem(
        evidence_id=doc.get("doc_id", "unknown"),
        issue_id=issue_id,
        evidence_title=doc.get("title", ""),
        summary=doc.get("summary", ""),
        relevance_score=float(doc.get("relevance_score", 0.0)),
        provenance=doc.get("provenance", []),
    )
