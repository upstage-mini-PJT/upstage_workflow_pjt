from __future__ import annotations

from typing import Any, Literal, TypedDict


SourceType = Literal["CASELAW", "DISPUTE", "WEB"]
RetrievalMethod = Literal[
    "vector",
    "keyword",
    "hybrid",
    "vector_plain",
    "vector_hyde",
    "vector_reverse_hyde",
]
RetrievalMode = Literal["plain", "hyde", "reverse_hyde", "hybrid_hyde"]
RiskLevel = Literal["CONSERVATIVE", "BALANCED", "AGGRESSIVE"]
OutputStyle = Literal["USER_READABLE", "EXPERT_LIKE"]
NodeType = Literal[
    "ROOT",
    "GROUP",
    "DOCUMENT",
    "CHUNK",
    "ANALYSIS",
    "STRATEGY",
    "RESEARCH",
    "SCORE_STEP",
]


class TreeNode(TypedDict, total=False):
    node_id: str
    node_type: NodeType
    parent_id: str | None
    title: str
    payload: dict[str, Any]
    children: list["TreeNode"]


class IngestionDoc(TypedDict, total=False):
    doc_id: str
    source_type: SourceType
    title: str
    body: str
    published_at: str
    jurisdiction: str | None
    tags: list[str]
    url: str | None
    meta: dict[str, Any]


class VectorChunkMetadata(TypedDict, total=False):
    title: str
    published_at: str | None
    url: str | None
    tags: list[str]


class VectorChunk(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    source_type: SourceType
    chunk_text: str
    chunk_index: int
    embedding_model: str
    embedding_dim: int
    token_count: int
    metadata: VectorChunkMetadata


class RetrievalProvenance(TypedDict, total=False):
    index_name: str
    retrieved_at: str
    retrieval_method: RetrievalMethod
    query_variant: str


class RAGItem(TypedDict, total=False):
    source_type: SourceType
    doc_id: str
    chunk_id: str
    title: str
    snippet: str
    score: float
    rerank_score: float
    url: str | None
    published_at: str | None
    provenance: RetrievalProvenance


class RetrievalStats(TypedDict, total=False):
    candidate_count: int
    returned_count: int
    latency_ms: int


class RAGRetrievalResult(TypedDict, total=False):
    query_id: str
    query: str
    retrieval_mode: RetrievalMode
    query_variants: list[str]
    filters: dict[str, Any]
    items: list[RAGItem]
    tree: TreeNode
    stats: RetrievalStats


class AnalysisOptions(TypedDict, total=False):
    risk_level: RiskLevel
    output_style: OutputStyle


class ComparativeAnalysisInput(TypedDict, total=False):
    structured_case: dict[str, Any]
    structured_case_tree: TreeNode
    rag_result: RAGRetrievalResult
    analysis_options: AnalysisOptions


class ScoringTrace(TypedDict, total=False):
    precedent_score: int
    case_adjustment: int
    case_adjustment_source: Literal["llm", "heuristic", "zero"]
    total_score: int
    guardrails_applied: list[str]
    cited_case_ids: list[str]
    rationale: str
    tree: TreeNode


class ResearchPack(TypedDict, total=False):
    caselaw_results: list[dict[str, Any]]
    web_sources: list[dict[str, Any]]


class ComparativeAnalysisOutput(TypedDict, total=False):
    analysis_report: dict[str, Any]
    analysis_tree: TreeNode
    strategy_plan: dict[str, Any]
    strategy_tree: TreeNode
    research_pack: ResearchPack
    research_tree: TreeNode
    scoring_trace: ScoringTrace
