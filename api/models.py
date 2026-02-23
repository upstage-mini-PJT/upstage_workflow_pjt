from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class StartWorkflowResponse(BaseModel):
    thread_id: str
    status: str = "running"


class WorkflowStatusResponse(BaseModel):
    thread_id: str
    status: str
    required_documents: list[str] = []
    required_document_items: list[dict[str, Any]] = []
    error: str | None = None


class ResumeResponse(BaseModel):
    thread_id: str
    status: str = "running"


class WorkflowResultResponse(BaseModel):
    thread_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
