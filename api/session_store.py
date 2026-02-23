from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    thread_id: str
    status: str = "pending"  # pending | running | awaiting_docs | complete | error
    required_documents: list[str] = field(default_factory=list)
    required_document_ids: list[str] = field(default_factory=list)
    required_document_items: list[dict[str, Any]] = field(default_factory=list)
    onboarding_state: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    runnable_config: Any = None  # RunnableConfig — in-memory only, not serializable


_store: dict[str, SessionState] = {}


def create_session(thread_id: str) -> SessionState:
    session = SessionState(thread_id=thread_id)
    _store[thread_id] = session
    return session


def get_session(thread_id: str) -> SessionState | None:
    return _store.get(thread_id)


def update_session(thread_id: str, **kwargs: Any) -> None:
    session = _store.get(thread_id)
    if session is None:
        return
    for key, value in kwargs.items():
        setattr(session, key, value)
