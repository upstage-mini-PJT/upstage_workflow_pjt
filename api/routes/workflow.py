from __future__ import annotations

import logging
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("api.routes.workflow")

# Write errors to a file so they're visible regardless of uvicorn's subprocess stdout handling
_LOG_FILE = Path(__file__).resolve().parent.parent / "api_errors.log"


def _log_error_to_file(label: str, tb: str) -> None:
    try:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n{label}\n{tb}\n")
    except Exception:
        pass

from api.models import (
    ResumeResponse,
    StartWorkflowResponse,
    WorkflowResultResponse,
    WorkflowStatusResponse,
)
from api.session_store import create_session, get_session, update_session
from api.workflow import (
    resume_onboarding_workflow,
    run_analysis_workflow,
    start_onboarding_workflow,
)

router = APIRouter()

_UPLOAD_BASE = Path(tempfile.gettempdir()) / "workflow_uploads"


def _upload_dir(thread_id: str) -> Path:
    path = _UPLOAD_BASE / thread_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Background task runners
# ---------------------------------------------------------------------------

async def _bg_run_onboarding(thread_id: str, denial_file_path: str, join_date: str) -> None:
    try:
        data = await start_onboarding_workflow(thread_id, denial_file_path, join_date)
        if data["interrupted"]:
            update_session(
                thread_id,
                status="awaiting_docs",
                runnable_config=data["config"],
                required_documents=data["required_documents"],
                required_document_ids=data["required_document_ids"],
                required_document_items=data["required_document_items"],
            )
        else:
            onboarding_state = data["graph_result"]
            result = await run_analysis_workflow(thread_id, onboarding_state)
            update_session(
                thread_id,
                status="complete",
                onboarding_state=onboarding_state,
                result=result,
            )
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("_bg_run_onboarding FAILED [thread=%s]\n%s", thread_id, tb)
        _log_error_to_file(f"_bg_run_onboarding [thread={thread_id}]", tb)
        update_session(thread_id, status="error", error=tb)


async def _bg_resume_onboarding(thread_id: str, doc_paths: list[str]) -> None:
    session = get_session(thread_id)
    if session is None or session.runnable_config is None:
        update_session(thread_id, status="error", error="Session or config not found for resume")
        return
    config = session.runnable_config
    try:
        data = await resume_onboarding_workflow(doc_paths, config)
        if data["interrupted"]:
            update_session(
                thread_id,
                status="awaiting_docs",
                required_documents=data["required_documents"],
                required_document_ids=data["required_document_ids"],
                required_document_items=data["required_document_items"],
            )
        else:
            onboarding_state = data["graph_result"]
            result = await run_analysis_workflow(thread_id, onboarding_state)
            update_session(
                thread_id,
                status="complete",
                onboarding_state=onboarding_state,
                result=result,
            )
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("_bg_resume_onboarding FAILED [thread=%s]\n%s", thread_id, tb)
        _log_error_to_file(f"_bg_resume_onboarding [thread={thread_id}]", tb)
        update_session(thread_id, status="error", error=tb)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/start", response_model=StartWorkflowResponse)
async def start_workflow(
    background_tasks: BackgroundTasks,
    denial_file: UploadFile = File(..., description="지급거절 명세서 (PDF/MD/TXT)"),
    join_date: str = Form(..., description="보험 가입일 YYYYMMDD"),
) -> StartWorkflowResponse:
    join_date = join_date.strip()
    if len(join_date) != 8 or not join_date.isdigit():
        raise HTTPException(status_code=422, detail="join_date는 YYYYMMDD 형식이어야 합니다")

    thread_id = str(uuid.uuid4())
    create_session(thread_id)
    update_session(thread_id, status="running")

    # Save uploaded file
    upload_dir = _upload_dir(thread_id)
    suffix = Path(denial_file.filename or "denial.bin").suffix or ".bin"
    denial_path = upload_dir / f"denial{suffix}"
    denial_path.write_bytes(await denial_file.read())

    background_tasks.add_task(_bg_run_onboarding, thread_id, str(denial_path), join_date)
    return StartWorkflowResponse(thread_id=thread_id, status="running")


@router.get("/status/{thread_id}", response_model=WorkflowStatusResponse)
async def get_status(thread_id: str) -> WorkflowStatusResponse:
    session = get_session(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return WorkflowStatusResponse(
        thread_id=thread_id,
        status=session.status,
        required_documents=session.required_documents,
        required_document_items=session.required_document_items,
        error=session.error,
    )


@router.post("/resume/{thread_id}", response_model=ResumeResponse)
async def resume_workflow(
    thread_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="추가 제출 서류"),
) -> ResumeResponse:
    session = get_session(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "awaiting_docs":
        raise HTTPException(
            status_code=400,
            detail=f"세션이 서류 대기 상태가 아닙니다 (현재 상태: {session.status})",
        )

    resume_dir = _upload_dir(thread_id) / "resume"
    resume_dir.mkdir(parents=True, exist_ok=True)

    doc_paths: list[str] = []
    for uploaded in files:
        filename = uploaded.filename or "doc.bin"
        dest = resume_dir / filename
        dest.write_bytes(await uploaded.read())
        doc_paths.append(str(dest))

    update_session(thread_id, status="running")
    background_tasks.add_task(_bg_resume_onboarding, thread_id, doc_paths)
    return ResumeResponse(thread_id=thread_id, status="running")


@router.get("/result/{thread_id}", response_model=WorkflowResultResponse)
async def get_result(thread_id: str) -> WorkflowResultResponse:
    session = get_session(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return WorkflowResultResponse(
        thread_id=thread_id,
        status=session.status,
        result=session.result,
        error=session.error,
    )
