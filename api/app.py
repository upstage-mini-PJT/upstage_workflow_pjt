from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.workflow import router as workflow_router

app = FastAPI(
    title="보험금 이의신청 AI API",
    description="LangGraph 기반 보험 분쟁 분석 시스템 REST API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8502",
        "http://127.0.0.1:8502",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workflow_router, prefix="/api/workflow", tags=["workflow"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
