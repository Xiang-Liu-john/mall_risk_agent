from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from .store import (
    backend_stats,
    create_agent_run,
    decide_approval,
    ensure_session,
    finish_agent_run,
    list_approvals,
    recent_runs,
    save_message,
)


app = FastAPI(title="Mall Risk Agent Backend", version="0.1.0")


class SessionPayload(BaseModel):
    session_id: str
    user_id: str
    title: str = "经营风险 Agent 会话"
    source_channel: str = "api"


class MessagePayload(BaseModel):
    session_id: str
    role: str
    content: str
    run_id: str | None = None
    model: str | None = None
    token_usage: dict[str, Any] | None = None
    latency_ms: int | None = None
    status: str = "ok"


class RunPayload(BaseModel):
    session_id: str
    user_id: str
    question: str
    use_llm: bool = True
    model_list: list[str] = []


class FinishRunPayload(BaseModel):
    status: str
    intent: str | None = None
    latency_ms: int | None = None
    error_message: str | None = None
    metrics: dict[str, Any] | None = None


class ApprovalDecisionPayload(BaseModel):
    status: str
    decided_by: str | None = None
    decision_note: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict[str, int | str]:
    return backend_stats()


@app.get("/approvals")
def approvals(run_id: str | None = None, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return list_approvals(run_id=run_id, status=status, limit=limit)


@app.get("/runs/recent")
def list_recent_runs(limit: int = 10) -> list[dict[str, Any]]:
    return recent_runs(limit=limit)


@app.post("/sessions")
def upsert_session(payload: SessionPayload) -> dict[str, str]:
    ensure_session(payload.session_id, payload.user_id, payload.title, payload.source_channel)
    return {"session_id": payload.session_id, "status": "ok"}


@app.post("/messages")
def create_message(payload: MessagePayload) -> dict[str, str]:
    message_id = save_message(
        session_id=payload.session_id,
        role=payload.role,
        content=payload.content,
        run_id=payload.run_id,
        model=payload.model,
        token_usage=payload.token_usage,
        latency_ms=payload.latency_ms,
        status=payload.status,
    )
    return {"message_id": message_id, "status": "ok"}


@app.post("/runs")
def create_run(payload: RunPayload) -> dict[str, str]:
    run_id = create_agent_run(
        session_id=payload.session_id,
        user_id=payload.user_id,
        question=payload.question,
        use_llm=payload.use_llm,
        model_list=payload.model_list,
    )
    return {"run_id": run_id, "status": "running"}


@app.patch("/runs/{run_id}")
def finish_run(run_id: str, payload: FinishRunPayload) -> dict[str, str]:
    finish_agent_run(
        run_id=run_id,
        status=payload.status,
        intent=payload.intent,
        latency_ms=payload.latency_ms,
        error_message=payload.error_message,
        metrics=payload.metrics,
    )
    return {"run_id": run_id, "status": payload.status}


@app.patch("/approvals/{approval_id}")
def decide_approval_endpoint(approval_id: str, payload: ApprovalDecisionPayload) -> dict[str, str]:
    decide_approval(
        approval_id,
        status=payload.status,
        decided_by=payload.decided_by,
        decision_note=payload.decision_note,
    )
    return {"approval_id": approval_id, "status": payload.status}
