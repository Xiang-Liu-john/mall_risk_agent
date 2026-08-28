from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DB_DIR = PROJECT_DIR / "backend_db"
BACKEND_DB_PATH = BACKEND_DB_DIR / "mall_agent_backend.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def connect(db_path: Path = BACKEND_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_backend_db(db_path: Path = BACKEND_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                source_channel TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                run_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                token_usage_json TEXT,
                latency_ms INTEGER,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_runs (
                run_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                intent TEXT,
                status TEXT NOT NULL,
                use_llm INTEGER NOT NULL,
                model_list TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                latency_ms INTEGER,
                error_message TEXT,
                retrieval_latency_ms INTEGER,
                llm_latency_ms INTEGER,
                verifier_latency_ms INTEGER,
                total_latency_ms INTEGER,
                retrieved_chunk_count INTEGER,
                reranked_chunk_count INTEGER,
                model_name TEXT,
                tool_calls_json TEXT,
                tool_success_count INTEGER,
                tool_failure_count INTEGER,
                token_usage_json TEXT,
                error_type TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_steps (
                step_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL,
                detail TEXT NOT NULL,
                input_json TEXT,
                output_json TEXT,
                latency_ms INTEGER,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rag_evidence (
                evidence_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                content_preview TEXT NOT NULL,
                row_id TEXT,
                dense_score REAL,
                lexical_score REAL,
                fusion_score REAL,
                rerank_score REAL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_approvals (
                approval_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                decided_at TEXT,
                decided_by TEXT,
                decision_note TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_created
                ON conversation_messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_runs_session_started
                ON agent_runs(session_id, started_at);
            CREATE INDEX IF NOT EXISTS idx_steps_run_created
                ON agent_steps(run_id, created_at);
            """
        )
        _ensure_columns(
            conn,
            "agent_runs",
            {
                "retrieval_latency_ms": "INTEGER",
                "llm_latency_ms": "INTEGER",
                "verifier_latency_ms": "INTEGER",
                "total_latency_ms": "INTEGER",
                "retrieved_chunk_count": "INTEGER",
                "reranked_chunk_count": "INTEGER",
                "model_name": "TEXT",
                "tool_calls_json": "TEXT",
                "tool_success_count": "INTEGER",
                "tool_failure_count": "INTEGER",
                "token_usage_json": "TEXT",
                "error_type": "TEXT",
            },
        )
        _ensure_columns(
            conn,
            "rag_evidence",
            {
                "dense_score": "REAL",
                "lexical_score": "REAL",
                "fusion_score": "REAL",
                "rerank_score": "REAL",
            },
        )
        conn.commit()


def _ensure_columns(conn: sqlite3.Connection, table_name: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    for column, column_type in columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {column_type}")


def ensure_user(user_id: str, display_name: str | None = None) -> None:
    init_backend_db()
    with connect() as conn:
        now = utc_now()
        conn.execute(
            """
            INSERT INTO users (user_id, display_name, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO NOTHING
            """,
            (user_id, display_name, now),
        )
        conn.commit()


def ensure_session(
    session_id: str,
    user_id: str,
    title: str = "经营风险 Agent 会话",
    source_channel: str = "streamlit",
) -> None:
    ensure_user(user_id)
    with connect() as conn:
        now = utc_now()
        conn.execute(
            """
            INSERT INTO conversation_sessions
                (session_id, user_id, title, source_channel, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (session_id, user_id, title, source_channel, now, now),
        )
        conn.commit()


def create_agent_run(
    session_id: str,
    user_id: str,
    question: str,
    use_llm: bool,
    model_list: list[str],
) -> str:
    ensure_session(session_id, user_id)
    run_id = new_id("run")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO agent_runs
                (run_id, session_id, user_id, question, status, use_llm, model_list, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, session_id, user_id, question, "running", int(use_llm), ",".join(model_list), utc_now()),
        )
        conn.commit()
    return run_id


def finish_agent_run(
    run_id: str,
    status: str,
    intent: str | None = None,
    latency_ms: int | None = None,
    error_message: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    metrics = metrics or {}
    with connect() as conn:
        conn.execute(
            """
            UPDATE agent_runs
            SET status = ?,
                intent = COALESCE(?, intent),
                finished_at = ?,
                latency_ms = ?,
                error_message = ?,
                retrieval_latency_ms = ?,
                llm_latency_ms = ?,
                verifier_latency_ms = ?,
                total_latency_ms = ?,
                retrieved_chunk_count = ?,
                reranked_chunk_count = ?,
                model_name = ?,
                tool_calls_json = ?,
                tool_success_count = ?,
                tool_failure_count = ?,
                token_usage_json = ?,
                error_type = ?
            WHERE run_id = ?
            """,
            (
                status,
                intent,
                utc_now(),
                latency_ms,
                error_message,
                metrics.get("retrieval_latency_ms"),
                metrics.get("llm_latency_ms"),
                metrics.get("verifier_latency_ms"),
                metrics.get("total_latency_ms", latency_ms),
                metrics.get("retrieved_chunk_count"),
                metrics.get("reranked_chunk_count"),
                metrics.get("model_name"),
                json.dumps(metrics.get("tool_calls") or [], ensure_ascii=False),
                metrics.get("tool_success_count"),
                metrics.get("tool_failure_count"),
                json.dumps(metrics.get("token_usage") or {}, ensure_ascii=False),
                metrics.get("error_type"),
                run_id,
            ),
        )
        conn.commit()


def save_message(
    session_id: str,
    role: str,
    content: str,
    run_id: str | None = None,
    model: str | None = None,
    token_usage: dict[str, Any] | None = None,
    latency_ms: int | None = None,
    status: str = "ok",
) -> str:
    message_id = new_id("msg")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_messages
                (message_id, session_id, run_id, role, content, model, token_usage_json, latency_ms, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                run_id,
                role,
                content,
                model,
                json.dumps(token_usage or {}, ensure_ascii=False),
                latency_ms,
                status,
                utc_now(),
            ),
        )
        conn.execute(
            "UPDATE conversation_sessions SET updated_at = ? WHERE session_id = ?",
            (utc_now(), session_id),
        )
        conn.commit()
    return message_id


def save_agent_step(
    run_id: str,
    step_name: str,
    detail: str,
    status: str = "完成",
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    latency_ms: int | None = None,
) -> str:
    step_id = new_id("step")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO agent_steps
                (step_id, run_id, step_name, status, detail, input_json, output_json, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_id,
                run_id,
                step_name,
                status,
                detail,
                json.dumps(input_data or {}, ensure_ascii=False),
                json.dumps(output_data or {}, ensure_ascii=False),
                latency_ms,
                utc_now(),
            ),
        )
        conn.commit()
    return step_id


def save_rag_evidence(
    run_id: str,
    rank: int,
    source: str,
    title: str,
    content_preview: str,
    row_id: str | None = None,
    dense_score: float | None = None,
    lexical_score: float | None = None,
    fusion_score: float | None = None,
    rerank_score: float | None = None,
) -> str:
    evidence_id = new_id("evi")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO rag_evidence
                (evidence_id, run_id, rank, source, title, content_preview, row_id,
                 dense_score, lexical_score, fusion_score, rerank_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence_id,
                run_id,
                rank,
                source,
                title,
                content_preview,
                row_id,
                dense_score,
                lexical_score,
                fusion_score,
                rerank_score,
                utc_now(),
            ),
        )
        conn.commit()
    return evidence_id


def create_approval(run_id: str, action_type: str, target: str, status: str = "pending") -> str:
    init_backend_db()
    approval_id = new_id("apv")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO agent_approvals
                (approval_id, run_id, action_type, target, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (approval_id, run_id, action_type, target, status, utc_now()),
        )
        conn.commit()
    return approval_id


def decide_approval(
    approval_id: str,
    status: str,
    decided_by: str | None = None,
    decision_note: str | None = None,
) -> None:
    if status not in {"approved", "rejected"}:
        raise ValueError("approval status must be approved or rejected")
    with connect() as conn:
        conn.execute(
            """
            UPDATE agent_approvals
            SET status = ?, decided_at = ?, decided_by = ?, decision_note = ?
            WHERE approval_id = ?
            """,
            (status, utc_now(), decided_by, decision_note, approval_id),
        )
        conn.commit()


def list_approvals(run_id: str | None = None, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    init_backend_db()
    clauses = []
    params: list[Any] = []
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT approval_id, run_id, action_type, target, status, created_at,
                   decided_at, decided_by, decision_note
            FROM agent_approvals
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def backend_stats() -> dict[str, int | str]:
    init_backend_db()
    with connect() as conn:
        stats = {
            "db_path": str(BACKEND_DB_PATH),
            "sessions": conn.execute("SELECT COUNT(*) FROM conversation_sessions").fetchone()[0],
            "messages": conn.execute("SELECT COUNT(*) FROM conversation_messages").fetchone()[0],
            "runs": conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0],
            "steps": conn.execute("SELECT COUNT(*) FROM agent_steps").fetchone()[0],
            "evidence": conn.execute("SELECT COUNT(*) FROM rag_evidence").fetchone()[0],
            "approvals": conn.execute("SELECT COUNT(*) FROM agent_approvals").fetchone()[0],
        }
    return stats


def recent_runs(limit: int = 10) -> list[dict[str, Any]]:
    init_backend_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT run_id, session_id, user_id, question, intent, status, use_llm,
                   model_list, started_at, finished_at, latency_ms
            FROM agent_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
