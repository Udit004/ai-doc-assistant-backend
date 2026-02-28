"""
Agent API endpoints.

POST /agent        — Run the goal agent, stream events via SSE.
GET  /agent        — List the current user's past agent runs.
GET  /agent/{id}   — Retrieve a specific run (full detail for replay/debug).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.doc_agent import DocumentAgent
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.agent_run import AgentRun
from app.models.conversation import Conversation
from app.models.user import User
from app.schemas.agent import AgentRequest, AgentRunListItem, AgentRunRead

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sse_line(event_type: str, data: dict[str, Any]) -> str:
    """Format a single SSE message (two trailing newlines required by spec)."""
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


def _validate_conversation(
    conversation_id: int | None,
    db: Session,
    current_user: User,
) -> None:
    """Raise HTTP 404/403 if the conversation doesn't belong to the user."""
    if not conversation_id:
        return
    conv = db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this conversation")


# ──────────────────────────────────────────────────────────────────────────────
# POST /agent — streaming endpoint
# ──────────────────────────────────────────────────────────────────────────────

@router.post("")
def run_agent(
    request: AgentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Run the goal agent and stream progress events via Server-Sent Events.

    Event types (in order):
    - ``plan``        — sub-tasks created by the Planner
    - ``step_start``  — a sub-task is about to be executed
    - ``step_result`` — a sub-task completed (includes retrieved chunks + answer)
    - ``eval``        — the Evaluator's quality assessment
    - ``final``       — the Synthesizer's merged answer (last event)
    - ``error``       — emitted if an unrecoverable error occurs
    """
    _validate_conversation(request.conversation_id, db, current_user)

    # Capture primitives NOW — before any commit expires the ORM object.
    # Accessing current_user.id inside the generator after a commit raises
    # DetachedInstanceError because the session flushes and detaches the object.
    user_id: int = current_user.id
    goal: str = request.goal
    conversation_id: int | None = request.conversation_id

    # Create the AgentRun record immediately so callers can track it
    run = AgentRun(
        user_id=user_id,
        conversation_id=conversation_id,
        goal=goal,
        status="running",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id

    def event_stream():
        agent = DocumentAgent()
        all_steps: list[dict] = []
        all_evals: list[dict] = []
        current_plan: list[dict] = []
        final_score: float | None = None
        final_answer: str | None = None
        iterations = 0
        error_msg: str | None = None

        try:
            for event in agent.run(
                goal=goal,
                db=db,
                user_id=user_id,
                conversation_id=conversation_id,
            ):
                evt_type: str = event["type"]
                evt_data: dict = event["data"]

                # Accumulate data for DB persistence
                if evt_type == "plan":
                    current_plan = evt_data.get("tasks", [])
                    iterations = evt_data.get("iteration", iterations)
                elif evt_type == "step_result":
                    all_steps.append(evt_data)
                elif evt_type == "eval":
                    all_evals.append(evt_data)
                    final_score = evt_data.get("score")
                    iterations = evt_data.get("iteration", iterations)
                elif evt_type == "final":
                    iterations = evt_data.get("iterations", iterations)
                    final_score = evt_data.get("final_score", final_score)
                    final_answer = evt_data.get("answer")

                yield _sse_line(evt_type, evt_data)

        except Exception as exc:
            logger.exception("DocumentAgent run %d failed: %s", run_id, exc)
            error_msg = str(exc)
            yield _sse_line("error", {"message": error_msg, "run_id": run_id})

        finally:
            # Persist the run result regardless of success/failure
            try:
                persisted_run = db.get(AgentRun, run_id)
                if persisted_run:
                    persisted_run.status = "failed" if error_msg else "completed"
                    persisted_run.iterations = iterations
                    persisted_run.eval_score = final_score
                    persisted_run.plan_json = json.dumps(current_plan)
                    persisted_run.steps_json = json.dumps(all_steps)
                    persisted_run.eval_history_json = json.dumps(all_evals)
                    persisted_run.final_answer = final_answer
                    persisted_run.completed_at = datetime.now(timezone.utc)
                    db.commit()
            except Exception as persist_exc:
                logger.warning("Failed to persist agent run %d: %s", run_id, persist_exc)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # disable nginx buffering
            "X-Agent-Run-Id": str(run_id),  # so clients know the run ID immediately
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /agent — list runs
# ──────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[AgentRunListItem])
def list_agent_runs(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the most recent agent runs for the current user."""
    stmt = (
        select(AgentRun)
        .where(AgentRun.user_id == current_user.id)
        .order_by(AgentRun.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    runs = db.execute(stmt).scalars().all()
    return runs


# ──────────────────────────────────────────────────────────────────────────────
# GET /agent/{run_id} — single run detail
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{run_id}", response_model=AgentRunRead)
def get_agent_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve a specific agent run by ID (for replay/debug)."""
    run = db.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    if run.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this run")
    return run
