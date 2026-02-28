"""Pydantic schemas for the agent API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Request
# ──────────────────────────────────────────────

class AgentRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=4000, description="The user's high-level goal")
    conversation_id: int | None = Field(
        default=None,
        description="Existing conversation to load history from (optional)",
    )


# ──────────────────────────────────────────────
# SSE stream events
# ──────────────────────────────────────────────

class AgentStreamEvent(BaseModel):
    """
    Every Server-Sent Event the agent endpoint emits follows this shape.

    ``type`` tells the frontend which UI component to update.
    ``data`` is a free-form dict whose keys depend on the event type:

    plan        → {iteration, tasks: [{id, query}]}
    step_start  → {iteration, task_id, query}
    step_result → {iteration, task_id, query, answer, context, failed}
    eval        → {iteration, score, completeness, accuracy, sufficient, gaps}
    final       → {answer, context, iterations, final_score}
    error       → {message}
    """

    type: Literal["plan", "step_start", "step_result", "eval", "final", "error"]
    data: dict[str, Any]


# ──────────────────────────────────────────────
# Stored run (GET /agent/{run_id})
# ──────────────────────────────────────────────

class AgentRunRead(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    user_id: int | None
    conversation_id: int | None
    goal: str
    status: str
    iterations: int
    eval_score: float | None
    final_answer: str | None
    created_at: datetime
    completed_at: datetime | None


class AgentRunListItem(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    goal: str
    status: str
    iterations: int
    eval_score: float | None
    created_at: datetime
    completed_at: datetime | None
