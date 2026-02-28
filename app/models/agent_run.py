"""
AgentRun — persists one complete agent execution to the database.

Every field that would be a Python object (plan, steps, eval_history) is
stored as JSON text so the table needs no schema changes as the agent
evolves — we can add fields to the JSON payloads freely.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Ownership
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    conversation_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Core fields
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="running", index=True
    )  # running | completed | failed

    # Agent internals — stored as JSON strings
    plan_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    steps_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    eval_history_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Summary stats
    iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    eval_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Final output
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships (backref-free — avoids conflicts with existing models)
    user: Mapped[Any] = relationship("User", foreign_keys=[user_id], lazy="select")
    conversation: Mapped[Any] = relationship(
        "Conversation", foreign_keys=[conversation_id], lazy="select"
    )
