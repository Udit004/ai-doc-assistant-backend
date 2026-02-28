"""
Agent state dataclasses.

These are plain Python dataclasses (no ORM, no Pydantic) so they can be
passed around cheaply between the planner, executor, evaluator and
synthesizer without any serialization overhead at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class SubTask:
    """One atomic sub-question produced by the Planner."""

    id: int
    query: str
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class StepResult:
    """Output of Executor for a single SubTask."""

    task_id: int
    query: str
    context: list[str]       # retrieved + reranked document chunks
    answer: str              # LLM answer for this sub-question
    failed: bool = False     # True when retrieval or LLM call errored


@dataclass
class EvalResult:
    """Output of Evaluator — Gemini's self-assessment of the combined answer."""

    score: float              # 0–10 overall quality score
    completeness: int         # 1–5
    accuracy: int             # 1–5
    gaps: list[str]           # gap descriptions → become queries for next iteration
    sufficient: bool          # True when score >= configured threshold


@dataclass
class AgentState:
    """Full mutable state for one agent run; threaded through the loop."""

    goal: str
    plan: list[SubTask] = field(default_factory=list)
    results: list[StepResult] = field(default_factory=list)
    eval_history: list[EvalResult] = field(default_factory=list)
    iterations: int = 0
    final_answer: str = ""
    error: str = ""
