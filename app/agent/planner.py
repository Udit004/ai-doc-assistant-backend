"""
Planner — decomposes a user goal into 2-4 retrievable sub-questions.

The Planner sends a structured JSON-requesting prompt to Gemini and parses
the response into a list of SubTask objects.  If parsing fails for any
reason, it falls back to treating the full goal as a single sub-task.
"""

from __future__ import annotations

import json
import logging

import httpx

from app.agent.state import SubTask, TaskStatus
from app.core.config import settings

logger = logging.getLogger(__name__)

_PLANNER_PROMPT = """\
You are a planning assistant for a document Q&A system.
Your job is to break a user's goal into 2-4 specific, self-contained sub-questions \
that can each be answered by searching a document knowledge base independently.

Rules:
- Each sub-question must be retrievable on its own (no references to other sub-questions).
- Focus only on the user's documents — do not rely on world knowledge.
- Use at most {max_tasks} sub-questions.
- If the goal is already simple and specific, return exactly 1 sub-question.

Goal: {goal}

Respond ONLY with valid JSON in this exact format:
{{"tasks": ["sub-question 1", "sub-question 2", "sub-question 3"]}}
"""


def _call_gemini_raw(prompt: str) -> str:
    """Send a single prompt to Gemini and return the raw text response."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,           # deterministic planning
            "maxOutputTokens": 512,
        },
    }
    with httpx.Client(timeout=settings.gemini_timeout_seconds) as client:
        resp = client.post(url, params={"key": settings.gemini_api_key}, json=payload)
        resp.raise_for_status()

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()


def _parse_tasks(raw: str, goal: str, max_tasks: int) -> list[SubTask]:
    """
    Extract task strings from Gemini JSON output.

    Falls back gracefully if the model returns non-JSON or an empty list.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
        tasks_raw: list[str] = data.get("tasks") or []
    except json.JSONDecodeError:
        logger.warning("Planner: JSON parse failed, using full goal as single task")
        tasks_raw = []

    if not tasks_raw:
        tasks_raw = [goal]

    # Enforce limit
    tasks_raw = tasks_raw[:max_tasks]

    return [
        SubTask(id=i, query=q.strip(), status=TaskStatus.PENDING)
        for i, q in enumerate(tasks_raw)
        if q.strip()
    ]


class Planner:
    """Decomposes a goal into an ordered list of sub-tasks."""

    def __init__(self, max_tasks: int | None = None) -> None:
        self._max_tasks = max_tasks or settings.agent_max_subtasks

    def plan(self, goal: str) -> list[SubTask]:
        """
        Call Gemini to decompose *goal* into sub-tasks.

        Falls back to a single sub-task containing the raw goal on any error.

        Parameters
        ----------
        goal : str
            The user's full natural-language goal.

        Returns
        -------
        list[SubTask]
            Ordered list of 1-4 sub-tasks ready for the Executor.
        """
        logger.info("Planner: decomposing goal: %r", goal)
        prompt = _PLANNER_PROMPT.format(goal=goal, max_tasks=self._max_tasks)

        try:
            raw = _call_gemini_raw(prompt)
            tasks = _parse_tasks(raw, goal, self._max_tasks)
        except Exception as exc:
            logger.warning("Planner: Gemini call failed (%s), single-task fallback", exc)
            tasks = [SubTask(id=0, query=goal, status=TaskStatus.PENDING)]

        logger.info("Planner: created %d sub-tasks", len(tasks))
        for t in tasks:
            logger.debug("  [%d] %s", t.id, t.query)
        return tasks

    def plan_from_gaps(self, gaps: list[str]) -> list[SubTask]:
        """
        Convert evaluator-identified gaps into new sub-tasks for the next iteration.

        Parameters
        ----------
        gaps : list[str]
            Short descriptions of missing information.

        Returns
        -------
        list[SubTask]
            One SubTask per gap (up to max_tasks).
        """
        tasks = [
            SubTask(id=i, query=gap.strip(), status=TaskStatus.PENDING)
            for i, gap in enumerate(gaps[: self._max_tasks])
            if gap.strip()
        ]
        logger.info("Planner: created %d gap-fill sub-tasks", len(tasks))
        return tasks
