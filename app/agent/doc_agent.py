"""
DocumentAgent — the main Plan-and-Execute loop.

Flow per run
------------
1. Planner decomposes the user's goal into N sub-tasks.
2. Executor runs each sub-task (retrieve → answer) sequentially.
3. Evaluator (LLM-as-judge) scores the combined answers.
4. If score < threshold AND iterations < max, Planner generates gap-fill
   sub-tasks and the loop repeats.
5. On exit, Synthesizer merges everything into one final answer.

All important moments are yielded as AgentEvent dicts so the API layer
can stream them to the frontend via SSE.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from sqlalchemy.orm import Session

from app.agent.evaluator import Evaluator
from app.agent.executor import Executor
from app.agent.planner import Planner
from app.agent.state import AgentState, EvalResult, StepResult, SubTask
from app.agent.synthesizer import Synthesizer
from app.core.config import settings
from app.services.memory_service import build_contextual_history

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Event type literals (match AgentStreamEvent on frontend)
# ──────────────────────────────────────────────
EVT_PLAN = "plan"
EVT_STEP_START = "step_start"
EVT_STEP_RESULT = "step_result"
EVT_EVAL = "eval"
EVT_FINAL = "final"
EVT_ERROR = "error"

AgentEvent = dict[str, Any]


class DocumentAgent:
    """
    Orchestrates the full Plan-and-Execute loop for document Q&A.

    Parameters
    ----------
    max_iterations : int
        Maximum number of plan→execute→evaluate cycles before stopping.
    eval_threshold : float
        Minimum Evaluator score (0-10) to consider the answer sufficient.
    """

    def __init__(
        self,
        max_iterations: int | None = None,
        eval_threshold: float | None = None,
    ) -> None:
        self._max_iter = max_iterations or settings.agent_max_iterations
        self._threshold = eval_threshold if eval_threshold is not None else settings.agent_eval_threshold

        self._planner = Planner()
        self._executor = Executor()
        self._evaluator = Evaluator(threshold=self._threshold)
        self._synthesizer = Synthesizer()

    # ──────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────

    def run(
        self,
        goal: str,
        db: Session,
        user_id: int,
        conversation_id: int | None = None,
    ) -> Generator[AgentEvent, None, AgentState]:
        """
        Execute the agent loop and yield SSE-ready event dicts.

        Parameters
        ----------
        goal : str
            The user's natural-language goal.
        db : Session
            Active database session (used for retrieval and memory).
        user_id : int
            ID of the authenticated user whose documents are searched.
        conversation_id : int | None
            If provided, recent conversation messages are loaded as history.

        Yields
        ------
        AgentEvent
            Dicts with ``type`` and ``data`` keys ready for SSE serialisation.

        Returns
        -------
        AgentState
            Complete final state (also accessible via the last "final" event).
        """
        state = AgentState(goal=goal)
        history = self._load_history(db, conversation_id)

        logger.info("DocumentAgent: starting run | goal=%r | max_iter=%d", goal, self._max_iter)

        while state.iterations < self._max_iter:
            state.iterations += 1
            logger.info("DocumentAgent: iteration %d/%d", state.iterations, self._max_iter)

            # ── 1. Plan ──────────────────────────────────────────────────
            if state.iterations == 1:
                plan = self._planner.plan(goal)
            else:
                # Use gaps identified by previous evaluation
                last_eval: EvalResult = state.eval_history[-1]
                if not last_eval.gaps:
                    logger.info("DocumentAgent: no gaps reported, stopping early")
                    break
                plan = self._planner.plan_from_gaps(last_eval.gaps)

            state.plan = plan
            yield self._evt(EVT_PLAN, {
                "iteration": state.iterations,
                "tasks": [{"id": t.id, "query": t.query} for t in plan],
            })

            # ── 2. Execute ───────────────────────────────────────────────
            for task in plan:
                yield self._evt(EVT_STEP_START, {
                    "iteration": state.iterations,
                    "task_id": task.id,
                    "query": task.query,
                })

                result: StepResult = self._executor.execute(
                    task=task,
                    db=db,
                    user_id=user_id,
                    history=history,
                )
                state.results.append(result)

                yield self._evt(EVT_STEP_RESULT, {
                    "iteration": state.iterations,
                    "task_id": result.task_id,
                    "query": result.query,
                    "answer": result.answer,
                    "context": result.context,
                    "failed": result.failed,
                })

            # ── 3. Evaluate ──────────────────────────────────────────────
            eval_result: EvalResult = self._evaluator.evaluate(goal, state.results)
            state.eval_history.append(eval_result)

            yield self._evt(EVT_EVAL, {
                "iteration": state.iterations,
                "score": eval_result.score,
                "completeness": eval_result.completeness,
                "accuracy": eval_result.accuracy,
                "sufficient": eval_result.sufficient,
                "gaps": eval_result.gaps,
            })

            if eval_result.sufficient:
                logger.info(
                    "DocumentAgent: sufficient at iteration %d (score=%.1f)",
                    state.iterations,
                    eval_result.score,
                )
                break

            if state.iterations >= self._max_iter:
                logger.info("DocumentAgent: max iterations reached, stopping")
                break

        # ── 4. Synthesize ────────────────────────────────────────────────
        final_answer, all_context = self._synthesizer.synthesize(goal, state.results)
        state.final_answer = final_answer

        last_eval = state.eval_history[-1] if state.eval_history else None
        yield self._evt(EVT_FINAL, {
            "answer": final_answer,
            "context": all_context,
            "iterations": state.iterations,
            "final_score": last_eval.score if last_eval else None,
        })

        logger.info(
            "DocumentAgent: done | iterations=%d | answer_len=%d",
            state.iterations,
            len(final_answer),
        )
        return state

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _evt(event_type: str, data: dict[str, Any]) -> AgentEvent:
        return {"type": event_type, "data": data}

    @staticmethod
    def _load_history(db: Session, conversation_id: int | None) -> list[str] | None:
        if not conversation_id:
            return None
        try:
            return build_contextual_history(db, conversation_id)
        except Exception as exc:
            logger.warning("DocumentAgent: failed to load history: %s", exc)
            return None
