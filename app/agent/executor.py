"""
Executor — runs one SubTask through the RAG pipeline and returns a StepResult.

Wires together RetrievalTool (hybrid dense+keyword) and AnswerTool (Gemini),
then packages the outputs into a StepResult dataclass.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.agent.state import StepResult, SubTask, TaskStatus
from app.agent.tools.answer import AnswerTool
from app.agent.tools.retrieval import RetrievalTool

logger = logging.getLogger(__name__)


class Executor:
    """Executes a single SubTask: retrieve → answer → StepResult."""

    def __init__(
        self,
        retrieval_tool: RetrievalTool | None = None,
        answer_tool: AnswerTool | None = None,
    ) -> None:
        self._retrieval = retrieval_tool or RetrievalTool()
        self._answer = answer_tool or AnswerTool()

    def execute(
        self,
        task: SubTask,
        db: Session,
        user_id: int,
        history: list[str] | None = None,
    ) -> StepResult:
        """
        Run retrieval + answer generation for one sub-task.

        Parameters
        ----------
        task : SubTask
            The sub-question to answer.
        db : Session
            Active database session.
        user_id : int
            Owner whose document chunks are searched.
        history : list[str] | None
            Short conversation history for context continuity.

        Returns
        -------
        StepResult
            Contains the retrieved chunks, the generated answer, and a
            ``failed`` flag so the agent loop can handle errors gracefully.
        """
        logger.info("Executor: running task [%d] %r", task.id, task.query)
        task.status = TaskStatus.RUNNING

        # Step 1 — retrieve relevant chunks
        context: list[str] = self._retrieval.run(task.query, db=db, user_id=user_id)

        # Step 2 — generate answer for this sub-question
        answer: str = self._answer.run(task.query, context=context, history=history)

        failed = answer.startswith("[ERROR:")
        task.status = TaskStatus.FAILED if failed else TaskStatus.DONE

        logger.info(
            "Executor: task [%d] → %d chunks, answer len=%d, failed=%s",
            task.id,
            len(context),
            len(answer),
            failed,
        )

        return StepResult(
            task_id=task.id,
            query=task.query,
            context=context,
            answer=answer,
            failed=failed,
        )
