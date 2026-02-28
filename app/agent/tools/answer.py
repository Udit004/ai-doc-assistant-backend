"""
AnswerTool — wraps the existing Gemini answer-generation service.

Accepts pre-retrieved context chunks and conversation history via kwargs
so it keeps the same BaseTool interface as every other tool.
"""

from __future__ import annotations

import logging

from app.agent.tools.base import BaseTool
from app.services.llm_service import LLMServiceError, generate_answer

logger = logging.getLogger(__name__)


class AnswerTool(BaseTool):
    """Calls Gemini to answer a sub-question given retrieved context."""

    @property
    def name(self) -> str:
        return "answer"

    def run(self, query: str, **kwargs) -> str:
        """
        Generate an answer for *query*.

        Required kwargs
        ---------------
        context : list[str]   — retrieved chunks (may be empty)

        Optional kwargs
        ---------------
        history : list[str]   — conversation history strings

        Returns
        -------
        str
            LLM answer, or an error sentinel string on failure.
        """
        context: list[str] = kwargs.get("context", [])
        history: list[str] | None = kwargs.get("history")

        try:
            answer = generate_answer(query=query, context=context, history=history)
            logger.debug("AnswerTool: query=%r → %d chars", query, len(answer))
            return answer
        except LLMServiceError as exc:
            logger.warning("AnswerTool failed for query %r: %s", query, exc)
            return f"[ERROR: {exc}]"
