"""
RetrievalTool — wraps the existing hybrid-RAG retrieval + lexical reranker.

The tool is stateless: it accepts the db session and user_id through **kwargs
so the same tool instance can be reused across agent iterations.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.agent.tools.base import BaseTool
from app.core.config import settings
from app.services.reranker_service import rerank_context
from app.services.retrieval_service import retrieve_context

logger = logging.getLogger(__name__)


class RetrievalTool(BaseTool):
    """Hybrid dense + keyword retrieval with lexical reranking."""

    def __init__(self, top_k: int | None = None, rerank_k: int | None = None) -> None:
        self._top_k = top_k or settings.retrieval_candidate_k
        self._rerank_k = rerank_k or settings.retrieval_rerank_k

    @property
    def name(self) -> str:
        return "retrieval"

    def run(self, query: str, **kwargs) -> str:
        """
        Retrieve and rerank document chunks for *query*.

        Required kwargs
        ---------------
        db : Session
        user_id : int

        Returns
        -------
        str
            Newline-separated chunk texts (empty string if nothing found).
        """
        db: Session = kwargs["db"]
        user_id: int = kwargs["user_id"]

        try:
            raw_chunks = retrieve_context(db, query, k=self._top_k, user_id=user_id)
            ranked = rerank_context(query, raw_chunks, k=self._rerank_k)
            logger.debug("RetrievalTool: query=%r → %d chunks", query, len(ranked))
            return ranked  # list[str] consumed by Executor
        except Exception as exc:
            logger.warning("RetrievalTool failed for query %r: %s", query, exc)
            return []
