from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.agent.doc_agent import DocumentAgent
from app.core.config import settings
from app.services.advanced_retrieval_service import retrieve_expanded_context
from app.services.llm_service import LLMServiceError, generate_answer
from app.services.memory_service import build_contextual_history
from app.services.query_intelligence_service import (
    QueryAnalysis,
    choose_pipeline,
    analyze_query,
    evaluate_context,
    expand_query,
)
from app.services.reranker_service import rerank_context_with_scores


class SmartChatError(RuntimeError):
    pass


@dataclass(slots=True)
class SmartChatResult:
    answer: str
    context: list[str]
    pipeline: str
    route_reason: str
    query_expansions: list[str]
    analysis: QueryAnalysis
    context_coverage: float | None = None
    context_sufficient: bool | None = None


def _run_agent_path(
    db: Session,
    query: str,
    user_id: int,
    conversation_id: int | None,
) -> tuple[str, list[str], str]:
    final_answer = ""
    final_context: list[str] = []
    route_reason = "Complex query routed to iterative agent reasoning"

    agent = DocumentAgent()
    for event in agent.run(
        goal=query,
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
    ):
        if event["type"] == "final":
            payload = event["data"]
            final_answer = payload.get("answer", "")
            final_context = payload.get("context", []) or []

    if not final_answer:
        raise SmartChatError("Agent completed without a final answer")
    return final_answer, final_context, route_reason


def _run_rag_path(
    db: Session,
    query: str,
    user_id: int,
    conversation_id: int | None,
    analysis: QueryAnalysis,
    expansions: list[str],
) -> SmartChatResult:
    history = (
        build_contextual_history(db, conversation_id, recent_limit=settings.memory_recent_limit)
        if conversation_id
        else None
    )
    candidates = retrieve_expanded_context(
        db=db,
        user_id=user_id,
        expanded_queries=expansions,
        candidate_k=settings.retrieval_candidate_k,
    )
    reranked = rerank_context_with_scores(
        query=query,
        contexts=candidates,
        k=settings.retrieval_rerank_k,
    )
    top_context = [item["content"] for item in reranked]
    avg_score = (
        (sum(item["score"] for item in reranked) / len(reranked))
        if reranked
        else None
    )
    context_eval = evaluate_context(analysis.tokens, top_context, avg_rank_score=avg_score)

    # Escalate to agent only when context is weak and query is not trivial.
    if not context_eval.sufficient and analysis.complexity_score >= 0.35:
        answer, ctx, _ = _run_agent_path(db, query, user_id, conversation_id)
        return SmartChatResult(
            answer=answer,
            context=ctx,
            pipeline="agent_fallback",
            route_reason=f"RAG context insufficient ({context_eval.reason}); escalated to agent",
            query_expansions=expansions,
            analysis=analysis,
            context_coverage=context_eval.coverage_score,
            context_sufficient=False,
        )

    try:
        answer = generate_answer(query=query, context=top_context, history=history)
    except LLMServiceError as exc:
        raise SmartChatError(str(exc)) from exc

    return SmartChatResult(
        answer=answer,
        context=top_context,
        pipeline="rag",
        route_reason=f"Direct RAG path selected ({context_eval.reason})",
        query_expansions=expansions,
        analysis=analysis,
        context_coverage=context_eval.coverage_score,
        context_sufficient=context_eval.sufficient,
    )


def run_smart_chat(
    db: Session,
    query: str,
    user_id: int,
    conversation_id: int | None = None,
) -> SmartChatResult:
    analysis = analyze_query(query)
    expansions = expand_query(analysis, limit=settings.retrieval_expansion_k)
    preferred_pipeline = choose_pipeline(analysis)
    try:
        if preferred_pipeline == "agent":
            answer, ctx, reason = _run_agent_path(db, query, user_id, conversation_id)
            return SmartChatResult(
                answer=answer,
                context=ctx,
                pipeline="agent",
                route_reason=reason,
                query_expansions=expansions,
                analysis=analysis,
            )

        return _run_rag_path(db, query, user_id, conversation_id, analysis, expansions)
    except SmartChatError:
        raise
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise SmartChatError(f"Smart pipeline failed: {exc}") from exc
