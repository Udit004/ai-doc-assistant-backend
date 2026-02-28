from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import settings

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "which",
    "with",
}

_COMPLEXITY_HINTS = {
    "compare",
    "tradeoff",
    "trade-off",
    "analyze",
    "architecture",
    "design",
    "workflow",
    "end-to-end",
    "step by step",
    "plan",
    "strategy",
    "difference",
    "together",
}

_TERM_EXPANSIONS: dict[str, list[str]] = {
    "auth": ["authentication", "authorization", "token", "jwt"],
    "login": ["authentication", "token", "session"],
    "api": ["endpoint", "route", "handler"],
    "rag": ["retrieval augmented generation", "retrieval pipeline", "context chunks"],
    "vector": ["embedding", "similarity", "dense retrieval"],
    "rerank": ["re-rank", "ranking", "relevance scoring"],
    "chunk": ["document chunk", "context block"],
    "celery": ["background jobs", "task queue", "worker"],
    "db": ["database", "postgres", "sqlite", "persistence"],
}


@dataclass(slots=True)
class QueryAnalysis:
    cleaned_query: str
    tokens: list[str]
    intent_tags: list[str]
    complexity_score: float
    prefer_agent: bool


@dataclass(slots=True)
class ContextEvaluation:
    sufficient: bool
    coverage_score: float
    reason: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())


def _informative_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in _STOP_WORDS]


def analyze_query(query: str) -> QueryAnalysis:
    cleaned = " ".join(query.split()).strip()
    tokens = _tokenize(cleaned)
    info_tokens = _informative_tokens(tokens)

    low = cleaned.lower()
    hint_hits = sum(hint in low for hint in _COMPLEXITY_HINTS)
    multi_part = (
        "," in low
        or " and " in low
        or " then " in low
        or " vs " in low
        or "?" in low and low.count("?") > 1
    )
    length_factor = min(len(info_tokens) / 18.0, 1.0)

    complexity_score = min(1.0, (hint_hits * 0.22) + (0.22 if multi_part else 0.0) + (length_factor * 0.56))
    prefer_agent = complexity_score >= 0.55

    tags: list[str] = []
    if multi_part:
        tags.append("multi_part")
    if hint_hits:
        tags.append("reasoning")
    if any(token in {"how", "why"} for token in tokens):
        tags.append("explanatory")
    if not tags:
        tags.append("direct_lookup")

    return QueryAnalysis(
        cleaned_query=cleaned,
        tokens=info_tokens,
        intent_tags=tags,
        complexity_score=complexity_score,
        prefer_agent=prefer_agent,
    )


def expand_query(analysis: QueryAnalysis, limit: int | None = None) -> list[str]:
    max_expansions = limit or settings.retrieval_expansion_k
    base = analysis.cleaned_query
    expansions: list[str] = [base]

    if analysis.tokens:
        compact = " ".join(analysis.tokens[:10])
        if compact and compact != base:
            expansions.append(compact)

    for token in analysis.tokens:
        if token in _TERM_EXPANSIONS:
            expansion = f"{base} {' '.join(_TERM_EXPANSIONS[token])}"
            expansions.append(expansion)
        if len(expansions) >= max_expansions:
            break

    # deterministic de-dup preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for text in expansions:
        key = text.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
        if len(deduped) >= max_expansions:
            break
    return deduped


def choose_pipeline(analysis: QueryAnalysis) -> str:
    return "agent" if analysis.prefer_agent else "rag"


def evaluate_context(
    query_terms: list[str],
    contexts: list[str],
    avg_rank_score: float | None = None,
) -> ContextEvaluation:
    if not contexts:
        return ContextEvaluation(
            sufficient=False,
            coverage_score=0.0,
            reason="No chunks retrieved",
        )

    unique_terms = list(dict.fromkeys(term for term in query_terms if len(term) > 2))
    if not unique_terms:
        unique_terms = _informative_tokens(_tokenize(" ".join(contexts[:1])))

    hit_count = 0
    context_blob = " ".join(contexts).lower()
    for term in unique_terms:
        if term in context_blob:
            hit_count += 1

    coverage = hit_count / max(len(unique_terms), 1)
    char_count = sum(len(item) for item in contexts)
    enough_chunks = len(contexts) >= settings.context_min_chunks
    enough_chars = char_count >= settings.context_min_chars
    enough_coverage = coverage >= settings.context_min_coverage

    if avg_rank_score is not None and avg_rank_score < 0.28:
        return ContextEvaluation(
            sufficient=False,
            coverage_score=coverage,
            reason="Low rerank confidence",
        )

    sufficient = enough_chunks and enough_chars and enough_coverage
    if sufficient:
        reason = "Context appears sufficient"
    else:
        reason = "Context is thin; escalation recommended"

    return ContextEvaluation(
        sufficient=sufficient,
        coverage_score=coverage,
        reason=reason,
    )
