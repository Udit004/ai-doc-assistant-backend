from __future__ import annotations

import math
import re

from app.services.embedding_service import EmbeddingServiceError, embed_query, embed_texts

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


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        if token not in _STOP_WORDS
    ]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_context_with_scores(query: str, contexts: list[str], k: int = 5) -> list[dict]:
    if not contexts:
        return []

    terms = _tokenize(query)
    query_vector: list[float] | None = None
    context_vectors: list[list[float]] = []
    try:
        query_vector = embed_query(query)
        context_vectors = embed_texts(contexts)
    except EmbeddingServiceError:
        query_vector = None
        context_vectors = []

    scored: list[dict] = []
    for index, context in enumerate(contexts):
        text_low = context.lower()
        term_hits = sum(term in text_low for term in terms)
        lexical_score = term_hits / max(len(terms), 1)

        semantic_score = 0.0
        if query_vector and index < len(context_vectors):
            semantic_score = max(0.0, _cosine_similarity(query_vector, context_vectors[index]))

        target_length = 400
        length_score = max(0.0, 1.0 - (abs(len(context) - target_length) / target_length))

        final_score = (semantic_score * 0.55) + (lexical_score * 0.35) + (length_score * 0.10)
        scored.append(
            {
                "content": context,
                "score": round(final_score, 4),
                "lexical_score": round(lexical_score, 4),
                "semantic_score": round(semantic_score, 4),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:k]


def rerank_context(query: str, contexts: list[str], k: int = 3) -> list[str]:
    return [item["content"] for item in rerank_context_with_scores(query, contexts, k=k)]
