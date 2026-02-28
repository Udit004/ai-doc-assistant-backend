from __future__ import annotations

from collections import defaultdict

from sqlalchemy.orm import Session

from app.services.retrieval_service import retrieve_context


def _rrf_merge(results_by_query: list[list[str]], k: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    rrf_constant = 60

    for query_index, rows in enumerate(results_by_query):
        query_weight = 1.0 if query_index == 0 else 0.9
        for rank, chunk in enumerate(rows, start=1):
            scores[chunk] += query_weight / (rrf_constant + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [chunk for chunk, _ in ranked[:k]]


def retrieve_expanded_context(
    db: Session,
    user_id: int,
    expanded_queries: list[str],
    candidate_k: int = 20,
) -> list[str]:
    """
    Retrieve candidate chunks using multiple query variants and fuse rankings.
    """
    if not expanded_queries:
        return []

    all_ranked: list[list[str]] = []
    for query in expanded_queries:
        rows = retrieve_context(db, query=query, user_id=user_id, k=candidate_k)
        if rows:
            all_ranked.append(rows)

    if not all_ranked:
        return []
    if len(all_ranked) == 1:
        return all_ranked[0][:candidate_k]

    return _rrf_merge(all_ranked, k=candidate_k)
