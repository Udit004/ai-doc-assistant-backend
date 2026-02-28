from collections import defaultdict

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embedding_service import EmbeddingServiceError, embed_query


def _dense_retrieval(db: Session, query: str, limit: int, user_id: int) -> list[str]:
    query_vector = embed_query(query)
    stmt = (
        select(Chunk.content)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.user_id == user_id)
        .where(Chunk.embedding_vector.is_not(None))
        .order_by(Chunk.embedding_vector.cosine_distance(query_vector))
        .limit(limit)
    )
    return db.execute(stmt).scalars().all()


def _keyword_retrieval(db: Session, query: str, limit: int, user_id: int) -> list[str]:
    ts_query = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(func.to_tsvector("english", Chunk.content), ts_query)
    stmt = (
        select(Chunk.content)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.user_id == user_id)
        .where(func.to_tsvector("english", Chunk.content).op("@@")(ts_query))
        .order_by(desc(rank))
        .limit(limit)
    )
    return db.execute(stmt).scalars().all()


def _keyword_fallback(db: Session, query: str, limit: int, user_id: int) -> list[str]:
    terms = [token.lower() for token in query.split() if token.strip()]
    if not terms:
        return []

    all_chunks = db.execute(
        select(Chunk.content)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.user_id == user_id)
    ).scalars().all()
    scored: list[tuple[int, str]] = []
    for content in all_chunks:
        content_lower = content.lower()
        score = sum(term in content_lower for term in terms)
        if score > 0:
            scored.append((score, content))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [content for _, content in scored[:limit]]


def _rrf_merge(dense: list[str], keyword: list[str], k: int) -> list[str]:
    rrf_constant = 60
    scores: dict[str, float] = defaultdict(float)

    for rank_index, content in enumerate(dense, start=1):
        scores[content] += 1.0 / (rrf_constant + rank_index)
    for rank_index, content in enumerate(keyword, start=1):
        scores[content] += 1.0 / (rrf_constant + rank_index)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [content for content, _ in ranked[:k]]


def _recent_user_chunks(db: Session, user_id: int, limit: int) -> list[str]:
    stmt = (
        select(Chunk.content)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.user_id == user_id)
        .order_by(Chunk.created_at.desc())
        .limit(limit)
    )
    rows = db.execute(stmt).scalars().all()
    rows.reverse()
    return rows


def retrieve_context(db: Session, query: str, user_id: int, k: int = 5) -> list[str]:
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    dense_results: list[str] = []
    try:
        dense_results = _dense_retrieval(db, cleaned_query, limit=k * 3, user_id=user_id)
    except EmbeddingServiceError:
        dense_results = []

    try:
        keyword_results = _keyword_retrieval(db, cleaned_query, limit=k * 3, user_id=user_id)
    except Exception:
        keyword_results = _keyword_fallback(db, cleaned_query, limit=k * 3, user_id=user_id)

    if dense_results and keyword_results:
        return _rrf_merge(dense_results, keyword_results, k=k)
    if dense_results:
        return dense_results[:k]
    if keyword_results:
        return keyword_results[:k]
    return _recent_user_chunks(db, user_id=user_id, limit=k)
