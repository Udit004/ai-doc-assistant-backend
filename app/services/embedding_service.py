from functools import lru_cache

from app.core.config import settings


class EmbeddingServiceError(RuntimeError):
    pass


@lru_cache
def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingServiceError(
            "sentence-transformers is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return SentenceTransformer(
        settings.embedding_model_name,
        device=settings.embedding_device,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = _load_sentence_transformer()
    try:
        vectors = model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=settings.embedding_normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingServiceError("Failed to generate MiniLM embeddings") from exc

    return [vector.tolist() for vector in vectors]


def embed_query(query: str) -> list[float]:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise EmbeddingServiceError("Query text is empty")

    model = _load_sentence_transformer()
    try:
        vector = model.encode(
            [cleaned_query],
            batch_size=1,
            normalize_embeddings=settings.embedding_normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
    except Exception as exc:
        raise EmbeddingServiceError("Failed to generate query embedding") from exc

    return vector.tolist()

