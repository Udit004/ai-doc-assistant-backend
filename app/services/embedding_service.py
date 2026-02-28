import httpx

from app.core.config import settings

# Gemini Embedding REST endpoint
_GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:embedContent"
)
_GEMINI_BATCH_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:batchEmbedContents"
)
_TIMEOUT = 60.0


class EmbeddingServiceError(RuntimeError):
    pass


def _api_key() -> str:
    key = settings.gemini_api_key
    if not key:
        raise EmbeddingServiceError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )
    return key


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Gemini batchEmbedContents."""
    if not texts:
        return []

    model = settings.embedding_model_name
    url = _GEMINI_BATCH_URL.format(model=model)
    payload = {
        "requests": [
            {
                "model": f"models/{model}",
                "content": {"parts": [{"text": t}]},
            }
            for t in texts
        ]
    }

    try:
        response = httpx.post(
            url,
            json=payload,
            headers={"x-goog-api-key": _api_key()},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise EmbeddingServiceError(
            f"Gemini embedding API error {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise EmbeddingServiceError(f"Failed to call Gemini embedding API: {exc}") from exc

    data = response.json()
    embeddings = data.get("embeddings", [])
    if len(embeddings) != len(texts):
        raise EmbeddingServiceError(
            f"Gemini returned {len(embeddings)} embeddings for {len(texts)} texts"
        )

    return [item["values"] for item in embeddings]


def embed_query(query: str) -> list[float]:
    """Embed a single query string using Gemini embedContent."""
    cleaned = query.strip()
    if not cleaned:
        raise EmbeddingServiceError("Query text is empty")

    model = settings.embedding_model_name
    url = _GEMINI_EMBED_URL.format(model=model)
    payload = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": cleaned}]},
    }

    try:
        response = httpx.post(
            url,
            json=payload,
            headers={"x-goog-api-key": _api_key()},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise EmbeddingServiceError(
            f"Gemini embedding API error {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise EmbeddingServiceError(f"Failed to call Gemini embedding API: {exc}") from exc

    data = response.json()
    values = data.get("embedding", {}).get("values")
    if not values:
        raise EmbeddingServiceError("Gemini returned empty embedding values")

    return values

