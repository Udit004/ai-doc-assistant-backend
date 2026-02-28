import httpx

from app.core.config import settings


class LLMServiceError(RuntimeError):
    pass


def _build_prompt(query: str, context: list[str], history: list[str] | None = None) -> str:
    context_block = "\n\n".join(f"[{index + 1}] {item}" for index, item in enumerate(context))
    history_block = "\n".join(history[-8:] if history else [])

    return (
        "You are an AI documentation assistant. "
        "Answer using the provided context only when possible. "
        "If context is insufficient, explicitly say what is missing.\n\n"
        f"Conversation history:\n{history_block or 'No prior conversation'}\n\n"
        f"Retrieved context:\n{context_block or 'No retrieved context'}\n\n"
        f"User question:\n{query}\n\n"
        "Respond clearly and concisely. Include citations like [1], [2] when relevant."
    )


def _extract_text(response_data: dict) -> str:
    candidates = response_data.get("candidates") or []
    if not candidates:
        raise LLMServiceError("Gemini returned no candidates")

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    if not text_parts:
        raise LLMServiceError("Gemini response contained no text")
    return "\n".join(text_parts).strip()


def generate_answer(query: str, context: list[str], history: list[str] | None = None) -> str:
    if not settings.gemini_api_key:
        raise LLMServiceError("GEMINI_API_KEY is not set")

    prompt = _build_prompt(query=query, context=context, history=history)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": settings.gemini_temperature,
            "maxOutputTokens": settings.gemini_max_output_tokens,
        },
    }

    try:
        with httpx.Client(timeout=settings.gemini_timeout_seconds) as client:
            response = client.post(url, params={"key": settings.gemini_api_key}, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise LLMServiceError(
            f"Gemini request failed with status {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMServiceError(f"Gemini request failed: {exc}") from exc

    return _extract_text(response.json())

