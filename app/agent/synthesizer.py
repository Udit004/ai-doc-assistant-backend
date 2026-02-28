"""
Synthesizer — merges all step results into one coherent final answer.

After the agent loop concludes (either because the evaluator is satisfied or
max iterations are reached), the Synthesizer asks Gemini to produce a single
unified response with numbered citations, using all accumulated sub-answers
and retrieved chunks as source material.
"""

from __future__ import annotations

import logging

import httpx

from app.agent.state import StepResult
from app.core.config import settings
from app.services.llm_service import LLMServiceError

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = """\
You are a document assistant synthesizing a final answer for a user.

Original user goal:
{goal}

The following sub-questions were researched and answered using the document knowledge base:

{sub_answers}

Retrieved document excerpts (numbered for citation):
{context_blocks}

Instructions:
1. Synthesize a single, well-structured answer that fully addresses the original goal.
2. Use information from the sub-answers and document excerpts above — do not invent facts.
3. Add inline citations like [1], [2], … referencing the numbered document excerpts.
4. If certain aspects could not be found in the documents, explicitly state what is missing.
5. Format the answer clearly: use short paragraphs or bullet points where appropriate.

Final answer:
"""


def _call_gemini_raw(prompt: str) -> str:
    if not settings.gemini_api_key:
        raise LLMServiceError("GEMINI_API_KEY is not set")

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
    with httpx.Client(timeout=settings.gemini_timeout_seconds) as client:
        resp = client.post(url, params={"key": settings.gemini_api_key}, json=payload)
        resp.raise_for_status()

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMServiceError("Gemini returned no candidates")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()


def _format_sub_answers(results: list[StepResult]) -> str:
    lines: list[str] = []
    for idx, r in enumerate(results, 1):
        lines.append(f"[Sub-question {idx}]: {r.query}")
        lines.append(f"[Sub-answer   {idx}]: {r.answer}")
        lines.append("")
    return "\n".join(lines)


def _format_context_blocks(results: list[StepResult]) -> tuple[str, list[str]]:
    """
    De-duplicate and number all retrieved chunks across all step results.

    Returns
    -------
    (formatted_string, deduplicated_list)
    """
    seen: set[str] = set()
    unique_chunks: list[str] = []

    for r in results:
        for chunk in r.context:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)

    lines = [f"[{i + 1}] {chunk}" for i, chunk in enumerate(unique_chunks)]
    return "\n\n".join(lines), unique_chunks


class Synthesizer:
    """Merges all agent step results into a final answer via Gemini."""

    def synthesize(
        self,
        goal: str,
        results: list[StepResult],
    ) -> tuple[str, list[str]]:
        """
        Produce one coherent answer from all completed step results.

        Parameters
        ----------
        goal : str
            Original user goal.
        results : list[StepResult]
            All successful step results (failed ones are filtered out).

        Returns
        -------
        (final_answer, all_context_chunks)
            ``final_answer`` is the LLM-generated synthesis.
            ``all_context_chunks`` is the deduplicated list used for citations.
        """
        good = [r for r in results if not r.failed]
        if not good:
            logger.warning("Synthesizer: no successful results to synthesize")
            return "I could not find sufficient information in your documents to answer this goal.", []

        logger.info("Synthesizer: synthesizing from %d step results", len(good))

        sub_answers_text = _format_sub_answers(good)
        context_text, all_chunks = _format_context_blocks(good)

        prompt = _SYNTHESIS_PROMPT.format(
            goal=goal,
            sub_answers=sub_answers_text,
            context_blocks=context_text or "No document excerpts retrieved.",
        )

        try:
            answer = _call_gemini_raw(prompt)
        except LLMServiceError as exc:
            logger.warning("Synthesizer: Gemini call failed (%s), falling back to concatenation", exc)
            # Graceful degradation: join sub-answers directly
            answer = "\n\n".join(
                f"**{r.query}**\n{r.answer}" for r in good
            )

        logger.info("Synthesizer: final answer length=%d chars", len(answer))
        return answer, all_chunks
