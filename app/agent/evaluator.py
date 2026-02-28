"""
Evaluator — asks Gemini to score its own combined answer (LLM-as-judge).

The evaluator builds a structured scoring prompt that shows Gemini the
original goal and all per-step answers, then parses a JSON confidence score.
If the combined answer is below the configured quality threshold, the
evaluator returns a list of gaps that the Planner uses to generate new
sub-tasks for the next agent iteration.
"""

from __future__ import annotations

import json
import logging

import httpx

from app.agent.state import EvalResult, StepResult
from app.core.config import settings

logger = logging.getLogger(__name__)

_EVAL_PROMPT = """\
You are a quality-assurance agent for a document Q&A system.

Original user goal:
{goal}

The agent produced the following sub-answers by searching the document knowledge base:

{sub_answers}

Evaluate the COMBINED quality of these answers with respect to the original goal.

Scoring criteria:
- completeness (1–5): Does the combined answer fully address the goal?
- accuracy     (1–5): Is the information factually consistent with the document context?
- overall      (0–10): Combined quality score.

If completeness < 4 or accuracy < 4, list specific information gaps that remain unanswered.
These gaps will be used to search for additional context, so phrase them as concrete questions.

Respond ONLY with valid JSON in this exact format:
{{
  "completeness": 3,
  "accuracy": 4,
  "score": 6.5,
  "sufficient": false,
  "gaps": ["What is the error-handling strategy?", "How are retries implemented?"]
}}

If the answer is sufficient, set "sufficient": true and "gaps": [].
"""


def _call_gemini_raw(prompt: str) -> str:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512,
        },
    }
    with httpx.Client(timeout=settings.gemini_timeout_seconds) as client:
        resp = client.post(url, params={"key": settings.gemini_api_key}, json=payload)
        resp.raise_for_status()

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()


def _format_sub_answers(results: list[StepResult]) -> str:
    lines: list[str] = []
    for idx, r in enumerate(results, 1):
        lines.append(f"--- Sub-answer {idx} ---")
        lines.append(f"Question: {r.query}")
        lines.append(f"Answer: {r.answer}")
        lines.append("")
    return "\n".join(lines)


def _parse_eval(raw: str, threshold: float) -> EvalResult:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
        score = float(data.get("score", 0))
        completeness = int(data.get("completeness", 0))
        accuracy = int(data.get("accuracy", 0))
        gaps: list[str] = data.get("gaps") or []
        sufficient = bool(data.get("sufficient", score >= threshold))
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Evaluator: JSON parse failed, assuming insufficient")
        score, completeness, accuracy, gaps, sufficient = 0.0, 0, 0, [], False

    # Override sufficient flag based on threshold regardless of model's claim
    if score < threshold:
        sufficient = False

    return EvalResult(
        score=score,
        completeness=completeness,
        accuracy=accuracy,
        gaps=gaps,
        sufficient=sufficient,
    )


class Evaluator:
    """Scores the combined sub-answers against the original goal."""

    def __init__(self, threshold: float | None = None) -> None:
        self._threshold = threshold if threshold is not None else settings.agent_eval_threshold

    def evaluate(self, goal: str, results: list[StepResult]) -> EvalResult:
        """
        Ask Gemini to rate how well the combined sub-answers address *goal*.

        Parameters
        ----------
        goal : str
            The user's original full goal.
        results : list[StepResult]
            All step results from the current (and previous) iterations.

        Returns
        -------
        EvalResult
            Contains score, completeness/accuracy ratings, gap list, and a
            ``sufficient`` flag signalling whether the loop should continue.
        """
        logger.info("Evaluator: evaluating %d sub-answers", len(results))

        # Filter out failed steps — don't let errors mislead the evaluator
        good_results = [r for r in results if not r.failed]
        if not good_results:
            logger.warning("Evaluator: no successful results, returning insufficient")
            return EvalResult(
                score=0.0,
                completeness=0,
                accuracy=0,
                gaps=[goal],
                sufficient=False,
            )

        sub_answers_text = _format_sub_answers(good_results)
        prompt = _EVAL_PROMPT.format(goal=goal, sub_answers=sub_answers_text)

        try:
            raw = _call_gemini_raw(prompt)
            result = _parse_eval(raw, self._threshold)
        except Exception as exc:
            logger.warning("Evaluator: Gemini call failed (%s), assuming sufficient to avoid loop", exc)
            # Fail-safe: don't loop forever on API errors
            result = EvalResult(
                score=self._threshold,
                completeness=3,
                accuracy=3,
                gaps=[],
                sufficient=True,
            )

        logger.info(
            "Evaluator: score=%.1f completeness=%d accuracy=%d sufficient=%s gaps=%d",
            result.score,
            result.completeness,
            result.accuracy,
            result.sufficient,
            len(result.gaps),
        )
        return result
