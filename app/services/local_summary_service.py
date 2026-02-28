from __future__ import annotations

import re
from collections import Counter

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        if token not in STOP_WORDS
    ]


def _normalize_line(role: str, content: str) -> str:
    cleaned = " ".join(content.split())
    if len(cleaned) > 240:
        cleaned = f"{cleaned[:237]}..."
    return f"{role}: {cleaned}"


def summarize_messages_locally(
    messages: list[tuple[str, str]],
    previous_summary: str | None = None,
    max_lines: int = 10,
    max_chars: int = 1400,
) -> str:
    """
    Lightweight extractive summary that runs locally (no API call).
    """
    if not messages:
        return (previous_summary or "").strip()

    lines: list[str] = []
    if previous_summary:
        lines.append(f"summary: {' '.join(previous_summary.split())}")

    lines.extend(_normalize_line(role, content) for role, content in messages)
    if not lines:
        return ""

    freq = Counter(token for line in lines for token in _tokenize(line))
    scored: list[tuple[float, int, str]] = []

    for index, line in enumerate(lines):
        tokens = _tokenize(line)
        lexical = sum(freq[token] for token in tokens) / max(len(tokens), 1)
        role_boost = 0.25 if line.startswith("user:") else 0.15
        scored.append((lexical + role_boost, index, line))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = sorted(scored[:max_lines], key=lambda item: item[1])

    output: list[str] = []
    char_count = 0
    for _, _, line in selected:
        candidate = f"- {line}"
        next_count = char_count + len(candidate) + 1
        if output and next_count > max_chars:
            break
        output.append(candidate)
        char_count = next_count

    return "\n".join(output)[:max_chars].strip()
