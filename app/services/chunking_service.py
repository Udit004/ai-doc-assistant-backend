import re


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _select_breakpoint(text: str, start: int, target_end: int) -> int:
    floor = start + int((target_end - start) * 0.6)
    markers = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

    best = -1
    best_adjustment = 0
    for marker in markers:
        marker_index = text.rfind(marker, floor, target_end)
        if marker_index > best:
            best = marker_index
            best_adjustment = len(marker.rstrip())

    if best == -1:
        return target_end

    return best + best_adjustment


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            end = _select_breakpoint(normalized, start, end)
            if end <= start:
                end = min(start + chunk_size, text_length)

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = end - overlap

    return chunks
