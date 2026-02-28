import pytest

from app.services.chunking_service import chunk_text


def test_chunk_text_creates_deterministic_chunks_with_overlap():
    source_text = ("FastAPI makes backend development productive and reliable. " * 120).strip()

    chunks = chunk_text(source_text, chunk_size=320, overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 320 for chunk in chunks)

    overlap_seed = chunks[0][-20:].strip()
    assert overlap_seed
    assert overlap_seed in chunks[1][:120]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("sample text", chunk_size=300, overlap=300)

