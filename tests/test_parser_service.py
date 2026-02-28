import asyncio
from io import BytesIO

import pytest
from fastapi import UploadFile

from app.services.parser_service import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    parse_upload,
)


def test_parse_upload_accepts_text_files():
    upload = UploadFile(filename="notes.md", file=BytesIO(b"# Heading\nSome content"))

    parsed = asyncio.run(parse_upload(upload))

    assert "Heading" in parsed
    assert "Some content" in parsed


def test_parse_upload_rejects_empty_files():
    upload = UploadFile(filename="empty.txt", file=BytesIO(b""))

    with pytest.raises(EmptyFileError):
        asyncio.run(parse_upload(upload))


def test_parse_upload_rejects_unknown_extensions():
    upload = UploadFile(filename="archive.bin", file=BytesIO(b"\x00\x01\x02\x03"))

    with pytest.raises(UnsupportedFileTypeError):
        asyncio.run(parse_upload(upload))


def test_parse_upload_rejects_large_files():
    upload = UploadFile(filename="large.txt", file=BytesIO(b"a" * (2 * 1024 * 1024)))

    with pytest.raises(FileTooLargeError):
        asyncio.run(parse_upload(upload, max_size_mb=1))

