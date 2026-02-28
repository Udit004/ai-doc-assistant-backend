from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from pypdf import PdfReader


class FileParsingError(ValueError):
    pass


class EmptyFileError(FileParsingError):
    pass


class UnsupportedFileTypeError(FileParsingError):
    pass


class FileTooLargeError(FileParsingError):
    pass


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".json",
    ".html",
    ".htm",
}


def _decode_text(raw_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise FileParsingError("Could not decode file text content")


def _parse_pdf(raw_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(raw_bytes))
    page_text = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(chunk for chunk in page_text if chunk).strip()


async def parse_upload(file: UploadFile, max_size_mb: int = 20) -> str:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise EmptyFileError("Uploaded file is empty")

    max_bytes = max_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise FileTooLargeError(f"File exceeds {max_size_mb} MB limit")

    filename = file.filename or "uploaded-file"
    extension = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()

    if extension == ".pdf" or content_type == "application/pdf":
        parsed = _parse_pdf(raw_bytes)
        if parsed:
            return parsed
        raise FileParsingError("PDF file does not contain extractable text")

    if extension in TEXT_EXTENSIONS or content_type.startswith("text/"):
        return _decode_text(raw_bytes).strip()

    raise UnsupportedFileTypeError(
        f"Unsupported file type: extension '{extension or 'unknown'}' and content-type '{content_type or 'unknown'}'"
    )
