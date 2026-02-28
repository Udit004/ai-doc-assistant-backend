from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.user import User
from app.schemas.document import DocumentRead
from app.services.chunking_service import chunk_text
from app.services.embedding_service import EmbeddingServiceError, embed_texts
from app.services.parser_service import (
    EmptyFileError,
    FileParsingError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    parse_upload,
)

router = APIRouter(prefix="/upload", tags=["upload"])


def _mark_document_failed(db: Session, document: Document) -> None:
    try:
        db.rollback()
        document.status = "failed"
        db.add(document)
        db.commit()
    except SQLAlchemyError:
        db.rollback()


@router.post("", response_model=DocumentRead)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document_title = title or file.filename or "Untitled document"
    document = Document(
        user_id=current_user.id,
        title=document_title,
        filename=file.filename or "uploaded-file",
        status="processing",
    )

    try:
        db.add(document)
        db.commit()
        db.refresh(document)

        parsed_text = await parse_upload(file, max_size_mb=settings.max_upload_size_mb)
        chunks = chunk_text(
            parsed_text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        embeddings = embed_texts(chunks)

        if not chunks:
            _mark_document_failed(db, document)
            raise HTTPException(status_code=400, detail="No text could be extracted from uploaded file")

        chunk_rows = []
        for index, chunk_value in enumerate(chunks):
            chunk_rows.append(
                Chunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk_value,
                    embedding_vector=embeddings[index],
                    embedding_model=settings.embedding_model_name,
                )
            )
        document.status = "embedded"
        db.add_all(chunk_rows)
        db.add(document)
        db.commit()
        db.refresh(document)

        return DocumentRead(
            id=document.id,
            user_id=document.user_id,
            title=document.title,
            filename=document.filename,
            status=document.status,
            created_at=document.created_at,
            updated_at=document.updated_at,
            chunk_count=len(chunks),
        )
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Database connection failed. Please retry.",
        ) from exc
    except (EmptyFileError, UnsupportedFileTypeError, FileTooLargeError) as exc:
        _mark_document_failed(db, document)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileParsingError as exc:
        _mark_document_failed(db, document)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingServiceError as exc:
        _mark_document_failed(db, document)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        _mark_document_failed(db, document)
        raise HTTPException(status_code=500, detail="Failed to process uploaded file") from exc
