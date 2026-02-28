from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import models  # noqa: F401
from app.api import agent, auth, chat, health, upload
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.base import Base
from app.db.postgres import (
    ensure_chunk_vector_schema,
    ensure_user_ownership_schema,
    initialize_pgvector_extension,
)
from app.db.session import engine

configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_pgvector_extension()
    Base.metadata.create_all(bind=engine)
    ensure_chunk_vector_schema()
    ensure_user_ownership_schema()
    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_v1_prefix)
app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(upload.router, prefix=settings.api_v1_prefix)
app.include_router(chat.router, prefix=settings.api_v1_prefix)
app.include_router(agent.router, prefix=settings.api_v1_prefix)


@app.get("/", tags=["root"])
def root():
    return {"message": f"{settings.app_name} is running"}
