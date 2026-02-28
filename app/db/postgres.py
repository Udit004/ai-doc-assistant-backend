from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine


def _is_postgres() -> bool:
    return settings.database_url.startswith("postgresql")


def initialize_pgvector_extension() -> None:
    if not _is_postgres():
        return

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def ensure_chunk_vector_schema() -> None:
    if not _is_postgres():
        return

    dimension = int(settings.embedding_dimension)
    if dimension <= 0:
        raise ValueError("EMBEDDING_DIMENSION must be a positive integer")

    with engine.begin() as connection:
        connection.execute(
            text(
                f"ALTER TABLE IF EXISTS chunks "
                f"ADD COLUMN IF NOT EXISTS embedding_vector vector({dimension})"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE IF EXISTS chunks "
                "ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(255)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_vector "
                "ON chunks USING ivfflat (embedding_vector vector_cosine_ops) "
                "WITH (lists = 100)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_content_fts "
                "ON chunks USING GIN (to_tsvector('english', content))"
            )
        )


def ensure_user_ownership_schema() -> None:
    if not _is_postgres():
        return

    with engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS user_id INTEGER")
        )
        connection.execute(
            text("ALTER TABLE IF EXISTS conversations ADD COLUMN IF NOT EXISTS user_id INTEGER")
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id)")
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations (user_id)")
        )
        connection.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'fk_documents_user_id_users'
                    ) THEN
                        ALTER TABLE documents
                        ADD CONSTRAINT fk_documents_user_id_users
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
                    END IF;
                END;
                $$;
                """
            )
        )
        connection.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'fk_conversations_user_id_users'
                    ) THEN
                        ALTER TABLE conversations
                        ADD CONSTRAINT fk_conversations_user_id_users
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
                    END IF;
                END;
                $$;
                """
            )
        )
