from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Doc Assistant API"
    app_env: str = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./app.db"
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )
    max_upload_size_mb: int = 20
    chunk_size: int = 1000
    chunk_overlap: int = 150
    db_pool_pre_ping: bool = True
    db_pool_recycle_seconds: int = 300
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout_seconds: int = 30
    db_connect_timeout_seconds: int = 10
    db_keepalives_idle_seconds: int = 30
    db_keepalives_interval_seconds: int = 10
    db_keepalives_count: int = 5
    # Gemini text-embedding-004 produces 768-dimensional vectors
    embedding_model_name: str = "text-embedding-004"
    embedding_dimension: int = 768
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 20
    retrieval_rerank_k: int = 5
    retrieval_expansion_k: int = 4
    context_min_coverage: float = 0.35
    context_min_chunks: int = 2
    context_min_chars: int = 350
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_max_output_tokens: int = 1024
    gemini_temperature: float = 0.2
    gemini_timeout_seconds: int = 60

    secret_key: str = "change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Goal agent
    agent_max_iterations: int = 3
    agent_max_subtasks: int = 4
    agent_eval_threshold: float = 7.0

    # Conversation memory
    memory_recent_limit: int = 10
    memory_summary_trigger_messages: int = 18
    memory_summary_max_chars: int = 1400

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
