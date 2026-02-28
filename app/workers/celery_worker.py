from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "ai_doc_assistant",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)


@celery_app.task(name="health.ping")
def ping() -> str:
    return "pong"

