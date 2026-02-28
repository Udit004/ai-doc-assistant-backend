from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.conversation_memory import ConversationMemory
from app.models.message import Message
from app.services.local_summary_service import summarize_messages_locally


def _load_messages_asc(db: Session, conversation_id: int) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    return db.execute(stmt).scalars().all()


def _get_or_create_memory(db: Session, conversation_id: int) -> ConversationMemory:
    memory = db.get(ConversationMemory, conversation_id)
    if memory:
        return memory
    memory = ConversationMemory(conversation_id=conversation_id)
    db.add(memory)
    db.flush()
    return memory


def get_recent_messages(db: Session, conversation_id: int, limit: int = 10) -> list[str]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
    )
    rows = db.execute(stmt).scalars().all()
    rows.reverse()
    return [f"{item.role}: {item.content}" for item in rows]


def summarize_conversation_history(
    db: Session,
    conversation_id: int,
    recent_limit: int | None = None,
) -> str | None:
    """
    Persist an updated local summary when conversation history grows large.
    """
    recent_keep = recent_limit or settings.memory_recent_limit
    messages = _load_messages_asc(db, conversation_id)
    if len(messages) <= settings.memory_summary_trigger_messages:
        existing = db.get(ConversationMemory, conversation_id)
        return existing.summary_text if existing else None

    split_index = max(0, len(messages) - recent_keep)
    older_messages = messages[:split_index]
    if not older_messages:
        return None

    memory = _get_or_create_memory(db, conversation_id)
    older_upto_id = older_messages[-1].id
    if older_upto_id <= memory.summarized_upto_message_id:
        return memory.summary_text

    incremental = [
        (msg.role, msg.content)
        for msg in older_messages
        if msg.id > memory.summarized_upto_message_id
    ]
    if not incremental:
        return memory.summary_text

    memory.summary_text = summarize_messages_locally(
        incremental,
        previous_summary=memory.summary_text,
        max_chars=settings.memory_summary_max_chars,
    )
    memory.summarized_upto_message_id = older_upto_id
    db.add(memory)
    db.flush()
    return memory.summary_text


def build_contextual_history(
    db: Session,
    conversation_id: int,
    recent_limit: int | None = None,
) -> list[str]:
    """
    Return compact history for prompting: summary + latest turns.
    """
    recent_keep = recent_limit or settings.memory_recent_limit
    summary = summarize_conversation_history(db, conversation_id, recent_limit=recent_keep)
    recent_messages = get_recent_messages(db, conversation_id, limit=recent_keep)

    history: list[str] = []
    if summary:
        history.append(f"conversation_summary: {summary}")
    history.extend(recent_messages)
    return history
