from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.conversation import Conversation
from app.models.conversation_memory import ConversationMemory
from app.models.message import Message
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.conversation import ConversationListItem, ConversationRead, MessageRead
from app.services.smart_chat_service import SmartChatError, run_smart_chat

router = APIRouter(prefix="/chat", tags=["chat"])


def _build_default_title(message: str) -> str:
    cleaned = " ".join(message.split()).strip()
    if len(cleaned) <= 60:
        return cleaned or "New Conversation"
    return f"{cleaned[:57]}..."


def _resolve_conversation(
    db: Session,
    request: ChatRequest,
    current_user: User,
) -> Conversation:
    if request.conversation_id:
        conversation = db.get(Conversation, request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="You do not have access to this conversation")
        return conversation

    conversation = Conversation(
        title=_build_default_title(request.message),
        user_id=current_user.id,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversation = _resolve_conversation(db, request, current_user)

    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    conversation.updated_at = datetime.now(timezone.utc)
    db.commit()

    try:
        result = run_smart_chat(
            db=db,
            query=request.message,
            user_id=current_user.id,
            conversation_id=conversation.id,
        )
    except SmartChatError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=result.answer,
    )
    db.add(assistant_message)
    conversation.updated_at = datetime.now(timezone.utc)
    db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        answer=result.answer,
        context=result.context,
        pipeline=result.pipeline,
        route_reason=result.route_reason,
        query_expansions=result.query_expansions,
        context_coverage=result.context_coverage,
        context_sufficient=result.context_sufficient,
    )


@router.get("/conversations", response_model=list[ConversationListItem])
def list_conversations(
    limit: int = 30,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .limit(limit)
        .offset(offset)
    )
    conversations = db.execute(stmt).scalars().all()

    items: list[ConversationListItem] = []
    for convo in conversations:
        count_stmt = select(func.count(Message.id)).where(Message.conversation_id == convo.id)
        message_count = db.execute(count_stmt).scalar_one()

        last_stmt = (
            select(Message)
            .where(Message.conversation_id == convo.id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(1)
        )
        last_message = db.execute(last_stmt).scalars().first()
        preview = None
        created_at = None
        if last_message:
            text = " ".join(last_message.content.split())
            preview = text if len(text) <= 140 else f"{text[:137]}..."
            created_at = last_message.created_at

        items.append(
            ConversationListItem(
                id=convo.id,
                title=convo.title,
                updated_at=convo.updated_at,
                message_count=message_count,
                last_message_preview=preview,
                last_message_at=created_at,
            )
        )

    return items


@router.get("/conversations/{conversation_id}", response_model=ConversationRead)
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversation = db.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this conversation")

    message_stmt = (
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    rows = db.execute(message_stmt).scalars().all()

    memory = db.get(ConversationMemory, conversation.id)

    return ConversationRead(
        id=conversation.id,
        title=conversation.title,
        summary=memory.summary_text if memory else None,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[MessageRead.model_validate(item) for item in rows],
    )
