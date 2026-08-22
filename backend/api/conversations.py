import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import datetime

from backend.database.database import get_db, ConversationRepository
from backend.database.models import UserModel
from backend.auth.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime.datetime


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    message_count: int


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    messages: List[MessageResponse]


class CreateConversationRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Optional custom conversation title")


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: Optional[CreateConversationRequest] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Creates a new conversation session for the authenticated user."""
    title = request.title if request else None
    conv = ConversationRepository.get_or_create_conversation(
        db,
        title=title,
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=0,
    )


@router.get("", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = 50,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lists conversation sessions belonging to the authenticated user."""
    user_filter = None if current_user.role == "ADMIN" else current_user.id
    convs = ConversationRepository.list_conversations(
        db,
        user_id=user_filter,
        tenant_id=current_user.tenant_id,
        limit=limit,
    )
    return [
        ConversationSummary(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=len(c.messages),
        )
        for c in convs
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves a specific conversation with all historical messages and citations."""
    conv = ConversationRepository.get_conversation(db, conversation_id=conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found."
        )

    # Enforce isolation unless admin
    if current_user.role != "ADMIN" and conv.user_id and conv.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this conversation.",
        )

    messages = [
        MessageResponse(
            id=m.id,
            conversation_id=m.conversation_id,
            role=m.role,
            content=m.content,
            sources=m.sources or [],
            timestamp=m.timestamp,
        )
        for m in conv.messages
    ]

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=messages,
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deletes a conversation session."""
    user_filter = None if current_user.role == "ADMIN" else current_user.id
    success = ConversationRepository.delete_conversation(db, conversation_id=conversation_id, user_id=user_filter)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found or unauthorized.",
        )
    return {"message": f"Conversation '{conversation_id}' deleted."}
