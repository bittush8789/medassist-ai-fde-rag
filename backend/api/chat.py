import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.database import get_db, ConversationRepository, AuditLogRepository
from backend.database.models import UserModel
from backend.auth.dependencies import get_current_user
from backend.rag.chain import MedicalRAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])

# Singleton pipeline instance
_rag_pipeline: Optional[MedicalRAGPipeline] = None


def get_rag_pipeline() -> MedicalRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = MedicalRAGPipeline()
    return _rag_pipeline


class SourceCitation(BaseModel):
    document: str
    page: int
    section: Optional[str] = "General"
    chunk_id: Optional[str] = None
    relevance_score: Optional[float] = None
    snippet: Optional[str] = None


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(default=None, description="Conversation session ID")
    message: str = Field(..., min_length=1, max_length=2000, description="User's medical query")


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: List[SourceCitation]
    latency_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    rerank_ms: Optional[float] = None
    llm_ms: Optional[float] = None
    model: Optional[str] = None
    retrieval_count: Optional[int] = None
    candidates_count: Optional[int] = None
    rbac_role: Optional[str] = None
    rbac_tenant: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
    pipeline: MedicalRAGPipeline = Depends(get_rag_pipeline),
):
    """
    Processes user medical question through two-stage RAG pipeline
    with strictly enforced RBAC and Tenant-level metadata filtering.
    """
    user_query = request.message.strip()
    if not user_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty."
        )

    # 1. Retrieve or create tenant-isolated conversation session
    conv = ConversationRepository.get_or_create_conversation(
        db,
        conversation_id=request.conversation_id,
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    conversation_id = conv.id

    # 2. Fetch recent conversation history
    existing_messages = ConversationRepository.get_conversation_history(db, conversation_id=conversation_id, limit=8)
    history = [{"role": m.role, "content": m.content} for m in existing_messages]

    # 3. Save user's message
    ConversationRepository.add_message(
        db,
        conversation_id=conversation_id,
        role="user",
        content=user_query,
    )

    # 4. Run RAG Pipeline with user's Role and Tenant ID
    try:
        rag_output = pipeline.answer_query(
            query=user_query,
            conversation_id=conversation_id,
            history=history,
            user_role=current_user.role,
            user_tenant_id=current_user.tenant_id,
        )
    except Exception as e:
        logger.error(f"RAG processing failed: {str(e)}", exc_info=True)
        AuditLogRepository.log_event(
            db,
            user_id=current_user.id,
            username=current_user.username,
            role=current_user.role,
            tenant_id=current_user.tenant_id,
            action="RAG_QUERY",
            status="ERROR",
            details=f"Query error: {str(e)[:120]}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the medical query."
        )

    answer_text = rag_output.get("answer", "")
    sources_data = rag_output.get("sources", [])
    latency_ms = rag_output.get("latency_ms")

    # 5. Save assistant's answer with citations
    ConversationRepository.add_message(
        db,
        conversation_id=conversation_id,
        role="assistant",
        content=answer_text,
        sources=sources_data,
    )

    # 6. Audit successful RAG query
    AuditLogRepository.log_event(
        db,
        user_id=current_user.id,
        username=current_user.username,
        role=current_user.role,
        tenant_id=current_user.tenant_id,
        action="RAG_QUERY",
        status="SUCCESS",
        details=f"Retrieved {len(sources_data)} sources across tenant '{current_user.tenant_id}'",
    )

    formatted_sources = [
        SourceCitation(
            document=s.get("document", "Unknown"),
            page=s.get("page", 1),
            section=s.get("section", "General"),
            chunk_id=s.get("chunk_id"),
            relevance_score=s.get("relevance_score"),
            snippet=s.get("snippet"),
        )
        for s in sources_data
    ]

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer_text,
        sources=formatted_sources,
        latency_ms=latency_ms,
        retrieval_ms=rag_output.get("retrieval_ms"),
        rerank_ms=rag_output.get("rerank_ms"),
        llm_ms=rag_output.get("llm_ms"),
        model=rag_output.get("model"),
        retrieval_count=rag_output.get("retrieval_count"),
        candidates_count=rag_output.get("candidates_count"),
        rbac_role=current_user.role,
        rbac_tenant=current_user.tenant_id,
    )
