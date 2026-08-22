import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.database import get_db, UserRepository, AuditLogRepository
from backend.database.models import UserModel, AuditLogModel
from backend.auth.dependencies import require_roles
from backend.rag.retriever import MedicalRetriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Administration & Users"])


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., min_length=5, max_length=128)
    password: str = Field(..., min_length=6, max_length=128)
    role: str = Field(default="CUSTOMER", description="ADMIN, FDE_ENGINEER, or CUSTOMER")
    tenant_id: str = Field(default="customer_001", description="Tenant ID (e.g. system, customer_001)")
    full_name: str = Field(..., min_length=2, max_length=128)


class UserItemResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    tenant_id: str
    full_name: str
    is_active: bool
    created_at: str


class AuditLogItemResponse(BaseModel):
    id: str
    user_id: Optional[str]
    username: Optional[str]
    role: Optional[str]
    tenant_id: Optional[str]
    action: str
    status: str
    details: Optional[str]
    timestamp: str


@router.get("/users", response_model=List[UserItemResponse])
async def list_users_endpoint(
    current_admin: UserModel = Depends(require_roles("ADMIN")),
    db: Session = Depends(get_db),
):
    """Lists all users in the system (ADMIN only)."""
    users = UserRepository.list_users(db)
    return [
        UserItemResponse(
            id=u.id,
            username=u.username,
            email=u.email,
            role=u.role,
            tenant_id=u.tenant_id,
            full_name=u.full_name,
            is_active=u.is_active,
            created_at=u.created_at.isoformat() if u.created_at else "",
        )
        for u in users
    ]


@router.post("/users", response_model=UserItemResponse, status_code=status.HTTP_201_CREATED)
async def create_user_endpoint(
    request: UserCreateRequest,
    current_admin: UserModel = Depends(require_roles("ADMIN")),
    db: Session = Depends(get_db),
):
    """Creates a new user with specified role and tenant (ADMIN only)."""
    valid_roles = ["ADMIN", "FDE_ENGINEER", "CUSTOMER"]
    role_upper = request.role.strip().upper()
    if role_upper not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{request.role}'. Allowed roles: {valid_roles}",
        )

    # Check for existing username or email
    if UserRepository.get_by_username(db, username=request.username.strip().lower()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken.",
        )

    new_user = UserRepository.create_user(
        db,
        username=request.username,
        email=request.email,
        password=request.password,
        role=role_upper,
        tenant_id=request.tenant_id,
        full_name=request.full_name,
    )

    # Log user creation audit
    AuditLogRepository.log_event(
        db,
        user_id=current_admin.id,
        username=current_admin.username,
        role=current_admin.role,
        tenant_id=current_admin.tenant_id,
        action="USER_CREATED",
        status="SUCCESS",
        details=f"Created user '{new_user.username}' with role '{new_user.role}' and tenant '{new_user.tenant_id}'",
    )

    return UserItemResponse(
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        role=new_user.role,
        tenant_id=new_user.tenant_id,
        full_name=new_user.full_name,
        is_active=new_user.is_active,
        created_at=new_user.created_at.isoformat() if new_user.created_at else "",
    )


@router.delete("/users/{user_id}")
async def delete_user_endpoint(
    user_id: str,
    current_admin: UserModel = Depends(require_roles("ADMIN")),
    db: Session = Depends(get_db),
):
    """Deletes user by ID (ADMIN only)."""
    if user_id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own administrative account.",
        )

    target_user = UserRepository.get_by_id(db, user_id=user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    username_cached = target_user.username
    UserRepository.delete_user(db, user_id=user_id)

    AuditLogRepository.log_event(
        db,
        user_id=current_admin.id,
        username=current_admin.username,
        role=current_admin.role,
        tenant_id=current_admin.tenant_id,
        action="USER_DELETED",
        status="SUCCESS",
        details=f"Deleted user account '{username_cached}' (ID: {user_id})",
    )

    return {"message": f"User '{username_cached}' successfully deleted."}


@router.get("/audit-logs", response_model=List[AuditLogItemResponse])
async def list_audit_logs_endpoint(
    current_admin: UserModel = Depends(require_roles("ADMIN")),
    db: Session = Depends(get_db),
):
    """Fetches security audit events (ADMIN only)."""
    logs = AuditLogRepository.list_logs(db, limit=100)
    return [
        AuditLogItemResponse(
            id=l.id,
            user_id=l.user_id,
            username=l.username,
            role=l.role,
            tenant_id=l.tenant_id,
            action=l.action,
            status=l.status,
            details=l.details,
            timestamp=l.timestamp.isoformat() if l.timestamp else "",
        )
        for l in logs
    ]
