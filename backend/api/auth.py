import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.database import get_db, UserRepository, AuditLogRepository
from backend.database.models import UserModel
from backend.auth.security import verify_password, create_access_token
from backend.auth.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64, description="Username")
    password: str = Field(..., min_length=1, max_length=128, description="Password")


class UserProfileResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    tenant_id: str
    full_name: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfileResponse


@router.post("/login", response_model=LoginResponse)
async def login_endpoint(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    """
    Authenticates user credentials, logs the audit event,
    and returns a signed JWT with RBAC and tenant claims.
    """
    username = request.username.strip().lower()
    user = UserRepository.get_by_username(db, username=username)

    if not user or not user.is_active or not verify_password(request.password, user.hashed_password):
        # Audit failed login attempt
        AuditLogRepository.log_event(
            db,
            user_id=user.id if user else None,
            username=username,
            role=user.role if user else None,
            tenant_id=user.tenant_id if user else None,
            action="LOGIN_FAILED",
            status="ERROR",
            details="Invalid username or password attempt.",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate JWT token
    token = create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
        tenant_id=user.tenant_id,
        full_name=user.full_name,
    )

    # Audit successful login
    AuditLogRepository.log_event(
        db,
        user_id=user.id,
        username=user.username,
        role=user.role,
        tenant_id=user.tenant_id,
        action="LOGIN",
        status="SUCCESS",
        details="User successfully authenticated.",
    )

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user=UserProfileResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            tenant_id=user.tenant_id,
            full_name=user.full_name,
        ),
    )


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user: UserModel = Depends(get_current_user),
):
    """Returns profile for currently authenticated user."""
    return UserProfileResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        tenant_id=current_user.tenant_id,
        full_name=current_user.full_name,
    )


@router.post("/logout")
async def logout_endpoint(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Logs audit event upon user logout."""
    AuditLogRepository.log_event(
        db,
        user_id=current_user.id,
        username=current_user.username,
        role=current_user.role,
        tenant_id=current_user.tenant_id,
        action="LOGOUT",
        status="SUCCESS",
        details="User session terminated.",
    )
    return {"message": "Logged out successfully."}
