import logging
from typing import Optional, List, Callable
from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session

from backend.auth.security import decode_access_token
from backend.database.database import get_db, UserRepository, AuditLogRepository
from backend.database.models import UserModel

logger = logging.getLogger(__name__)


async def get_current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> UserModel:
    """
    FastAPI dependency that extracts and validates JWT Bearer token,
    returning the authenticated UserModel.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.split(" ")[1].strip()
    payload = decode_access_token(token)

    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = UserRepository.get_by_id(db, user_id=payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or deactivated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_roles(*allowed_roles: str) -> Callable:
    """
    FastAPI dependency factory enforcing Role-Based Access Control (RBAC).
    Raises HTTP 403 Forbidden if user's role is not within allowed_roles.
    """
    def role_checker(
        current_user: UserModel = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> UserModel:
        if current_user.role not in allowed_roles:
            AuditLogRepository.log_event(
                db,
                user_id=current_user.id,
                username=current_user.username,
                role=current_user.role,
                tenant_id=current_user.tenant_id,
                action="UNAUTHORIZED_ACCESS",
                status="FORBIDDEN",
                details=f"User attempted to access restricted endpoint requiring roles: {allowed_roles}",
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action.",
            )
        return current_user

    return role_checker
