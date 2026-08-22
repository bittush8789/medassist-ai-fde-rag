import uuid
from datetime import datetime, timezone
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from backend.config import settings
from backend.database.models import Base, UserModel, AuditLogModel, ConversationModel, MessageModel
from backend.auth.security import hash_password

logger = logging.getLogger(__name__)

# Create engine for SQLite
connect_args = {"check_same_thread": False} if "sqlite" in settings.database_url else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def seed_default_users(db: Session):
    """Pre-seeds default enterprise RBAC demo accounts if they do not exist."""
    demo_users = [
        {
            "id": "usr_admin_001",
            "username": "admin",
            "email": "admin@hospital-system.org",
            "password": "Admin@12345",
            "role": "ADMIN",
            "tenant_id": "system",
            "full_name": "System Administrator",
        },
        {
            "id": "usr_fde_001",
            "username": "engineer",
            "email": "fde.engineer@antigravity.ai",
            "password": "Engineer@12345",
            "role": "FDE_ENGINEER",
            "tenant_id": "customer_001",
            "full_name": "Alex Mercer (FDE Lead)",
        },
        {
            "id": "usr_cust_001",
            "username": "customer1",
            "email": "dr.smith@metrohealth.org",
            "password": "Customer@12345",
            "role": "CUSTOMER",
            "tenant_id": "customer_001",
            "full_name": "Dr. Sarah Smith (MetroHealth)",
        },
        {
            "id": "usr_cust_002",
            "username": "customer2",
            "email": "dr.jones@apexclinic.org",
            "password": "Customer@12345",
            "role": "CUSTOMER",
            "tenant_id": "customer_002",
            "full_name": "Dr. Marcus Jones (Apex Clinic)",
        },
    ]

    for u in demo_users:
        existing = db.query(UserModel).filter(UserModel.username == u["username"]).first()
        if not existing:
            user = UserModel(
                id=u["id"],
                username=u["username"],
                email=u["email"],
                hashed_password=hash_password(u["password"]),
                role=u["role"],
                tenant_id=u["tenant_id"],
                full_name=u["full_name"],
                is_active=True,
                created_at=utc_now(),
            )
            db.add(user)
            logger.info(f"Seeded default demo user '{u['username']}' with role '{u['role']}'.")
    db.commit()


def init_db():
    """Initializes the database schema and seeds demo credentials."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_default_users(db)
    finally:
        db.close()
    logger.info("Database tables and seed users initialized.")


def get_db():
    """FastAPI Dependency for obtaining database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class UserRepository:
    """Repository handling user management and credential verification."""

    @staticmethod
    def get_by_id(db: Session, user_id: str) -> Optional[UserModel]:
        return db.query(UserModel).filter(UserModel.id == user_id).first()

    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[UserModel]:
        return db.query(UserModel).filter(UserModel.username == username).first()

    @staticmethod
    def list_users(db: Session, limit: int = 100) -> List[UserModel]:
        return db.query(UserModel).order_by(UserModel.created_at.desc()).limit(limit).all()

    @staticmethod
    def create_user(
        db: Session,
        username: str,
        email: str,
        password: str,
        role: str,
        tenant_id: str,
        full_name: str,
    ) -> UserModel:
        user = UserModel(
            id=f"usr_{uuid.uuid4().hex[:12]}",
            username=username.strip().lower(),
            email=email.strip().lower(),
            hashed_password=hash_password(password),
            role=role.upper(),
            tenant_id=tenant_id.strip().lower(),
            full_name=full_name.strip(),
            is_active=True,
            created_at=utc_now(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def delete_user(db: Session, user_id: str) -> bool:
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        if user:
            db.delete(user)
            db.commit()
            return True
        return False


class AuditLogRepository:
    """Repository handling security audit logs for RBAC actions."""

    @staticmethod
    def log_event(
        db: Session,
        user_id: Optional[str],
        username: Optional[str],
        role: Optional[str],
        tenant_id: Optional[str],
        action: str,
        status: str = "SUCCESS",
        details: Optional[str] = None,
    ) -> AuditLogModel:
        log_entry = AuditLogModel(
            id=f"log_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            username=username,
            role=role,
            tenant_id=tenant_id,
            action=action,
            status=status,
            details=details,
            timestamp=utc_now(),
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        return log_entry

    @staticmethod
    def list_logs(db: Session, limit: int = 100) -> List[AuditLogModel]:
        return db.query(AuditLogModel).order_by(AuditLogModel.timestamp.desc()).limit(limit).all()


class ConversationRepository:
    """Repository handling all database operations for conversations and messages with tenant isolation."""

    @staticmethod
    def get_or_create_conversation(
        db: Session,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: str = "customer_001",
    ) -> ConversationModel:
        if conversation_id:
            conv = db.query(ConversationModel).filter(ConversationModel.id == conversation_id).first()
            if conv:
                return conv

        new_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        conv = ConversationModel(
            id=new_id,
            user_id=user_id,
            tenant_id=tenant_id,
            title=title or "New Medical Consultation",
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv

    @staticmethod
    def list_conversations(
        db: Session,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ConversationModel]:
        query = db.query(ConversationModel)
        if user_id:
            query = query.filter(ConversationModel.user_id == user_id)
        elif tenant_id and tenant_id != "system":
            query = query.filter(ConversationModel.tenant_id == tenant_id)
        return query.order_by(ConversationModel.updated_at.desc()).limit(limit).all()

    @staticmethod
    def get_conversation(db: Session, conversation_id: str) -> Optional[ConversationModel]:
        return db.query(ConversationModel).filter(ConversationModel.id == conversation_id).first()

    @staticmethod
    def delete_conversation(db: Session, conversation_id: str, user_id: Optional[str] = None) -> bool:
        query = db.query(ConversationModel).filter(ConversationModel.id == conversation_id)
        if user_id:
            query = query.filter(ConversationModel.user_id == user_id)
        conv = query.first()
        if conv:
            db.delete(conv)
            db.commit()
            return True
        return False

    @staticmethod
    def update_conversation_title(db: Session, conversation_id: str, title: str):
        conv = db.query(ConversationModel).filter(ConversationModel.id == conversation_id).first()
        if conv:
            conv.title = title
            conv.updated_at = utc_now()
            db.commit()

    @staticmethod
    def add_message(
        db: Session,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> MessageModel:
        msg = MessageModel(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources,
            timestamp=utc_now(),
        )
        db.add(msg)

        conv = db.query(ConversationModel).filter(ConversationModel.id == conversation_id).first()
        if conv:
            conv.updated_at = utc_now()

        db.commit()
        db.refresh(msg)
        return msg

    @staticmethod
    def get_conversation_history(db: Session, conversation_id: str, limit: int = 10) -> List[MessageModel]:
        return (
            db.query(MessageModel)
            .filter(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.timestamp.asc())
            .limit(limit)
            .all()
        )
