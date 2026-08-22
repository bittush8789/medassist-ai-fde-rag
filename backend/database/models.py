import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Boolean, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class UserModel(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="CUSTOMER")  # ADMIN, FDE_ENGINEER, CUSTOMER
    tenant_id = Column(String(64), nullable=False, default="customer_001", index=True)
    full_name = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), nullable=True, index=True)
    username = Column(String(64), nullable=True, index=True)
    role = Column(String(32), nullable=True)
    tenant_id = Column(String(64), nullable=True, index=True)
    action = Column(String(64), nullable=False, index=True)  # LOGIN, RAG_QUERY, DOCUMENT_ACCESS, etc.
    status = Column(String(32), nullable=False, default="SUCCESS")  # SUCCESS, FORBIDDEN, ERROR
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)


class ConversationModel(Base):
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=True, index=True)
    tenant_id = Column(String(64), nullable=False, default="customer_001", index=True)
    title = Column(String(255), default="New Medical Consultation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    messages = relationship("MessageModel", back_populates="conversation", cascade="all, delete-orphan", order_by="MessageModel.timestamp")


class MessageModel(Base):
    __tablename__ = "messages"

    id = Column(String(64), primary_key=True, index=True)
    conversation_id = Column(String(64), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" or "assistant" or "system"
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)  # List of citations/sources attached to assistant message
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    conversation = relationship("ConversationModel", back_populates="messages")
