from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    gender = Column(String)
    created_at = Column(DateTime, server_default=func.now())

class HealthProfile(Base):
    __tablename__ = "health_profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    conditions = Column(Text, default="")
    medications = Column(Text, default="")
    updated_at = Column(DateTime, server_default=func.now())

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    message = Column(Text)
    response = Column(Text)
    agent_name = Column(String)
    severity = Column(String)
    escalated = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    policy_action = Column(String)   # BLOCKED / ESCALATED / ALLOWED
    message_snippet = Column(String) # first 100 chars only
    created_at = Column(DateTime, server_default=func.now())