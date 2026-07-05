import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, DateTime, Enum, Integer, Float, JSON, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


def utcnow():
    dt = datetime.now(timezone.utc)
    return dt.replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    source = Column(String, nullable=False)
    doc_type = Column(String, nullable=False, default="textbook")
    authors = Column(JSONB, default=list)
    date = Column(DateTime, nullable=True)
    raw_file_path = Column(String, nullable=True)
    status = Column(String, nullable=False, default="processing")
    created_at = Column(DateTime, default=utcnow)
    index_completed = Column(Boolean, default=False)
    chunks_count = Column(Integer, default=0)
    file_hash = Column(String, nullable=True, unique=True, index=True)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    page = Column(Integer, nullable=True)
    section = Column(String, nullable=True)
    position = Column(Integer, nullable=False)
    metadata_ = Column("metadata", JSONB, default=dict)

    document = relationship("Document", back_populates="chunks")


class Problem(Base):
    __tablename__ = "problems"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    statement = Column(Text, nullable=False)
    target_kpi = Column(String, nullable=True)
    target_delta = Column(String, nullable=True)
    constraints = Column(JSONB, nullable=True)
    domain = Column(String, nullable=True)
    status = Column(String, nullable=False, default="created")
    document_ids = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    hypotheses = relationship("Hypothesis", back_populates="problem", cascade="all, delete-orphan")


class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    problem_id = Column(UUID(as_uuid=True), ForeignKey("problems.id"), nullable=False)
    statement = Column(Text, nullable=False)
    mechanism = Column(Text, nullable=False)
    citations = Column(JSONB, default=list)
    novelty = Column(Float, nullable=False)
    feasibility = Column(Float, nullable=False)
    impact = Column(Float, nullable=False)
    risk = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    composite_score = Column(Float, nullable=False)
    reasoning_trace = Column(Text, nullable=True)
    risks = Column(JSONB, default=list)
    verification_plan = Column(Text, nullable=True)
    feedback_status = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    problem = relationship("Problem", back_populates="hypotheses")
