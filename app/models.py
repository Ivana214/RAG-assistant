from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    num_pages: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    collection: Mapped[str] = mapped_column(
        String(64), nullable=False, default="default", server_default="default"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    question_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    knowledge_point: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_document_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    extra: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(get_settings().embedding_dim), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    collection: Mapped[str] = mapped_column(
        String(64), nullable=False, default="default", server_default="default"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Task(Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    collection: Mapped[str] = mapped_column(
        String(64), nullable=False, default="default", server_default="default"
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="queued"
    )
    processed_chunks: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
