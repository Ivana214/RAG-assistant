from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    num_pages: int
    chunk_count: int
    collection: str
    created_at: datetime


class UploadResponse(DocumentOut):
    message: str


class UploadTaskResponse(BaseModel):
    task_id: str
    filename: str
    collection: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    filename: str
    collection: str
    status: str
    progress: str
    processed_chunks: int
    total_chunks: int
    message: str | None = None
    error: str | None = None
    document_id: int | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)
    collection: str = Field(default="default", max_length=64)


class Source(BaseModel):
    document_id: int
    filename: str
    page_number: int
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    answer: str
    sources: list[Source]
    collection: str
    created_at: datetime


class QuestionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    content: str
    answer: str
    question_type: str | None = None
    knowledge_point: str | None = None
    source_document_id: int | None = None
    created_at: datetime
