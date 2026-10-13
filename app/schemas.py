from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    num_pages: int
    chunk_count: int
    created_at: datetime


class UploadResponse(DocumentOut):
    message: str


class UploadTaskResponse(BaseModel):
    task_id: str
    filename: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    filename: str
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


class Source(BaseModel):
    document_id: int
    filename: str
    page_number: int
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
