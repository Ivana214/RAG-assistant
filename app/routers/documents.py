import asyncio
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.config import get_settings
from app.database import async_session_factory
from app.models import Chunk, Document
from app.schemas import TaskStatusResponse, UploadTaskResponse
from app.services.chunking import chunk_text
from app.services.ollama import OllamaError, ollama_client
from app.services.pdf import extract_pdf_pages

router = APIRouter(prefix="/documents", tags=["documents"])

settings = get_settings()
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/octet-stream",
    "application/x-pdf",
}
TASKS: dict[str, dict] = {}
STATUS_LABELS = {
    "queued": "排队中",
    "processing": "处理中",
    "completed": "已完成",
    "failed": "失败",
}


def _parse_and_chunk(content: bytes) -> tuple[int, list[dict]]:
    pdf_result = extract_pdf_pages(content)
    chunks = []
    for page in pdf_result["pages"]:
        for text in chunk_text(
            page["text"], settings.chunk_size, settings.chunk_overlap
        ):
            chunks.append({"page_number": page["page_number"], "content": text})
    return pdf_result["total_pages"], chunks


async def _process_pdf(
    task_id: str,
    filename: str,
    content_type: str | None,
    content: bytes,
) -> None:
    task = TASKS[task_id]
    try:
        task["status"] = "processing"

        total_pages, chunks = await asyncio.to_thread(_parse_and_chunk, content)
        total_chunks = len(chunks)
        task["total_chunks"] = total_chunks
        if total_chunks == 0:
            raise ValueError("未能从 PDF 中提取到文本，可能是扫描件")

        try:
            embeddings = await ollama_client.embed_texts(
                [chunk["content"] for chunk in chunks],
                progress_callback=lambda processed: task.update(
                    {"processed_chunks": processed}
                ),
            )
        except OllamaError as exc:
            raise ValueError(str(exc)) from exc

        records = list(zip(chunks, embeddings))

        async with async_session_factory() as db:
            document = Document(
                filename=filename,
                content_type=content_type or "application/pdf",
                num_pages=total_pages,
                chunk_count=total_chunks,
            )
            db.add(document)
            await db.flush()
            await db.run_sync(
                lambda session: session.bulk_insert_mappings(
                    Chunk,
                    [
                        {
                            "document_id": document.id,
                            "chunk_index": index,
                            "page_number": chunk["page_number"],
                            "content": chunk["content"],
                            "embedding": embedding,
                        }
                        for index, (chunk, embedding) in enumerate(
                            records, start=1
                        )
                    ],
                )
            )
            await db.commit()
            await db.refresh(document)
            task["document_id"] = document.id

        task["processed_chunks"] = total_chunks
        task["status"] = "completed"
        task["message"] = f"已解析并写入 {total_chunks} 个文本块"
    except Exception as exc:
        task["status"] = "failed"
        task["error"] = str(exc) or repr(exc)


@router.post(
    "/upload",
    response_model=UploadTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadTaskResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持上传 PDF 文件")

    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="不支持的文件类型，仅支持 PDF")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="PDF 文件内容为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF 文件不能超过 25MB")
    if not content.lstrip().startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="文件内容不是有效的 PDF")

    task_id = uuid4().hex
    TASKS[task_id] = {
        "task_id": task_id,
        "filename": file.filename,
        "status": "queued",
        "processed_chunks": 0,
        "total_chunks": 0,
        "message": None,
        "error": None,
        "document_id": None,
    }
    background_tasks.add_task(
        _process_pdf,
        task_id,
        file.filename,
        file.content_type,
        content,
    )
    return UploadTaskResponse(
        task_id=task_id,
        filename=file.filename,
        status="queued",
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return TaskStatusResponse(
        task_id=task["task_id"],
        filename=task["filename"],
        status=STATUS_LABELS.get(task["status"], task["status"]),
        progress=(
            f"已处理 {task['processed_chunks']} / 总 {task['total_chunks']} 块"
        ),
        processed_chunks=task["processed_chunks"],
        total_chunks=task["total_chunks"],
        message=task.get("message"),
        error=task.get("error"),
        document_id=task.get("document_id"),
    )
