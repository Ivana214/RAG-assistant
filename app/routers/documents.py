import asyncio
import json
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import delete, select

from app.config import get_settings
from app.database import async_session_factory
from app.models import Chunk, Document, Question, Task
from app.schemas import TaskStatusResponse, UploadTaskResponse
from app.services.chunking import chunk_text
from app.services.ollama import OllamaError, ollama_client
from app.services.parser import ParserError, parse_file

router = APIRouter(prefix="/documents", tags=["documents"])

settings = get_settings()
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".csv", ".docx", ".pptx"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/csv",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/octet-stream",
}
STATUS_LABELS = {
    "queued": "排队中",
    "processing": "处理中",
    "completed": "已完成",
    "failed": "失败",
}


def _parse_and_chunk(filename: str, content: bytes) -> tuple[int, list[dict]]:
    parsed = parse_file(filename, content)
    chunks = []
    for item in parsed["items"]:
        for text in chunk_text(
            item["text"], settings.chunk_size, settings.chunk_overlap
        ):
            chunks.append(
                {
                    "page_number": item["page_number"],
                    "content": text,
                    "metadata": item["metadata"],
                }
            )
    return parsed["total_items"], chunks


async def _set_task(task_id: str, **fields) -> None:
    async with async_session_factory() as db:
        task = await db.get(Task, task_id)
        if task is None:
            return
        for key, value in fields.items():
            setattr(task, key, value)
        await db.commit()


async def _set_progress(task_id: str, processed: int) -> None:
    await _set_task(task_id, processed_chunks=processed)


def _parse_question_json(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if len(lines) > 2 else lines)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("模型未返回 JSON 数组")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("模型返回的不是 JSON 数组")
    return data


async def _extract_questions(raw_text: str) -> list[dict]:
    prompt = (
        "你是题库整理助手。请从下面的资料中提取所有题目，"
        "输出一个 JSON 数组，每个元素包含 content、answer、"
        "question_type、knowledge_point 四个字段。"
        "不要输出 JSON 之外的任何内容。\n\n资料:\n"
        f"{raw_text[:12000]}"
    )
    messages = [
        {
            "role": "system",
            "content": "你负责从学习资料中提取结构化题目。",
        },
        {"role": "user", "content": prompt},
    ]
    answer = await ollama_client.chat(messages)
    questions = _parse_question_json(answer)
    result = []
    for item in questions:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        result.append(
            {
                "content": content,
                "answer": str(item.get("answer") or "").strip(),
                "question_type": str(item.get("question_type") or "").strip()
                or None,
                "knowledge_point": str(item.get("knowledge_point") or "")
                .strip()
                or None,
            }
        )
    return result


async def _process_pdf(
    task_id: str,
    filename: str,
    content_type: str | None,
    collection: str,
    content: bytes,
    extract_questions: bool = False,
) -> None:
    try:
        await _set_task(task_id, status="processing")

        total_pages, chunks = await asyncio.to_thread(
            _parse_and_chunk, filename, content
        )
        total_chunks = len(chunks)
        await _set_task(task_id, total_chunks=total_chunks)
        if total_chunks == 0:
            raise ValueError("未能从 PDF 中提取到文本，可能是扫描件")

        try:
            embeddings = await ollama_client.embed_texts(
                [chunk["content"] for chunk in chunks],
                progress_callback=lambda processed: _set_progress(
                    task_id, processed
                ),
            )
        except OllamaError as exc:
            raise ValueError(str(exc)) from exc

        records = list(zip(chunks, embeddings))
        questions = []

        async with async_session_factory() as db:
            old_ids = (
                await db.execute(
                    select(Document.id).where(
                        Document.filename == filename,
                        Document.collection == collection,
                    )
                )
            ).scalars().all()
            for document_id in old_ids:
                await db.execute(
                    delete(Chunk).where(Chunk.document_id == document_id)
                )
                await db.execute(
                    delete(Document).where(Document.id == document_id)
                )

            document = Document(
                filename=filename,
                content_type=content_type or "application/pdf",
                num_pages=total_pages,
                chunk_count=total_chunks,
                collection=collection,
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
                            "extra": chunk["metadata"],
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
            document_id = document.id

        if extract_questions and Path(filename).suffix.lower() in {
            ".pdf",
            ".docx",
        }:
            raw_text = "\n".join(chunk["content"] for chunk in chunks)
            try:
                questions = await _extract_questions(raw_text)
            except (OllamaError, ValueError) as exc:
                raise ValueError(f"题库提取失败: {exc}") from exc

            if questions:
                async with async_session_factory() as question_db:
                    question_db.add_all(
                        [
                            Question(
                                content=question["content"],
                                answer=question["answer"],
                                question_type=question["question_type"],
                                knowledge_point=question["knowledge_point"],
                                source_document_id=document_id,
                            )
                            for question in questions
                        ]
                    )
                    await question_db.commit()

        await _set_task(
            task_id,
            processed_chunks=total_chunks,
            status="completed",
            message=(
                f"已解析并写入 {total_chunks} 个文本块"
                + (
                    f"，提取 {len(questions)} 道题目"
                    if extract_questions and questions
                    else ""
                )
            ),
            document_id=document_id,
        )
    except Exception as exc:
        await _set_task(task_id, status="failed", error=str(exc) or repr(exc))


@router.post(
    "/upload",
    response_model=UploadTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form("default"),
    extract_questions: bool = Form(False),
) -> UploadTaskResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="只支持上传 PDF、Excel、CSV、Word、PPT 文件",
        )

    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="不支持的文件类型，仅支持 PDF、Excel、CSV、Word、PPT",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件不能超过 25MB")
    if extension == ".pdf" and not content.lstrip().startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="文件内容不是有效的 PDF")

    collection = (collection or "default").strip() or "default"
    task_id = uuid4().hex
    async with async_session_factory() as db:
        db.add(
            Task(
                task_id=task_id,
                filename=file.filename,
                collection=collection,
                status="queued",
            )
        )
        await db.commit()

    background_tasks.add_task(
        _process_pdf,
        task_id,
        file.filename,
        file.content_type,
        collection,
        content,
        extract_questions,
    )
    return UploadTaskResponse(
        task_id=task_id,
        filename=file.filename,
        collection=collection,
        status="queued",
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    async with async_session_factory() as db:
        task = await db.get(Task, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return TaskStatusResponse(
            task_id=task.task_id,
            filename=task.filename,
            collection=task.collection,
            status=STATUS_LABELS.get(task.status, task.status),
            progress=(
                f"已处理 {task.processed_chunks} / 总 {task.total_chunks} 块"
            ),
            processed_chunks=task.processed_chunks,
            total_chunks=task.total_chunks,
            message=task.message,
            error=task.error,
            document_id=task.document_id,
        )
