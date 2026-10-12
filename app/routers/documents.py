from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models import Chunk, Document
from app.schemas import UploadResponse
from app.services.chunking import chunk_text
from app.services.ollama import OllamaError, ollama_client
from app.services.pdf import PDFProcessingError, extract_pdf_pages

router = APIRouter(prefix="/documents", tags=["documents"])

settings = get_settings()
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持上传 PDF 文件")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="PDF 文件内容为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF 文件不能超过 25MB")

    try:
        pdf_result = extract_pdf_pages(content)
    except PDFProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = []
    for page in pdf_result["pages"]:
        for text in chunk_text(
            page["text"], settings.chunk_size, settings.chunk_overlap
        ):
            chunks.append({"page_number": page["page_number"], "content": text})

    if not chunks:
        raise HTTPException(
            status_code=400, detail="未能从 PDF 中提取到文本，可能是扫描件"
        )

    try:
        embeddings = await ollama_client.embed_texts(
            [chunk["content"] for chunk in chunks]
        )
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    document = Document(
        filename=file.filename,
        content_type=file.content_type or "application/pdf",
        num_pages=pdf_result["total_pages"],
        chunk_count=len(chunks),
    )

    try:
        db.add(document)
        await db.flush()
        db.add_all(
            [
                Chunk(
                    document_id=document.id,
                    chunk_index=index,
                    page_number=chunk["page_number"],
                    content=chunk["content"],
                    embedding=embedding,
                )
                for index, (chunk, embedding) in enumerate(
                    zip(chunks, embeddings), start=1
                )
            ]
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    await db.refresh(document)
    return UploadResponse(
        id=document.id,
        filename=document.filename,
        num_pages=document.num_pages,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        message=f"已解析并写入 {document.chunk_count} 个文本块",
    )
