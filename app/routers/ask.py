from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Chunk, Conversation, Document
from app.schemas import AskRequest, AskResponse, Source
from app.services.ollama import OllamaError, ollama_client

router = APIRouter(prefix="/ask", tags=["ask"])

SYSTEM_PROMPT = (
    "你是一名严谨的考研备考助手。请只依据提供的资料回答用户问题，"
    "并在回答中引用资料编号，例如 [1]、[2]。"
    "如果资料不足以回答，请明确说明资料中没有相关内容。"
)


def build_prompt(context_blocks: list[str], question: str) -> str:
    numbered = "\n\n".join(
        f"[{index}] {block}" for index, block in enumerate(context_blocks, start=1)
    )
    return f"资料:\n{numbered}\n\n问题: {question}"


@router.post("", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    db: AsyncSession = Depends(get_db),
) -> AskResponse:
    try:
        question_embedding = (await ollama_client.embed_texts([payload.question]))[0]
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    distance = Chunk.embedding.cosine_distance(question_embedding)
    stmt = (
        select(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.collection == payload.collection)
        .order_by(distance)
        .limit(payload.top_k)
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        answer = "数据库中没有可检索的资料，请先上传 PDF。"
        sources = []
    else:
        context_blocks = []
        sources = []
        for chunk, filename, chunk_distance in rows:
            context_blocks.append(
                f"来源: {filename} 第{chunk.page_number}页\n{chunk.content}"
            )
            sources.append(
                Source(
                    document_id=chunk.document_id,
                    filename=filename,
                    page_number=chunk.page_number,
                    snippet=chunk.content[:200],
                    score=round(1 - chunk_distance, 4),
                )
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_prompt(context_blocks, payload.question),
            },
        ]
        try:
            answer = await ollama_client.chat(messages)
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    db.add(
        Conversation(
            question=payload.question,
            answer=answer,
            sources=[source.model_dump() for source in sources],
            collection=payload.collection,
        )
    )
    await db.commit()

    return AskResponse(
        answer=answer,
        sources=sources,
    )
