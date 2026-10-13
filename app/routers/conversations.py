from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Conversation
from app.schemas import ConversationResponse

router = APIRouter(tags=["conversations"])


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
) -> list[Conversation]:
    stmt = (
        select(Conversation)
        .order_by(Conversation.created_at.desc())
        .limit(20)
    )
    return (await db.execute(stmt)).scalars().all()
