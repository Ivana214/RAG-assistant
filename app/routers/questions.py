from fastapi import APIRouter, Depends
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Question
from app.schemas import QuestionResponse

router = APIRouter(tags=["questions"])


@router.get("/questions", response_model=list[QuestionResponse])
async def list_questions(
    knowledge_point: str | None = None,
    q: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[Question]:
    stmt = select(Question).order_by(Question.created_at.desc())

    if knowledge_point:
        stmt = stmt.where(
            Question.knowledge_point.ilike(f"%{knowledge_point}%")
        )

    if q:
        pattern = f"%{q}%"
        stmt = stmt.where(
            or_(
                Question.content.ilike(pattern),
                Question.answer.ilike(pattern),
                Question.knowledge_point.ilike(pattern),
            )
        )

    stmt = stmt.limit(200)
    return (await db.execute(stmt)).scalars().all()
