from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.models import Base

settings = get_settings()

engine = create_async_engine(settings.database_url, pool_pre_ping=True)
async_session_factory = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS "
                "collection VARCHAR(64) NOT NULL DEFAULT 'default'"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_documents_collection "
                "ON documents (collection)"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata JSONB"
            )
        )


async def dispose_db() -> None:
    await engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
