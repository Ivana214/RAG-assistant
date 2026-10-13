from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.database import dispose_db, init_db
from app.routers import ask, documents
from app.services.ollama import ollama_client

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()
    yield
    await ollama_client.aclose()
    await dispose_db()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(ask.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
