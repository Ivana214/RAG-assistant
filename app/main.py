from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.database import dispose_db, init_db
from app.routers import ask, documents
from app.services.ollama import ollama_client

settings = get_settings()
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()
    yield
    await ollama_client.aclose()
    await dispose_db()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(ask.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
