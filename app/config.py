from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RAGzhiyan"
    database_url: str = (
        "postgresql+asyncpg://admin:password123@localhost:5432/kaoyan_db"
    )
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "qwen2.5:7b"
    chunk_size: int = 500
    chunk_overlap: int = 0
    embedding_dim: int = 768
    request_timeout: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
