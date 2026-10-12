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
        "postgresql+psycopg://admin:password123@localhost:5432/kaoyan_db"
    )
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    chat_model: str = "qwen2.5:7b"
    chunk_size: int = 800
    chunk_overlap: int = 160
    embedding_dim: int = 1024
    request_timeout: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
