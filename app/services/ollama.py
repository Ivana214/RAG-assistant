from typing import Any

import httpx

from app.config import get_settings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, Any] = {
            "model": settings.embedding_model,
            "input": texts,
        }
        try:
            response = await self._client.post("/api/embed", json=payload)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama embedding 请求失败: {exc}") from exc

        if response.status_code != 200:
            raise OllamaError(
                f"Ollama embedding 失败 ({response.status_code}): "
                f"{response.text[:300]}"
            )

        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError("Ollama 返回的 embedding 数据格式异常")
        return embeddings

    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": settings.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama chat 请求失败: {exc}") from exc

        if response.status_code != 200:
            raise OllamaError(
                f"Ollama chat 失败 ({response.status_code}): {response.text[:300]}"
            )

        data = response.json()
        message = data.get("message") or {}
        answer = message.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise OllamaError("Ollama 返回的对话数据格式异常")
        return answer.strip()


settings = get_settings()
ollama_client = OllamaClient(settings.ollama_base_url, settings.request_timeout)
