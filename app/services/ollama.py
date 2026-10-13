import asyncio
from collections.abc import Callable
from typing import Any

import requests

from app.config import get_settings

EMBED_BATCH_SIZE = 16


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def aclose(self) -> None:
        return None

    def _request_url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _post_json(self, path: str, payload: dict[str, Any]) -> requests.Response:
        url = self._request_url(path)
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self._timeout,
                proxies={"http": None, "https": None},
            )
        except requests.RequestException as exc:
            print("Ollama 请求异常:", str(exc), flush=True)
            print("请求 URL:", url, flush=True)
            raise OllamaError(
                f"Ollama 请求异常: {str(exc)} (URL: {url})"
            ) from exc

        print("response.status_code:", response.status_code, flush=True)
        print("response.text:", response.text, flush=True)
        return response

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = EMBED_BATCH_SIZE,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        if batch_size <= 0:
            raise OllamaError("batch_size 必须大于 0")

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload: dict[str, Any] = {
                "model": settings.embedding_model,
                "input": batch,
            }
            response = await asyncio.to_thread(
                self._post_json, "/api/embed", payload
            )

            if response.status_code != 200:
                raise OllamaError(
                    f"Ollama embedding 失败 ({response.status_code}): "
                    f"{response.text}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise OllamaError(
                    f"Ollama embedding 返回的不是合法 JSON: {response.text}"
                ) from exc

            batch_embeddings = data.get("embeddings")
            if (
                not isinstance(batch_embeddings, list)
                or len(batch_embeddings) != len(batch)
            ):
                raise OllamaError(
                    f"Ollama embedding 返回的数据格式异常: {data}"
                )

            embeddings.extend(batch_embeddings)
            if progress_callback is not None:
                progress_callback(len(embeddings))

        return embeddings

    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": settings.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        response = await asyncio.to_thread(
            self._post_json, "/api/chat", payload
        )

        if response.status_code != 200:
            raise OllamaError(
                f"Ollama chat 失败 ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError(
                f"Ollama chat 返回的不是合法 JSON: {response.text}"
            ) from exc

        message = data.get("message") or {}
        answer = message.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise OllamaError(f"Ollama chat 返回的数据格式异常: {data}")
        return answer.strip()


settings = get_settings()
ollama_client = OllamaClient(settings.ollama_base_url, settings.request_timeout)
