import asyncio
from typing import Any

import requests

from app.config import get_settings


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

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, Any] = {
            "model": "bge-m3",
            "input": texts,
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

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError(
                f"Ollama embedding 返回的数据格式异常: {data}"
            )
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
