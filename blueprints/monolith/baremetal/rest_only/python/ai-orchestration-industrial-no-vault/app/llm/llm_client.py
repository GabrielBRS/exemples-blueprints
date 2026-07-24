"""Seam do provedor de LLM — o mesmo papel do ExampleRepository nos backends.

O orquestrador LangGraph recebe um LLMClient e nao sabe se por tras ha vLLM,
Ollama, OpenAI ou um fake de teste. On-premise first: o client fala o dialeto
OpenAI-compatible via httpx puro (sem SDK), entao qualquer servidor que exponha
/chat/completions serve.
"""

from typing import Protocol

import httpx


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleClient:
    """POST {base_url}/chat/completions — funciona com vLLM, Ollama, LM Studio, OpenAI."""

    def __init__(self, base_url: str, model: str, api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def complete(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


class FakeLLMClient:
    """Dev/teste: respostas deterministicas, zero rede. Satisfaz o Protocol."""

    def __init__(self, canned: list[str] | None = None) -> None:
        self._canned = canned or []
        self._i = 0

    async def complete(self, system: str, user: str) -> str:
        if self._i < len(self._canned):
            out = self._canned[self._i]
            self._i += 1
            return out
        return f"[fake:{system[:24]}] {user[:120]}"
