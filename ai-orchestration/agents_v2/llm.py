"""Cliente vLLM — httpx puro + orjson, sem SDK.

Por que sem o SDK da OpenAI / sem `langchain-openai`:
  * o SDK traz Pydantic parsing por resposta e um retry loop próprio que
    conflita com o RetryPolicy do LangGraph (retry em duas camadas);
  * `langchain-openai` converte tudo para BaseMessage e de volta — alocação
    pura em cima de um payload que já é JSON;
  * on-prem não há multi-provider para abstrair: é vLLM, e o contrato é
    estável há anos.

O que sobra em Python aqui é montar um dict e ler bytes de um socket. Todo
o resto (JSON, TLS, HTTP, decode) está em C/Rust ou na GPU.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Final, Self

import httpx
import orjson

_SSE_PREFIX: Final[bytes] = b"data: "
_SSE_DONE: Final[bytes] = b"[DONE]"

Message = Mapping[str, str]


class LLMError(RuntimeError):
    """Falha de transporte ou de contrato do servidor de inferência."""


class VLLMClient:
    """Um cliente por processo. O pool de conexões é o recurso caro."""

    __slots__ = ("_http", "_model", "_legacy_guided")

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        max_conns: int = 64,
        legacy_guided: bool = False,
    ) -> None:
        self._model = model
        self._legacy_guided = legacy_guided
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "content-type": "application/json",
                **({"authorization": f"Bearer {api_key}"} if api_key else {}),
            },
            # Keepalive largo: reconectar por request destrói o p99 e
            # atrapalha o prefix caching do vLLM (afinidade de conexão).
            limits=httpx.Limits(
                max_connections=max_conns,
                max_keepalive_connections=max_conns,
                keepalive_expiry=90.0,
            ),
            timeout=httpx.Timeout(connect=2.0, read=180.0, write=10.0, pool=5.0),
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ #
    # payload
    # ------------------------------------------------------------------ #
    def _payload(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        stream: bool,
        json_schema: Mapping[str, Any] | None,
        choice: Sequence[str] | None,
        regex: str | None,
    ) -> bytes:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stream:
            # usage no último chunk, sem custo extra de round-trip
            body["stream_options"] = {"include_usage": True}

        # Decodificação restrita. O backend xgrammar compila a gramática uma
        # vez e mascara logits no device — custo ~0 no steady state e o
        # parse do lado do cliente deixa de poder falhar.
        so: dict[str, Any] = {}
        if json_schema is not None:
            so["json"] = json_schema
        if choice is not None:
            so["choice"] = list(choice)
        if regex is not None:
            so["regex"] = regex
        if so:
            if self._legacy_guided:  # builds anteriores à unificação
                body |= {f"guided_{k}": v for k, v in so.items()}
            else:
                body["structured_outputs"] = so

        return orjson.dumps(body)

    # ------------------------------------------------------------------ #
    # chamadas
    # ------------------------------------------------------------------ #
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: Mapping[str, Any] | None = None,
        choice: Sequence[str] | None = None,
        regex: str | None = None,
    ) -> str:
        payload = self._payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            json_schema=json_schema,
            choice=choice,
            regex=regex,
        )
        resp = await self._http.post("/chat/completions", content=payload)
        if resp.status_code >= 400:
            raise LLMError(f"vLLM {resp.status_code}: {resp.text[:400]}")
        data = orjson.loads(resp.content)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:  # pragma: no cover
            raise LLMError(f"resposta inesperada: {resp.content[:400]!r}") from exc

    async def complete_json(
        self,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        *,
        max_tokens: int = 512,
    ) -> Any:
        """JSON garantido pela gramática — sem try/except de parsing, sem retry."""
        raw = await self.complete(
            messages, temperature=0.0, max_tokens=max_tokens, json_schema=schema
        )
        return orjson.loads(raw)

    async def classify(self, messages: Sequence[Message], labels: Sequence[str]) -> str:
        """Rota/veredito: `choice` restringe a saída ao conjunto exato.

        Custa ~1 token decodificado. É o jeito barato de fazer roteamento —
        não gaste um schema JSON inteiro para escolher entre três palavras.
        """
        return (
            await self.complete(messages, temperature=0.0, max_tokens=8, choice=labels)
        ).strip()

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1536,
    ) -> AsyncIterator[str]:
        payload = self._payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            json_schema=None,
            choice=None,
            regex=None,
        )
        async with self._http.stream("POST", "/chat/completions", content=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise LLMError(f"vLLM {resp.status_code}: {body[:400]!r}")
            async for line in resp.aiter_lines():
                raw = line.encode()
                if not raw.startswith(_SSE_PREFIX):
                    continue
                chunk = raw[len(_SSE_PREFIX) :]
                if chunk == _SSE_DONE:
                    return
                delta = orjson.loads(chunk)["choices"]
                if not delta:  # chunk final só com usage
                    continue
                if piece := delta[0]["delta"].get("content"):
                    yield piece
