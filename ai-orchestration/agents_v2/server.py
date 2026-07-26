"""Borda HTTP — FastAPI sobre Granian (servidor em Rust).

Três coisas que só aparecem quando isso vai para o k3s:

1. **Durabilidade.** `durability="async"` grava o checkpoint fora do caminho
   crítico. Com `"exit"` você perde o thread se o pod cair no meio;
   com `"sync"` você paga fsync a cada superstep.

2. **Shutdown gracioso.** O k3s manda SIGTERM e conta até
   `terminationGracePeriodSeconds`. Sem `RunControl`, toda geração em voo
   morre no meio e o thread fica num estado que ninguém sabe retomar. Com
   ele, o runtime termina o superstep corrente, grava checkpoint e levanta
   `GraphDrained` — o mesmo `thread_id` continua no próximo pod.

3. **Interrupt não é erro.** `__interrupt__` chegando pelo stream é o
   fluxo normal de aprovação humana: emita para o front e espere o resume.
"""

from __future__ import annotations

import contextlib
import signal
from collections.abc import AsyncIterator
from typing import Any

import orjson
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from langgraph.types import Command
from pydantic import BaseModel, Field

from rag_graph.encoders import TritonEncoders
from rag_graph.graph import Deps, build_graph
from rag_graph.llm import VLLMClient
from rag_graph.settings import get_settings
from rag_graph.store import VectorStore


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    tenant_id: str
    thread_id: str


class ResumeRequest(BaseModel):
    thread_id: str
    aprovado: bool


def _sse(event: str, payload: Any) -> bytes:
    return b"event: %s\ndata: %s\n\n" % (event.encode(), orjson.dumps(payload))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Composition root: tudo que é caro nasce e morre aqui."""
    cfg = get_settings()

    llm = VLLMClient(
        cfg.llm_base_url,
        cfg.llm_model,
        api_key=cfg.llm_api_key.get_secret_value(),
        max_conns=cfg.llm_max_conns,
        legacy_guided=cfg.llm_legacy_guided,
    )
    encoders = TritonEncoders(
        cfg.triton_url,
        embed_model=cfg.embed_model,
        rerank_model=cfg.rerank_model,
        tokenizer_path=cfg.tokenizer_path,
        rerank_tokenizer_path=cfg.rerank_tokenizer_path,
        dim=cfg.embed_dim,
        max_tokens=cfg.embed_max_tokens,
    )
    store = await VectorStore.connect(
        cfg.dsn,
        min_size=cfg.pg_pool_min,
        max_size=cfg.pg_pool_max,
        stmt_cache=cfg.pg_stmt_cache,
        fts_config=cfg.fts_config,
        ef_search=cfg.hnsw_ef_search,
    )

    async with AsyncPostgresSaver.from_conn_string(cfg.dsn) as saver:
        await saver.setup()  # cria as tabelas de checkpoint (idempotente)
        app.state.graph = build_graph(
            Deps(llm=llm, encoders=encoders, store=store, cfg=cfg), checkpointer=saver
        )
        app.state.control = RunControl()
        app.state.store = store

        # Granian já roda sobre uvloop; aqui só interceptamos o sinal.
        signal.signal(signal.SIGTERM, lambda *_: app.state.control.request_drain("sigterm"))
        try:
            yield
        finally:
            await store.aclose()
            await encoders.aclose()
            await llm.aclose()


app = FastAPI(title="rag-graph", lifespan=lifespan)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> dict[str, str]:
    # live não toca o banco (senão um blip de rede reinicia o pod);
    # ready toca, porque sem banco não há retrieval nem checkpoint.
    if not await app.state.store.ping():
        raise HTTPException(status_code=503, detail="postgres indisponível")
    return {"status": "ready"}


@app.post("/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    graph = app.state.graph
    config = {"configurable": {"thread_id": req.thread_id}}

    async def events() -> AsyncIterator[bytes]:
        try:
            # version="v2": cada chunk é um StreamPart com type/ns/data,
            # em vez da tupla posicional do formato antigo.
            async for part in graph.astream(
                {"question": req.question, "tenant_id": req.tenant_id},
                config=config,
                stream_mode=["custom", "updates"],
                version="v2",
                durability="async",
                control=app.state.control,
            ):
                kind, data = part["type"], part["data"]

                if kind == "custom":
                    yield _sse("token", data)
                elif kind == "updates":
                    if "__interrupt__" in data:  # aprovação humana pendente
                        yield _sse("interrupt", data["__interrupt__"])
                    else:
                        yield _sse("step", {"node": next(iter(data), None)})
        except GraphDrained:
            # pod entrando em drain: o thread ficou retomável
            yield _sse("drained", {"thread_id": req.thread_id})
            return

        snapshot = await graph.aget_state(config)
        yield _sse(
            "done",
            {
                "answer": snapshot.values.get("answer", ""),
                "verdict": snapshot.values.get("verdict"),
                "citations": snapshot.values.get("citations", []),
                "trail": snapshot.values.get("trail", []),
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


@app.post("/resume")
async def resume(req: ResumeRequest) -> dict[str, Any]:
    """Retoma um thread parado num `interrupt`, possivelmente em outro pod."""
    config = {"configurable": {"thread_id": req.thread_id}}
    result = await app.state.graph.ainvoke(
        Command(resume={"aprovado": req.aprovado}), config=config, durability="async"
    )
    return {"answer": result.get("answer", "")}


@app.get("/threads/{thread_id}")
async def thread_state(thread_id: str) -> dict[str, Any]:
    snap = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    return {
        "next": list(snap.next),
        "values": {k: v for k, v in snap.values.items() if k != "candidates"},
    }
