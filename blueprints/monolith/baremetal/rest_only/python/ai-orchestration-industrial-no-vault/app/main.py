"""Composition root do orquestrador.

Aqui — e SO aqui — o nome em ORCHESTRATOR vira um adapter concreto. O resto do
app conhece apenas o Protocol. Adicionar um framework novo = escrever um adapter
+ registrar uma linha no _build_orchestrator.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request

from app.config import Config, load_config
from app.errors import install_error_handlers
from app.handlers import health_handler, orchestration_handler
from app.orchestrators.orchestrator import AgentOrchestrator
from app.services.orchestration_service import OrchestrationService


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _build_orchestrator(cfg: Config) -> AgentOrchestrator:
    """Registro de adapters. Imports lazy: so paga por quem for usado."""
    match cfg.orchestrator:
        case "echo":
            from app.orchestrators.echo_orchestrator import EchoOrchestrator
            return EchoOrchestrator()
        case "langgraph":
            from app.llm.llm_client import OpenAICompatibleClient
            from app.orchestrators.langgraph_orchestrator import LangGraphOrchestrator
            llm = OpenAICompatibleClient(
                cfg.llm_base_url, cfg.llm_model, cfg.llm_api_key.get_secret_value()
            )
            return LangGraphOrchestrator(llm)
        case "autogen":
            from app.orchestrators.autogen_orchestrator import AutoGenOrchestrator
            return AutoGenOrchestrator.from_openai_compatible(
                cfg.llm_base_url, cfg.llm_model, cfg.llm_api_key.get_secret_value()
            )
        case other:
            raise ValueError(f"ORCHESTRATOR desconhecido: {other!r} (use echo|langgraph|autogen)")


def create_app(
    config: Config | None = None,
    orchestrator_override: AgentOrchestrator | None = None,
) -> FastAPI:
    cfg = config or load_config()
    _setup_logging(cfg.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        orchestrator = orchestrator_override or _build_orchestrator(cfg)
        app.state.orchestration_service = OrchestrationService(orchestrator)
        logging.getLogger("app").info(
            "orquestrador '%s' pronto em http://%s:%s", orchestrator.name, cfg.host, cfg.port
        )
        yield

    app = FastAPI(title="ai-orchestrator-industrial-no-vault", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    install_error_handlers(app)
    app.include_router(health_handler.router)
    app.include_router(orchestration_handler.router)
    return app


def main() -> None:
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "app.main:create_app", factory=True,
        host=cfg.host, port=cfg.port, loop="uvloop", log_level=cfg.log_level.lower(),
    )


if __name__ == "__main__":
    main()
