"""AppError unificado -> resposta JSON. Igual aos irmaos backend."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("app")


class AppError(Exception):
    status_code = 500
    message = "erro interno do servidor"


class NotFoundError(AppError):
    status_code = 404
    message = "recurso nao encontrado"


class BadRequestError(AppError):
    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class OrchestrationError(AppError):
    """Falha na execucao do orquestrador (LLM fora, framework quebrou...)."""
    status_code = 502
    message = "falha na orquestracao"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("erro na orquestracao", exc_info=exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"status": exc.status_code, "message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("erro nao tratado", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"status": 500, "message": "erro interno do servidor"}},
        )
