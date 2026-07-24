"""Probes: /health/live (processo) e /health/ready (orquestrador montado)."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    svc = getattr(request.app.state, "orchestration_service", None)
    if svc is None:
        return JSONResponse(status_code=503, content={"status": "unready"})
    return JSONResponse(status_code=200, content={"status": "ready"})
