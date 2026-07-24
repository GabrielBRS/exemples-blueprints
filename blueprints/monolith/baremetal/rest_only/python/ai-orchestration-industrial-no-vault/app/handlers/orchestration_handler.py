"""Adapta HTTP <-> service. Sem regra de negocio, sem framework de IA."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status

from app.dto.orchestration_dto import RunResponse, RunTaskRequest
from app.services.orchestration_service import OrchestrationService

router = APIRouter(prefix="/api/v1/orchestrations", tags=["orchestrations"])


def get_service(request: Request) -> OrchestrationService:
    return request.app.state.orchestration_service


ServiceDep = Annotated[OrchestrationService, Depends(get_service)]


@router.post("/", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
async def run_task(payload: RunTaskRequest, service: ServiceDep) -> RunResponse:
    return await service.run_task(payload)


@router.get("/", response_model=list[RunResponse])
async def list_runs(service: ServiceDep) -> list[RunResponse]:
    return await service.list_runs()


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: UUID, service: ServiceDep) -> RunResponse:
    return await service.get_run(run_id)
