"""Regra de negocio agnostica: dispara a orquestracao e guarda os runs.

Depende do Protocol AgentOrchestrator, nunca de framework. O historico de runs
fica num dict em memoria de proposito (foco deste scaffold e a orquestracao);
persistir em Postgres = copiar o padrao repository dos irmaos backend.
"""

from uuid import UUID

from app.dto.orchestration_dto import RunResponse, RunTaskRequest
from app.errors import NotFoundError, OrchestrationError
from app.models.orchestration import OrchestrationRun, RunStatus
from app.orchestrators.orchestrator import AgentOrchestrator


class OrchestrationService:
    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        self._orchestrator = orchestrator
        self._runs: dict[UUID, OrchestrationRun] = {}

    async def run_task(self, data: RunTaskRequest) -> RunResponse:
        run = await self._orchestrator.run(data.task)
        self._runs[run.id] = run
        if run.status is RunStatus.FAILED:
            raise OrchestrationError()
        return RunResponse.from_model(run)

    async def get_run(self, run_id: UUID) -> RunResponse:
        run = self._runs.get(run_id)
        if run is None:
            raise NotFoundError()
        return RunResponse.from_model(run)

    async def list_runs(self) -> list[RunResponse]:
        return [RunResponse.from_model(r) for r in self._runs.values()]
