"""Contratos de entrada/saida da API."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.orchestration import OrchestrationRun


class RunTaskRequest(BaseModel):
    task: str = Field(min_length=1, max_length=8000)


class StepResponse(BaseModel):
    agent: str
    output: str


class RunResponse(BaseModel):
    id: UUID
    task: str
    orchestrator: str
    status: str
    steps: list[StepResponse]
    output: str | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None

    @classmethod
    def from_model(cls, run: OrchestrationRun) -> "RunResponse":
        return cls(
            id=run.id,
            task=run.task,
            orchestrator=run.orchestrator,
            status=run.status.value,
            steps=[StepResponse(agent=s.agent, output=s.output) for s in run.steps],
            output=run.output,
            error=run.error,
            created_at=run.created_at,
            finished_at=run.finished_at,
        )
