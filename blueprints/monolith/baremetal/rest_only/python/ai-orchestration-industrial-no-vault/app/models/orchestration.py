"""Entidades de dominio da orquestracao — agnosticas de framework.

Qualquer adapter (LangGraph, AutoGen, o proximo que surgir) tem que se traduzir
PARA estas estruturas. E o contrato que blinda o resto do app da rotatividade
de frameworks de agentes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class Step:
    """Um passo da orquestracao: qual agente/no falou e o que produziu."""
    agent: str
    output: str


@dataclass(slots=True)
class OrchestrationRun:
    task: str
    orchestrator: str
    id: UUID = field(default_factory=uuid4)
    status: RunStatus = RunStatus.RUNNING
    steps: list[Step] = field(default_factory=list)
    output: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
