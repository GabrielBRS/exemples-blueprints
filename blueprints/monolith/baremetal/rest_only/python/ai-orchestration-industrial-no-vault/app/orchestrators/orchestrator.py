"""O seam central: o framework de agentes e detalhe de implementacao.

Qualquer adapter promete a mesma coisa: recebe uma task, devolve um
OrchestrationRun preenchido (steps + output). O service, os handlers e os DTOs
nunca importam langgraph/autogen — se um framework morrer ou surgir outro,
troca-se o adapter e NADA acima muda.
"""

from typing import Protocol

from app.models.orchestration import OrchestrationRun


class AgentOrchestrator(Protocol):
    name: str

    async def run(self, task: str) -> OrchestrationRun: ...
