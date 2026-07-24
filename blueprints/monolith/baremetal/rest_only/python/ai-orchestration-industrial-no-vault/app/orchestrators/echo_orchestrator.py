"""Adapter de dev/teste: sem LLM, sem framework. Prova que o seam funciona e
serve de gabarito minimo para escrever um adapter novo."""

from datetime import datetime, timezone

from app.models.orchestration import OrchestrationRun, RunStatus, Step


class EchoOrchestrator:
    name = "echo"

    async def run(self, task: str) -> OrchestrationRun:
        run = OrchestrationRun(task=task, orchestrator=self.name)
        run.steps.append(Step(agent="echo", output=f"recebi: {task}"))
        run.output = task
        run.status = RunStatus.SUCCEEDED
        run.finished_at = datetime.now(timezone.utc)
        return run
