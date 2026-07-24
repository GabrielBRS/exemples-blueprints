"""Adapter AutoGen (autogen-agentchat): time planner + executor em RoundRobin.

Import lazy, como no LangGraph. Aqui o model client e do proprio AutoGen
(ChatCompletionClient); recebemos ele injetado para os testes usarem o
ReplayChatCompletionClient (respostas gravadas, zero LLM).
"""

from datetime import datetime, timezone
from typing import Any

from app.models.orchestration import OrchestrationRun, RunStatus, Step


class AutoGenOrchestrator:
    name = "autogen"

    def __init__(self, model_client: Any, max_messages: int = 4) -> None:
        self._model_client = model_client
        self._max_messages = max_messages

    @classmethod
    def from_openai_compatible(cls, base_url: str, model: str, api_key: str) -> "AutoGenOrchestrator":
        """Producao: vLLM/Ollama/OpenAI via client OpenAI do autogen-ext."""
        from autogen_ext.models.openai import OpenAIChatCompletionClient  # lazy

        client = OpenAIChatCompletionClient(
            model=model,
            base_url=base_url,
            api_key=api_key or "unused",
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            },
        )
        return cls(client)

    async def run(self, task: str) -> OrchestrationRun:
        from autogen_agentchat.agents import AssistantAgent  # lazy
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat

        run = OrchestrationRun(task=task, orchestrator=self.name)
        try:
            planner = AssistantAgent(
                "planner",
                model_client=self._model_client,
                system_message="Voce e um planejador. Produza um plano curto e objetivo.",
            )
            executor = AssistantAgent(
                "executor",
                model_client=self._model_client,
                system_message="Voce e um executor. Execute o plano e produza o resultado final.",
            )
            team = RoundRobinGroupChat(
                [planner, executor],
                termination_condition=MaxMessageTermination(self._max_messages),
            )
            result = await team.run(task=task)

            for msg in result.messages:
                source = getattr(msg, "source", "unknown")
                content = getattr(msg, "content", "")
                if source == "user" or not isinstance(content, str):
                    continue
                run.steps.append(Step(agent=source, output=content))

            run.output = run.steps[-1].output if run.steps else None
            run.status = RunStatus.SUCCEEDED
        except Exception as exc:  # noqa: BLE001
            run.status = RunStatus.FAILED
            run.error = str(exc)
        run.finished_at = datetime.now(timezone.utc)
        return run
