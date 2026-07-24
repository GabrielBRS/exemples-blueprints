"""Adapter LangGraph: grafo plan -> execute -> review.

Import do framework e LAZY (dentro da classe): o core sobe sem o langgraph
instalado se este adapter nao for selecionado. O LLM entra injetado (LLMClient),
entao o mesmo grafo roda com vLLM, Ollama ou um fake de teste.
"""

from datetime import datetime, timezone
from typing import TypedDict

from app.llm.llm_client import LLMClient
from app.models.orchestration import OrchestrationRun, RunStatus, Step


class _GraphState(TypedDict):
    task: str
    plan: str
    result: str
    review: str


class LangGraphOrchestrator:
    name = "langgraph"

    def __init__(self, llm: LLMClient) -> None:
        from langgraph.graph import END, START, StateGraph  # lazy

        self._llm = llm

        async def plan(state: _GraphState) -> dict:
            out = await self._llm.complete(
                "Voce e um planejador. Produza um plano curto e objetivo.",
                state["task"],
            )
            return {"plan": out}

        async def execute(state: _GraphState) -> dict:
            out = await self._llm.complete(
                "Voce e um executor. Execute o plano e produza o resultado.",
                f"Task: {state['task']}\nPlano: {state['plan']}",
            )
            return {"result": out}

        async def review(state: _GraphState) -> dict:
            out = await self._llm.complete(
                "Voce e um revisor. Avalie e finalize a resposta.",
                f"Task: {state['task']}\nResultado: {state['result']}",
            )
            return {"review": out}

        graph = StateGraph(_GraphState)
        graph.add_node("plan", plan)
        graph.add_node("execute", execute)
        graph.add_node("review", review)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", "review")
        graph.add_edge("review", END)
        self._compiled = graph.compile()

    async def run(self, task: str) -> OrchestrationRun:
        run = OrchestrationRun(task=task, orchestrator=self.name)
        try:
            final: _GraphState = await self._compiled.ainvoke(
                {"task": task, "plan": "", "result": "", "review": ""}
            )
            run.steps = [
                Step(agent="planner", output=final["plan"]),
                Step(agent="executor", output=final["result"]),
                Step(agent="reviewer", output=final["review"]),
            ]
            run.output = final["review"]
            run.status = RunStatus.SUCCEEDED
        except Exception as exc:  # noqa: BLE001 - traduz falha do framework p/ dominio
            run.status = RunStatus.FAILED
            run.error = str(exc)
        run.finished_at = datetime.now(timezone.utc)
        return run
