"""Estado do grafo.

Duas regras que evitam a maior parte dos incidentes de produção com LangGraph:

1. Todo campo escrito por mais de um nó em paralelo PRECISA de reducer.
   Sem `Annotated[..., reducer]` o runtime levanta InvalidUpdateError quando
   dois nós concorrentes escrevem a mesma chave — e o fan-out de grading
   escreve `graded` a partir de N tarefas simultâneas.

2. O estado é serializado a cada superstep. Nada de conexão, pool, cliente
   HTTP ou array grande aqui dentro — isso vai no `context`/runtime, que não
   é checkpointado.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from rag_graph.store import Doc

Route = Literal["vectorstore", "tool", "direct"]
Verdict = Literal["grounded", "ungrounded", "incomplete"]


def _last(_old: object, new: object) -> object:
    """Reducer de sobrescrita explícita (o default já é este; documenta a intenção)."""
    return new


class RagState(TypedDict, total=False):
    # entrada
    question: str
    tenant_id: str

    # roteamento e reescrita
    route: Route
    query: str
    rewrites: int

    # retrieval
    candidates: list[Doc]
    graded: Annotated[list[Doc], operator.add]  # escrito em paralelo via Send
    context: list[Doc]

    # geração
    answer: str
    citations: list[str]
    verdict: Verdict

    # observabilidade / compensação
    trail: Annotated[list[str], operator.add]
    failure: str | None
