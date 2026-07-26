"""Adaptive + Corrective RAG em LangGraph 1.2.

Topologia:

                     ┌──────────► direct ──────────────────────────┐
    START ─► route ──┼──────────► tool (interrupt HITL) ───────────┤
                     └─► retrieve ─► rerank ─┬─► generate ─► verify┤
                            ▲                │                 │   │
                            │                ├─► grade_one×N ──┤   │
                            │                │      (Send)     │   │
                            └──── rewrite ◄──┴─────────────────┘   ▼
                                                                  END

Decisões que valem mais que o diagrama:

* **O gate barato vem antes do gate caro.** O reranker já produz um score
  calibrado; se o topo passa do piso, vai direto para `generate` e não se
  gasta uma chamada de LLM por documento. O grading com LLM só existe para
  a faixa duvidosa. Muita implementação de "corrective RAG" chama o LLM
  N vezes por query sem necessidade.

* **Roteamento e veredito usam `choice`, não JSON.** Decodificação restrita
  a um conjunto de rótulos custa ~1 token e não pode falhar no parse.

* **Toda política de falha é do runtime, não do código.** RetryPolicy,
  TimeoutPolicy e error_handler ficam no `add_node`. Nenhum `try/except` +
  `for attempt in range(3)` espalhado pelos nós.

* **DI por closure com composition root.** Os nós fecham sobre `Deps`; o
  teste injeta um FakeLLM sem tocar em variável global. (A alternativa é
  `StateGraph(..., context_schema=Deps)` + `Runtime[Deps]` nos nós — use
  essa se precisar dos nós em escopo de módulo.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from langgraph.config import get_stream_writer
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RetryPolicy, Send, TimeoutPolicy, interrupt

from rag_graph.encoders import TritonEncoders
from rag_graph.llm import LLMError, VLLMClient
from rag_graph.settings import Settings
from rag_graph.state import RagState
from rag_graph.store import Doc, VectorStore


@dataclass(frozen=True, slots=True)
class Deps:
    llm: VLLMClient
    encoders: TritonEncoders
    store: VectorStore
    cfg: Settings


# --------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------- #
_ROUTE_SYS: Final = (
    "Você classifica a pergunta do usuário em exatamente um destino.\n"
    "vectorstore: precisa de documentos internos da empresa.\n"
    "tool: precisa de um número vivo de sistema transacional (saldo, posição, status).\n"
    "direct: conversa, definição genérica ou reformulação do que já foi dito."
)

_ANSWER_SYS: Final = (
    "Responda EXCLUSIVAMENTE com base no CONTEXTO. Cite as fontes no formato [n] "
    "referenciando o número do trecho. Se o contexto não contiver a resposta, diga "
    "isso explicitamente — não complete com conhecimento próprio."
)

_GRADE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "relevante": {"type": "boolean"},
        "motivo": {"type": "string", "maxLength": 160},
    },
    "required": ["relevante", "motivo"],
    "additionalProperties": False,
}


def _render_context(docs: list[Doc]) -> str:
    return "\n\n".join(
        f"[{i}] (fonte={d['meta'].get('uri', d['doc_id'])})\n{d['content']}"
        for i, d in enumerate(docs, 1)
    )


# --------------------------------------------------------------------- #
# construção
# --------------------------------------------------------------------- #
def build_graph(deps: Deps, *, checkpointer: Any = None) -> CompiledStateGraph:
    cfg = deps.cfg

    # ---------------------------- roteamento -------------------------- #
    async def route(state: RagState) -> Command:
        dest = await deps.llm.classify(
            [
                {"role": "system", "content": _ROUTE_SYS},
                {"role": "user", "content": state["question"]},
            ],
            labels=("vectorstore", "tool", "direct"),
        )
        goto = {"vectorstore": "retrieve", "tool": "tool", "direct": "direct"}.get(
            dest, "retrieve"
        )
        # Command = update + goto num só retorno. Evita um nó que só decide
        # e um conditional_edge que só relê o que o nó acabou de escrever.
        return Command(
            goto=goto,
            update={
                "route": dest,
                "query": state["question"],
                "rewrites": state.get("rewrites", 0),
                "trail": [f"route={dest}"],
            },
        )

    # ---------------------------- retrieval --------------------------- #
    async def retrieve(state: RagState) -> RagState:
        query = state["query"]
        vec = await deps.encoders.embed([query], query=True)
        docs = await deps.store.hybrid_search(
            vec[0],
            query,
            state["tenant_id"],
            per_leg=cfg.candidates_per_leg,
            rrf_k=cfg.rrf_k,
            w_dense=cfg.rrf_weight_dense,
            w_sparse=cfg.rrf_weight_sparse,
            limit=cfg.candidates_per_leg,
        )
        return {"candidates": docs, "trail": [f"retrieve={len(docs)}"]}

    async def rerank(state: RagState) -> RagState:
        cands = state["candidates"]
        if not cands:
            return {"context": [], "trail": ["rerank=0"]}

        scores = await deps.encoders.rerank(state["query"], [d["content"] for d in cands])
        # argsort em C sobre N floats; nada de sorted(key=lambda ...)
        order = np.argsort(-scores, kind="stable")[: cfg.rerank_keep]
        top = [cands[i] | {"score": float(scores[i])} for i in order]
        return {"context": top, "trail": [f"rerank top1={top[0]['score']:.3f}"]}

    # --------------------- gate: barato antes do caro ----------------- #
    def after_rerank(state: RagState) -> str | list[Send]:
        ctx = state.get("context") or []
        rewrites = state.get("rewrites", 0)
        budget = rewrites < cfg.max_rewrites

        if ctx and ctx[0]["score"] >= cfg.rerank_floor:
            return "generate"  # caminho quente: zero LLM extra
        if not ctx:
            return "rewrite" if budget else "generate"
        if not budget:
            return "generate"  # sem orçamento: responde com o que há e deixa o verify julgar
        # faixa duvidosa: fan-out de grading, um Send por documento.
        # As N tarefas rodam concorrentes no mesmo superstep.
        return [Send("grade_one", {"query": state["query"], "doc": d}) for d in ctx]

    async def grade_one(task: dict[str, Any]) -> dict[str, list[Doc]]:
        doc: Doc = task["doc"]
        verdict = await deps.llm.complete_json(
            [
                {"role": "system", "content": "Julgue se o TRECHO ajuda a responder a PERGUNTA."},
                {
                    "role": "user",
                    "content": f"PERGUNTA: {task['query']}\n\nTRECHO:\n{doc['content'][:2000]}",
                },
            ],
            schema=_GRADE_SCHEMA,
            max_tokens=96,
        )
        # o reducer operator.add em `graded` funde os N retornos
        return {"graded": [doc] if verdict["relevante"] else []}

    async def collect(state: RagState) -> RagState:
        kept = state.get("graded") or []
        return {"context": kept, "trail": [f"graded={len(kept)}"]}

    def after_collect(state: RagState) -> str:
        if state.get("context"):
            return "generate"
        return "rewrite" if state.get("rewrites", 0) < cfg.max_rewrites else "generate"

    # ---------------------------- reescrita --------------------------- #
    async def rewrite(state: RagState) -> RagState:
        new_q = await deps.llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Reescreva a pergunta para busca documental: expanda siglas, "
                        "acrescente sinônimos do domínio, remova pronomes. "
                        "Responda só com a nova pergunta."
                    ),
                },
                {"role": "user", "content": state["question"]},
            ],
            max_tokens=128,
        )
        return {
            "query": new_q.strip(),
            "rewrites": state.get("rewrites", 0) + 1,
            "graded": [],  # zera o acumulador antes da próxima volta
            "trail": [f"rewrite#{state.get('rewrites', 0) + 1}"],
        }

    # ---------------------------- geração ----------------------------- #
    async def generate(state: RagState) -> RagState:
        ctx = state.get("context") or []
        writer = get_stream_writer()  # canal `custom` do stream

        messages = [
            {"role": "system", "content": _ANSWER_SYS},
            {
                "role": "user",
                "content": f"CONTEXTO:\n{_render_context(ctx)}\n\nPERGUNTA: {state['question']}",
            },
        ]
        parts: list[str] = []
        async for piece in deps.llm.stream(messages, temperature=0.2):
            parts.append(piece)
            # Sem langchain-openai não existe stream_mode="messages": o token
            # sobe pelo writer e sai como evento `custom`.
            writer({"type": "token", "text": piece})

        return {
            "answer": "".join(parts),
            "citations": [d["meta"].get("uri", d["doc_id"]) for d in ctx],
            "trail": ["generate"],
        }

    async def verify(state: RagState) -> Command:
        """Groundedness gate. Barato: 1 token restrito a 3 rótulos."""
        ctx = state.get("context") or []
        if not ctx:
            return Command(goto=END, update={"verdict": "ungrounded"})

        verdict = await deps.llm.classify(
            [
                {
                    "role": "system",
                    "content": (
                        "A RESPOSTA é sustentada pelo CONTEXTO? "
                        "grounded = tudo sustentado; ungrounded = há afirmação sem "
                        "respaldo; incomplete = sustentada mas não responde à pergunta."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXTO:\n{_render_context(ctx)}\n\n"
                        f"PERGUNTA: {state['question']}\n\nRESPOSTA:\n{state['answer']}"
                    ),
                },
            ],
            labels=("grounded", "ungrounded", "incomplete"),
        )
        if verdict == "grounded" or state.get("rewrites", 0) >= cfg.max_rewrites:
            return Command(goto=END, update={"verdict": verdict, "trail": [f"verify={verdict}"]})
        return Command(goto="rewrite", update={"verdict": verdict, "trail": [f"verify={verdict}"]})

    # ------------------------- ramos alternativos --------------------- #
    async def direct(state: RagState) -> RagState:
        writer = get_stream_writer()
        parts: list[str] = []
        async for piece in deps.llm.stream(
            [{"role": "user", "content": state["question"]}], temperature=0.5
        ):
            parts.append(piece)
            writer({"type": "token", "text": piece})
        return {"answer": "".join(parts), "verdict": "grounded", "trail": ["direct"]}

    async def tool(state: RagState) -> RagState:
        """Ferramenta transacional com aprovação humana.

        `interrupt` persiste o checkpoint e devolve o controle ao chamador.
        O processo pode morrer aqui; a retomada com
        `Command(resume=...)` no mesmo thread_id continua exatamente deste ponto.
        """
        approval = interrupt(
            {
                "acao": "consulta_saldo",
                "pergunta": state["question"],
                "prompt": "Aprovar consulta ao core bancário?",
            }
        )
        if not approval.get("aprovado"):
            return {"answer": "Consulta não autorizada pelo operador.", "trail": ["tool=negado"]}
        # ... chamada gRPC ao sistema transacional ...
        return {"answer": "Saldo consultado: R$ …", "trail": ["tool=ok"]}

    # ------------------------- compensação (Saga) --------------------- #
    def on_retrieval_failure(state: RagState, error: NodeError) -> Command:
        """Roda depois que o RetryPolicy esgotou. Degrada, não derruba."""
        return Command(
            goto="generate",
            update={
                "context": [],
                "failure": f"retrieval indisponível: {error}",
                "trail": ["compensate=retrieve"],
            },
        )

    # ------------------------------ montagem -------------------------- #
    b = StateGraph(RagState)

    io_retry = RetryPolicy(max_attempts=3, initial_interval=0.25, backoff_factor=2.0)
    net_retry = RetryPolicy(max_attempts=3, retry_on=(LLMError, ConnectionError, TimeoutError))

    b.add_node("route", route, retry_policy=net_retry, timeout=TimeoutPolicy(run_timeout=15.0))
    b.add_node(
        "retrieve",
        retrieve,
        retry_policy=io_retry,
        timeout=TimeoutPolicy(run_timeout=8.0),
        error_handler=on_retrieval_failure,
    )
    b.add_node("rerank", rerank, retry_policy=io_retry, timeout=TimeoutPolicy(run_timeout=10.0))
    b.add_node("grade_one", grade_one, retry_policy=net_retry)
    b.add_node("collect", collect)
    b.add_node("rewrite", rewrite, retry_policy=net_retry)
    # idle_timeout reseta a cada token: uma geração longa é legítima,
    # um socket travado sem progresso não é.
    b.add_node(
        "generate",
        generate,
        retry_policy=net_retry,
        timeout=TimeoutPolicy(run_timeout=180.0, idle_timeout=20.0),
    )
    b.add_node("verify", verify, retry_policy=net_retry)
    b.add_node("direct", direct, timeout=TimeoutPolicy(run_timeout=120.0, idle_timeout=20.0))
    b.add_node("tool", tool)

    b.add_edge(START, "route")
    # route/verify usam Command(goto=...) → não precisam de add_edge de saída
    b.add_edge("retrieve", "rerank")
    b.add_conditional_edges("rerank", after_rerank, ["generate", "rewrite", "grade_one"])
    b.add_edge("grade_one", "collect")
    b.add_conditional_edges("collect", after_collect, ["generate", "rewrite"])
    b.add_edge("rewrite", "retrieve")
    b.add_edge("generate", "verify")
    b.add_edge("direct", END)
    b.add_edge("tool", END)

    return b.compile(checkpointer=checkpointer)
