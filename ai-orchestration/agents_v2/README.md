# rag-graph — Adaptive/Corrective RAG on-premise com LangGraph 1.2

Referência completa de um serviço de RAG agêntico rodando **inteiramente em
infraestrutura própria**: vLLM + Triton + Postgres/pgvector em k3s, com o
Python atuando só como *control plane*.

## Onde o trabalho realmente acontece

| Etapa | Executor | Linguagem do hot path |
|---|---|---|
| Tokenização | `tokenizers` (HF) | Rust, libera a GIL |
| Embedding / rerank | Triton + TensorRT | CUDA |
| Busca densa + lexical + fusão RRF | Postgres/pgvector | C |
| Geração | vLLM (paged attention, prefix cache) | CUDA |
| Serialização | orjson | C |
| Servidor HTTP | Granian | Rust |
| **Orquestração** | **LangGraph** | **Python — e só isso** |

Nenhum laço Python percorre tokens, vetores ou linhas de resultado. Os laços
que existem percorram *documentos* e *fronteiras de chunk* — dezenas de
iterações, não milhões.

## Topologia do grafo

```
                     ┌────────► direct ─────────────────────────┐
  START ─► route ────┼────────► tool (interrupt HITL) ──────────┤
                     └─► retrieve ─► rerank ─┬─► generate ─► verify
                            ▲                │            │     │
                            │                ├─► grade_one×N    │
                            │                │      (Send)      │
                            └──── rewrite ◄──┴──────────────────┘
```

* **route** — `structured_outputs.choice` restringe a saída a três rótulos.
  Custa ~1 token decodificado e não pode falhar no parse.
* **retrieve** — uma query SQL: KNN em `halfvec` + full-text em `tsvector`,
  fundidos por Reciprocal Rank Fusion dentro do Postgres.
* **rerank** — cross-encoder no Triton, um batch, `argsort` em C.
* **gate barato antes do gate caro** — se o top-1 do reranker passa do piso,
  vai direto gerar. O grading com LLM (fan-out por `Send`) só roda na faixa
  duvidosa. É o que separa um corrective RAG viável de um que gasta N
  chamadas de LLM por pergunta.
* **verify** — groundedness em 1 token; se falhar e houver orçamento,
  volta para `rewrite`.

## Confiabilidade é configuração, não código

```python
b.add_node("retrieve", retrieve,
           retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.25),
           timeout=TimeoutPolicy(run_timeout=8.0),
           error_handler=on_retrieval_failure)   # Saga/compensação
```

`idle_timeout` no nó `generate` reseta a cada token: uma geração longa é
legítima, um socket travado sem progresso não é.

## Executando

```bash
uv sync
psql "$RAG_PG_DSN" -f sql/001_schema.sql

# ingestão
uv run python -c "
import asyncio, pathlib
from rag_graph import ingest as ing
...
"

# serviço
uv run granian --interface asgi --loop uvloop rag_graph.server:app
```

```bash
curl -N -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Qual o prazo de retenção de logs no contrato X?",
       "tenant_id":"acme","thread_id":"t-001"}'
```

## Ajustes que valem mais que trocar de modelo

| Sintoma | Ajuste |
|---|---|
| Recall baixo em siglas/códigos | Aumente `rrf_weight_sparse` — o denso não acerta identificador |
| Resposta correta, citação errada | O problema é chunking, não o LLM: reduza `max_tokens` do chunk |
| p99 alto com carga | `--enable-chunked-prefill` no vLLM + `max_num_seqs` menor |
| Recall cai com filtro por tenant | `hnsw.iterative_scan = relaxed_order` e suba o `ef_search` |
| Checkpoint crescendo sem limite | `DeltaChannel` (1.2, beta) com `snapshot_frequency` |

## Notas de versão

Escrito contra `langgraph` 1.2.x (mai/2026). APIs usadas que são recentes:
`TimeoutPolicy` e `error_handler` no `add_node`, `RunControl`/`GraphDrained`,
`version="v2"` em `astream`, e `structured_outputs` no vLLM (substituiu os
antigos `guided_*`). Confirme o caminho de import de `DeltaChannel` na versão
instalada — segue em beta.
