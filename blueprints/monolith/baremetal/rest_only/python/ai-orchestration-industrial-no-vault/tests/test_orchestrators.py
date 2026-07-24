"""O seam em acao: os MESMOS asserts rodam contra adapters diferentes.

- LangGraph executa o grafo real (plan->execute->review) com um LLM fake.
- AutoGen executa o time real (RoundRobin) com o ReplayChatCompletionClient.
- O fluxo HTTP roda com o EchoOrchestrator.
Zero chamadas de rede, zero chave de API.
"""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import Config
from app.llm.llm_client import FakeLLMClient
from app.main import create_app
from app.models.orchestration import RunStatus
from app.orchestrators.autogen_orchestrator import AutoGenOrchestrator
from app.orchestrators.echo_orchestrator import EchoOrchestrator
from app.orchestrators.langgraph_orchestrator import LangGraphOrchestrator


# ---------- adapters, direto no seam ----------

async def test_echo_orchestrator() -> None:
    run = await EchoOrchestrator().run("ola")
    assert run.status is RunStatus.SUCCEEDED
    assert run.output == "ola"
    assert run.steps[0].agent == "echo"


async def test_langgraph_adapter_runs_real_graph_with_fake_llm() -> None:
    llm = FakeLLMClient(canned=["PLANO: 3 passos", "RESULTADO: feito", "REVISAO: aprovado"])
    run = await LangGraphOrchestrator(llm).run("organizar minha semana")

    assert run.status is RunStatus.SUCCEEDED
    assert [s.agent for s in run.steps] == ["planner", "executor", "reviewer"]
    assert run.steps[0].output == "PLANO: 3 passos"
    assert run.output == "REVISAO: aprovado"


async def test_autogen_adapter_runs_real_team_with_replay_client() -> None:
    from autogen_ext.models.replay import ReplayChatCompletionClient

    model_client = ReplayChatCompletionClient(
        ["plano: pesquisar e resumir", "resultado final: resumo pronto"]
    )
    run = await AutoGenOrchestrator(model_client, max_messages=3).run("resumir um artigo")

    assert run.status is RunStatus.SUCCEEDED
    agents = [s.agent for s in run.steps]
    assert "planner" in agents and "executor" in agents
    assert run.output == "resultado final: resumo pronto"


# ---------- o mesmo contrato, qualquer adapter (a prova da agnosticidade) ----------

async def test_any_adapter_satisfies_same_contract() -> None:
    adapters = [
        EchoOrchestrator(),
        LangGraphOrchestrator(FakeLLMClient()),
    ]
    for adapter in adapters:
        run = await adapter.run("mesma task")
        assert run.status is RunStatus.SUCCEEDED
        assert run.output
        assert run.finished_at is not None
        assert run.orchestrator == adapter.name


# ---------- fluxo HTTP completo (app inteiro, echo) ----------

def _cfg() -> Config:
    return Config(ORCHESTRATOR="echo", _env_file=None)


def test_http_flow() -> None:
    app = create_app(config=_cfg(), orchestrator_override=EchoOrchestrator())
    with TestClient(app) as client:
        assert client.get("/health/live").json()["status"] == "alive"
        assert client.get("/health/ready").status_code == 200

        # validacao pydantic: task vazia -> 422
        assert client.post("/api/v1/orchestrations/", json={"task": ""}).status_code == 422

        r = client.post("/api/v1/orchestrations/", json={"task": "diga oi"})
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "succeeded" and body["output"] == "diga oi"
        assert "x-request-id" in r.headers

        rid = body["id"]
        assert client.get(f"/api/v1/orchestrations/{rid}").json()["task"] == "diga oi"
        assert len(client.get("/api/v1/orchestrations/").json()) == 1
        assert client.get(f"/api/v1/orchestrations/{uuid4()}").status_code == 404
