"""Configuração — papel do `config.rs` nos scaffolds Rust.

Não-segredos vêm sempre do ambiente. Segredos usam SecretStr; num variant
`-vault` o provider trocaria por leitura fail-closed no Vault/OpenBao.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    app_env: Literal["development", "staging", "production"] = "development"

    # --- vLLM (OpenAI-compatible) ----------------------------------------
    llm_base_url: str = "http://192.168.15.201:8000/v1"
    llm_model: str = "Qwen/Qwen3-32B-AWQ"
    llm_api_key: SecretStr = SecretStr("")
    llm_max_conns: int = 64
    # vLLM unificou os antigos `guided_*` em `structured_outputs`.
    # Deixe True só se o cluster ainda roda um build antigo.
    llm_legacy_guided: bool = False

    # --- Triton (encoders) ------------------------------------------------
    triton_url: str = "192.168.15.201:8001"          # gRPC
    embed_model: str = "bge_m3_encoder"
    embed_dim: int = 1024
    embed_max_tokens: int = 512
    rerank_model: str = "bge_reranker_v2_m3"
    tokenizer_path: str = "/models/bge-m3/tokenizer.json"
    rerank_tokenizer_path: str = "/models/bge-reranker-v2-m3/tokenizer.json"

    # --- Postgres (pgvector + checkpointer) -------------------------------
    pg_dsn: SecretStr = SecretStr("postgresql://rag@127.0.0.1:5432/rag")
    pg_pool_min: int = 4
    pg_pool_max: int = 32
    pg_stmt_cache: int = 512
    fts_config: str = "portuguese"

    # --- retrieval --------------------------------------------------------
    candidates_per_leg: int = Field(default=60, ge=10)   # top-k de cada perna
    rrf_k: int = 60                                      # constante do RRF
    rrf_weight_dense: float = 1.0
    rrf_weight_sparse: float = 0.7
    rerank_keep: int = 8                                 # docs que vão ao prompt
    rerank_floor: float = 0.35                           # gate barato, sem LLM
    hnsw_ef_search: int = 120
    max_rewrites: int = 2

    @property
    def dsn(self) -> str:
        return self.pg_dsn.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
