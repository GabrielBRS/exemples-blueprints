"""Configuracao centralizada. O ORCHESTRATOR escolhe qual adapter o composition
root monta; o bloco LLM_* configura o endpoint OpenAI-compatible (vLLM/Ollama/...)."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    orchestrator: str = Field(default="echo", alias="ORCHESTRATOR")

    llm_base_url: str = Field(default="http://localhost:11434/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="llama3.1", alias="LLM_MODEL")
    llm_api_key: SecretStr = Field(default=SecretStr(""), alias="LLM_API_KEY")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8080, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def load_config() -> Config:
    return Config()
