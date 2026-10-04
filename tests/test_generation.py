import pytest
from pydantic import ValidationError

from technical_knowledge_assistant.config import Settings


def test_ollama_configuration_requires_only_a_model() -> None:
    settings = Settings(_env_file=None, llm_provider="ollama", llm_model="qwen2.5:7b")

    assert settings.ollama_base_url == "http://localhost:11434/v1"


def test_generation_provider_requires_a_model() -> None:
    with pytest.raises(ValidationError, match="LLM_MODEL"):
        Settings(_env_file=None, llm_provider="ollama")


def test_openai_configuration_requires_an_api_key() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(
            _env_file=None,
            llm_provider="openai",
            llm_model="gpt-4.1-mini",
            openai_api_key="",
        )