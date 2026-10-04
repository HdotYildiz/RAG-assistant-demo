"""Runtime configuration loaded from environment variables."""

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with safe defaults for local development."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "none"
    llm_model: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    index_directory: Path = Path(".rag-index")
    corpus_dataset: str = "freshstack/corpus-oct-2024"
    corpus_subset: str = "langchain"
    corpus_split: str = "train"
    retrieval_candidates: int = 40
    retrieval_top_k: int = 8
    min_evidence_term_overlap: int = 2
    max_context_characters: int = 12_000
    max_chunk_characters: int = 4_000
    generation_max_tokens: int = 512
    query_dataset: str = "freshstack/queries-oct-2024"
    query_subset: str = "langchain"
    query_split: str = "test"
    evaluation_directory: Path = Path("results")
    evaluation_seed: int = 42
    development_fraction: float = 0.7

    @model_validator(mode="after")
    def validate_generation_provider(self) -> "Settings":
        """Require configuration needed by the selected generation provider."""
        if self.llm_provider not in {"none", "ollama", "openai"}:
            raise ValueError("LLM_PROVIDER must be 'none', 'ollama', or 'openai'.")
        if self.llm_provider in {"ollama", "openai"} and not self.llm_model:
            raise ValueError("LLM_MODEL is required when LLM_PROVIDER enables generation.")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM_PROVIDER=openai.")
        if self.min_evidence_term_overlap < 1:
            raise ValueError("MIN_EVIDENCE_TERM_OVERLAP must be positive.")
        if self.max_context_characters < self.max_chunk_characters:
            raise ValueError("MAX_CONTEXT_CHARACTERS must be at least MAX_CHUNK_CHARACTERS.")
        if self.generation_max_tokens < 1:
            raise ValueError("GENERATION_MAX_TOKENS must be positive.")
        return self


settings = Settings()