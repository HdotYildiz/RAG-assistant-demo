"""Runtime configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with safe defaults for local development."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "none"
    llm_model: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    index_directory: Path = Path(".rag-index")
    corpus_dataset: str = "freshstack/corpus-oct-2024"
    corpus_subset: str = "langchain"
    corpus_split: str = "train"
    retrieval_candidates: int = 40
    retrieval_top_k: int = 8


settings = Settings()