"""
core/config.py
Centralised settings — loaded once at import time.
All env vars with defaults so the system is runnable without every key set.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Database ---
    database_url: str = "postgresql://postgres:password@localhost:5432/esg_pipeline"

    # --- Tavily ---
    tavily_api_key: str = ""

    # --- LLM ---
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    llm_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    # --- Storage ---
    pdf_storage_path: Path = Path("./storage/pdfs")

    # --- Pipeline ---
    parser_version: str = "1.2.0"       # bumped: spatial chunker enabled
    max_chunk_tokens: int = 500
    min_chunk_tokens: int = 200
    retrieval_top_k: int = 7
    use_spatial_chunker: bool = True     # set False to revert to block-based extraction

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64
    use_embedding_retrieval: bool = True   # set False to fall back to keyword-only

    # --- HTTP ---
    download_timeout_seconds: int = 60
    max_download_size_mb: int = 50

    # --- Logging ---
    log_level: str = "INFO"
    log_file: Path = Path("./logs/pipeline.log")

    def ensure_dirs(self) -> None:
        self.pdf_storage_path.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s