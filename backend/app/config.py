"""Central configuration. Values come from .env, with sensible defaults."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    corpus_dir: Path = BACKEND_DIR / "data" / "corpus"
    index_dir: Path = BACKEND_DIR / "data" / "index"

    # Embeddings (local, no API key needed)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_seq_length: int = 256
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Generation (real API keys live in .env)
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_fallback_model: str = "openai/gpt-oss-20b"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    llm_temperature: float = 0.2
    llm_max_tokens: int = 700
    llm_timeout: float = 60.0

    # Retrieval
    top_score_threshold: float = 0.4
    default_k: int = 3
    hybrid_candidates: int = 20

    # Server
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    ingest_admin_key: str = "change-me"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
