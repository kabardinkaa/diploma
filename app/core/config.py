from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """
    Настройки LLM-провайдера.

    Поддерживаем два варианта:
    - OpenAI напрямую: OPENAI_API_KEY
    - OpenRouter/OpenAI-compatible API: OPENROUTER_API_KEY + OPENROUTER_BASE_URL
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    openrouter_api_key: SecretStr | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str | None = Field(default=None, alias="OPENROUTER_BASE_URL")

    default_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    request_timeout: float = Field(default=60.0, alias="LLM_REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    @property
    def api_key(self) -> SecretStr:
        key = self.openai_api_key or self.openrouter_api_key

        if key is None:
            raise ValueError(
                "Не найден API-ключ. Добавь OPENAI_API_KEY или OPENROUTER_API_KEY в .env"
            )

        return key

    @property
    def base_url(self) -> str | None:
        return self.openai_base_url or self.openrouter_base_url


class Settings(BaseSettings):
    """
    Корневые настройки приложения.

    .env используется для локальной разработки.
    В production значения должны приходить из переменных окружения.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = Field(default="Diploma AI Assistant API", alias="APP_NAME")
    app_version: str = Field(default="3.4.0", alias="APP_VERSION")
    environment: str = Field(default="dev", alias="APP_ENV")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    cache_ttl_seconds: int = Field(default=300, alias="CACHE_TTL_SECONDS")
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    admin_token: SecretStr | None = Field(default=None, alias="ADMIN_TOKEN")
    internal_token: SecretStr | None = Field(default=None, alias="INTERNAL_TOKEN")
    moderation_openai_enabled: bool = Field(default=False, alias="MODERATION_OPENAI_ENABLED")
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-base",
        alias="EMBEDDING_MODEL",
    )
    embedding_batch_size: int = Field(default=32, ge=1, alias="EMBEDDING_BATCH_SIZE")
    embedding_cache_dir: Path = Field(
        default=Path(".cache/embeddings"),
        alias="EMBEDDING_CACHE_DIR",
    )
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="documents", alias="QDRANT_COLLECTION")
    embedding_dim: int = Field(default=768, ge=1, alias="EMBEDDING_DIM")
    rag_data_dir: Path = Field(
        default=Path("data/rag-block-03"),
        alias="RAG_DATA_DIR",
    )
    rag_collection: str = Field(default="rag_block_03", alias="RAG_COLLECTION")
    rag_baremetal_collection: str = Field(
        default="rag_block_03_baremetal",
        alias="RAG_BAREMETAL_COLLECTION",
    )
    rag_chunk_size: int = Field(default=512, ge=32, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=64, ge=0, alias="RAG_CHUNK_OVERLAP")
    rag_similarity_top_k: int = Field(
        default=3,
        ge=1,
        alias="RAG_SIMILARITY_TOP_K",
    )
    rag_min_score: float = Field(default=0.82, ge=0.0, le=1.0, alias="RAG_MIN_SCORE")

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    chat_repository: Literal["json", "postgres"] = Field(
        default="json",
        alias="CHAT_REPOSITORY",
    )
    chat_storage_dir: Path = Field(
        default=Path("./var/chats"),
        alias="CHAT_STORAGE_DIR",
    )
    chat_context_strategy: Literal["sliding", "hybrid"] = Field(
        default="sliding",
        alias="CHAT_CONTEXT_STRATEGY",
    )
    chat_context_window: int = Field(
        default=10,
        alias="CHAT_CONTEXT_WINDOW",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)

    def model_post_init(self, __context: object) -> None:
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

@lru_cache
def get_settings() -> Settings:
    return Settings()
