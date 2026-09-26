from functools import lru_cache
from ipaddress import ip_address, ip_network
from pathlib import Path
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.security.tokens import is_usable_secret, secret_value


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
        hide_input_in_errors=True,
    )

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    openrouter_api_key: SecretStr | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str | None = Field(default=None, alias="OPENROUTER_BASE_URL")

    default_model: str = Field(
        default="openai/gpt-5.4-mini",
        alias="OPENAI_MODEL",
    )
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
        hide_input_in_errors=True,
    )

    app_name: str = Field(default="Diploma AI Assistant API", alias="APP_NAME")
    app_version: str = Field(default="3.4.0", alias="APP_VERSION")
    environment: str = Field(default="dev", alias="APP_ENV")
    public_domain: str | None = Field(default=None, alias="PUBLIC_DOMAIN")
    public_proxy_ip: str | None = Field(default=None, alias="PUBLIC_PROXY_IP")
    trusted_proxy_cidrs: str = Field(default="", alias="TRUSTED_PROXY_CIDRS")
    public_session_secret: SecretStr | None = Field(
        default=None,
        alias="PUBLIC_SESSION_SECRET",
    )
    public_session_ttl_seconds: int = Field(
        default=30 * 24 * 60 * 60,
        ge=300,
        le=365 * 24 * 60 * 60,
        alias="PUBLIC_SESSION_TTL_SECONDS",
    )

    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    db_pool_min_size: int = Field(default=1, ge=0, alias="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=5, ge=1, alias="DB_POOL_MAX_SIZE")
    llm_cache_enabled: bool = Field(default=True, alias="LLM_CACHE_ENABLED")
    llm_cache_max_entries: int = Field(
        default=256,
        ge=1,
        alias="LLM_CACHE_MAX_ENTRIES",
    )
    llm_cache_ttl_seconds: float = Field(
        default=300.0,
        gt=0,
        alias="LLM_CACHE_TTL_SECONDS",
    )
    agent_checkpointer: Literal["memory", "sqlite", "postgres"] = Field(
        default="sqlite",
        alias="AGENT_CHECKPOINTER",
    )
    agent_sqlite_path: Path = Field(
        default=Path("./agent.db"),
        alias="AGENT_SQLITE_PATH",
    )
    agent_model: str = Field(
        default="openai/gpt-5.4-mini",
        alias="AGENT_MODEL",
    )
    admin_token: SecretStr | None = Field(default=None, alias="ADMIN_TOKEN")
    internal_token: SecretStr | None = Field(default=None, alias="INTERNAL_TOKEN")
    moderation_openai_enabled: bool = Field(default=False, alias="MODERATION_OPENAI_ENABLED")
    chat_max_tokens: int = Field(default=256, ge=1, le=4096, alias="CHAT_MAX_TOKENS")
    public_rate_limit_enabled: bool = Field(
        default=True,
        alias="PUBLIC_RATE_LIMIT_ENABLED",
    )
    public_rate_limit_requests: int = Field(
        default=30,
        ge=1,
        alias="PUBLIC_RATE_LIMIT_REQUESTS",
    )
    public_rate_limit_window_seconds: float = Field(
        default=60.0,
        gt=0,
        alias="PUBLIC_RATE_LIMIT_WINDOW_SECONDS",
    )
    public_max_concurrent_requests: int = Field(
        default=4,
        ge=1,
        alias="PUBLIC_MAX_CONCURRENT_REQUESTS",
    )
    public_generation_enabled: bool = Field(
        default=True,
        alias="PUBLIC_GENERATION_ENABLED",
    )
    public_generation_budget_requests: int = Field(
        default=0,
        ge=0,
        alias="PUBLIC_GENERATION_BUDGET_REQUESTS",
    )
    public_generation_budget_window_seconds: float = Field(
        default=24 * 60 * 60,
        gt=0,
        alias="PUBLIC_GENERATION_BUDGET_WINDOW_SECONDS",
    )
    public_data_retention_days: int = Field(
        default=0,
        ge=0,
        le=3650,
        alias="PUBLIC_DATA_RETENTION_DAYS",
    )
    retention_cleanup_interval_seconds: float = Field(
        default=60 * 60,
        ge=60,
        alias="RETENTION_CLEANUP_INTERVAL_SECONDS",
    )
    document_upload_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1,
        alias="DOCUMENT_UPLOAD_MAX_BYTES",
    )
    chat_media_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1,
        alias="CHAT_MEDIA_MAX_BYTES",
    )
    document_archive_max_entries: int = Field(
        default=1000,
        ge=1,
        alias="DOCUMENT_ARCHIVE_MAX_ENTRIES",
    )
    document_archive_max_uncompressed_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1,
        alias="DOCUMENT_ARCHIVE_MAX_UNCOMPRESSED_BYTES",
    )
    document_archive_max_compression_ratio: float = Field(
        default=100.0,
        ge=1.0,
        alias="DOCUMENT_ARCHIVE_MAX_COMPRESSION_RATIO",
    )
    reindex_max_files: int = Field(
        default=500,
        ge=1,
        alias="REINDEX_MAX_FILES",
    )
    reindex_max_total_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1,
        alias="REINDEX_MAX_TOTAL_BYTES",
    )
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
        default=Path("data"),
        alias="RAG_DATA_DIR",
    )
    rag_collection: str = Field(default="rag_block_03", alias="RAG_COLLECTION")
    rag_production_collection: str = Field(
        default="corporate_rag",
        alias="RAG_PRODUCTION_COLLECTION",
    )
    rag_docstore_path: Path = Field(
        default=Path(".cache/rag/docstore.json"),
        alias="RAG_DOCSTORE_PATH",
    )
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
    rag_chunking_strategy: Literal["fixed", "recursive", "semantic"] = Field(
        default="recursive",
        alias="RAG_CHUNKING_STRATEGY",
    )
    rag_retrieval_top_k: int = Field(default=5, ge=1, alias="RAG_RETRIEVAL_TOP_K")
    rag_reranker_enabled: bool = Field(default=False, alias="RAG_RERANKER_ENABLED")
    rag_reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        alias="RAG_RERANKER_MODEL",
    )
    rag_rerank_top_n: int = Field(default=5, ge=1, alias="RAG_RERANK_TOP_N")
    rag_condense_enabled: bool = Field(default=True, alias="RAG_CONDENSE_ENABLED")
    rag_max_sources: int = Field(default=5, ge=1, le=10, alias="RAG_MAX_SOURCES")
    rag_generation_model: str = Field(
        default="openai/gpt-5.4-mini",
        alias="RAG_GENERATION_MODEL",
    )
    rag_max_tokens: int = Field(default=256, ge=1, alias="RAG_MAX_TOKENS")

    eval_judge_provider: Literal["openai", "local"] = Field(
        default="local",
        alias="EVAL_JUDGE_PROVIDER",
    )
    eval_judge_model: str = Field(
        default="local-qwen-judge",
        alias="EVAL_JUDGE_MODEL",
    )
    eval_judge_base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        alias="EVAL_JUDGE_BASE_URL",
    )
    eval_judge_api_key: SecretStr = Field(
        default=SecretStr("lm-studio"),
        alias="EVAL_JUDGE_API_KEY",
    )
    eval_generation_model: str = Field(
        default="local-qwen-judge",
        alias="EVAL_GENERATION_MODEL",
    )
    eval_embedding_model: str = Field(
        default="intfloat/multilingual-e5-base",
        alias="EVAL_EMBEDDING_MODEL",
    )
    eval_request_timeout: float = Field(
        default=300.0,
        gt=0,
        alias="EVAL_REQUEST_TIMEOUT",
    )
    eval_judge_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=8192,
        alias="EVAL_JUDGE_MAX_TOKENS",
    )
    eval_golden_path: Path = Field(
        default=Path("tests/eval/golden_dataset.json"),
        alias="EVAL_GOLDEN_PATH",
    )
    eval_results_dir: Path = Field(
        default=Path("tests/eval/results"),
        alias="EVAL_RESULTS_DIR",
    )
    eval_concurrency: int = Field(default=3, ge=1, le=10, alias="EVAL_CONCURRENCY")
    eval_faithfulness_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="EVAL_FAITHFULNESS_THRESHOLD",
    )
    eval_answer_relevancy_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="EVAL_ANSWER_RELEVANCY_THRESHOLD",
    )
    eval_citation_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        alias="EVAL_CITATION_THRESHOLD",
    )
    rag_tracing_enabled: bool = Field(default=False, alias="RAG_TRACING_ENABLED")
    tracing_capture_content: bool = Field(
        default=True,
        alias="TRACING_CAPTURE_CONTENT",
    )
    log_prompt_preview_enabled: bool = Field(
        default=True,
        alias="LOG_PROMPT_PREVIEW_ENABLED",
    )

    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        alias="CORS_ORIGINS",
    )

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

    @property
    def trusted_proxy_networks(self) -> tuple[object, ...]:
        """Return validated networks used only for direct reverse-proxy peers."""

        return tuple(
            ip_network(value.strip(), strict=False)
            for value in self.trusted_proxy_cidrs.split(",")
            if value.strip()
        )

    def model_post_init(self, __context: object) -> None:
        if self.db_pool_min_size > self.db_pool_max_size:
            raise ValueError("DB_POOL_MIN_SIZE must not exceed DB_POOL_MAX_SIZE")
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.environment.lower() in {"prod", "production", "public"}:
            if (
                self.public_generation_enabled
                and self.public_generation_budget_requests <= 0
            ):
                raise ValueError(
                    "PUBLIC_GENERATION_BUDGET_REQUESTS must be positive "
                    "when public generation is enabled"
                )
            if self.public_data_retention_days <= 0:
                raise ValueError(
                    "PUBLIC_DATA_RETENTION_DAYS must be positive in public deployment"
                )
            if self.tracing_capture_content:
                raise ValueError(
                    "TRACING_CAPTURE_CONTENT must be false in public deployment"
                )
            if self.log_prompt_preview_enabled:
                raise ValueError(
                    "LOG_PROMPT_PREVIEW_ENABLED must be false in public deployment"
                )
            if not self.cors_origins:
                raise ValueError(
                    "CORS_ORIGINS must contain at least one HTTPS origin"
                )
            for origin in self.cors_origins:
                value = origin.strip()
                parsed = urlsplit(value)
                if value == "*":
                    raise ValueError("CORS_ORIGINS must not contain wildcard origins")
                if (
                    parsed.scheme != "https"
                    or not parsed.hostname
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.query
                    or parsed.fragment
                    or parsed.path != ""
                    or parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
                ):
                    raise ValueError(
                        "CORS_ORIGINS must contain only explicit HTTPS origins"
                    )
            missing = [
                name
                for name, value in (
                    ("ADMIN_TOKEN", self.admin_token),
                    ("INTERNAL_TOKEN", self.internal_token),
                    ("PUBLIC_SESSION_SECRET", self.public_session_secret),
                )
                if not is_usable_secret(value)
            ]
            if missing:
                joined = ", ".join(missing)
                raise ValueError(
                    f"Public deployment requires non-placeholder secrets: {joined}"
                )
            if len(secret_value(self.public_session_secret)) < 32:
                raise ValueError(
                    "PUBLIC_SESSION_SECRET must contain at least 32 characters"
                )

            domain = (self.public_domain or "").strip().lower()
            if (
                not domain
                or domain.startswith("change-me")
                or "://" in domain
                or "/" in domain
                or not re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", domain)
            ):
                raise ValueError(
                    "Public deployment requires a valid PUBLIC_DOMAIN host name"
                )

            provider_key = self.llm.openai_api_key or self.llm.openrouter_api_key
            required_secrets = [
                name
                for name, value in (
                    ("OPENAI_API_KEY or OPENROUTER_API_KEY", provider_key),
                    ("QDRANT_API_KEY", self.qdrant_api_key),
                )
                if not is_usable_secret(value)
            ]
            if required_secrets:
                joined = ", ".join(required_secrets)
                raise ValueError(
                    f"Public deployment requires non-placeholder credentials: {joined}"
                )

            try:
                proxy_networks = self.trusted_proxy_networks
            except ValueError as exc:
                raise ValueError(
                    "TRUSTED_PROXY_CIDRS must contain valid IP networks"
                ) from exc
            if not proxy_networks:
                raise ValueError(
                    "Public deployment requires TRUSTED_PROXY_CIDRS"
                )
            if any(network.prefixlen == 0 for network in proxy_networks):
                raise ValueError(
                    "TRUSTED_PROXY_CIDRS must not trust the entire Internet"
                )
            try:
                proxy_address = ip_address((self.public_proxy_ip or "").strip())
            except ValueError as exc:
                raise ValueError(
                    "Public deployment requires a valid PUBLIC_PROXY_IP"
                ) from exc
            if not any(proxy_address in network for network in proxy_networks):
                raise ValueError(
                    "PUBLIC_PROXY_IP must be included in TRUSTED_PROXY_CIDRS"
                )

@lru_cache
def get_settings() -> Settings:
    return Settings()
