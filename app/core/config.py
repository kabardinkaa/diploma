from functools import lru_cache

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

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()