from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """
    Настройки Telegram-бота.

    BOT_TOKEN хранится только в .env.
    .env не коммитится в Git.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str = Field(default="", alias="BOT_TOKEN")
    backend_url: str = Field(default="http://127.0.0.1:8000", alias="BACKEND_URL")
    admin_token: str = Field(default="", alias="ADMIN_TOKEN")
    internal_token: str = Field(default="", alias="INTERNAL_TOKEN")
    bot_admin_ids: list[int] = Field(default_factory=list, alias="BOT_ADMIN_IDS")
    rate_limit_requests: int = Field(default=10, ge=1, alias="BOT_RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: float = Field(
        default=60.0,
        gt=0,
        alias="BOT_RATE_LIMIT_WINDOW_SECONDS",
    )
    daily_quota: int = Field(default=100, ge=1, alias="BOT_DAILY_QUOTA")
    quota_max_users: int = Field(
        default=10_000,
        ge=1,
        alias="BOT_QUOTA_MAX_USERS",
    )
    quota_state_ttl_seconds: float = Field(
        default=2 * 24 * 60 * 60,
        gt=0,
        alias="BOT_QUOTA_STATE_TTL_SECONDS",
    )
    chat_cache_max_entries: int = Field(
        default=1000,
        ge=1,
        alias="BOT_CHAT_CACHE_MAX_ENTRIES",
    )
    chat_cache_ttl_seconds: float = Field(
        default=24 * 60 * 60,
        gt=0,
        alias="BOT_CHAT_CACHE_TTL_SECONDS",
    )
    photo_max_bytes: int = Field(
        default=2 * 1024 * 1024,
        ge=1,
        alias="BOT_PHOTO_MAX_BYTES",
    )
    media_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1,
        alias="BOT_MEDIA_MAX_BYTES",
    )

    @field_validator("bot_admin_ids", mode="before")
    @classmethod
    def parse_admin_ids(cls, value: Any) -> list[int]:
        if value is None or value == "":
            return []

        if isinstance(value, list):
            return [int(item) for item in value]

        if isinstance(value, int):
            return [value]

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return []

            if value.startswith("["):
                return value

            return [
                int(item.strip())
                for item in value.split(",")
                if item.strip()
            ]

        return value


@lru_cache
def get_bot_settings() -> BotSettings:
    return BotSettings()
