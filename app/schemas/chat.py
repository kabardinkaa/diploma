from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Роль сообщения",
    )

    content: str | list[dict[str, Any]] = Field(
        ...,
        repr=False,
        description="Текст сообщения или список мультимодальных content-part",
    )

    @field_validator("content")
    @classmethod
    def validate_content(
        cls,
        value: str | list[dict[str, Any]],
    ) -> str | list[dict[str, Any]]:
        if isinstance(value, str):
            if not value:
                raise ValueError("content must not be empty")

            if len(value) > 4000:
                raise ValueError("content must not exceed 4000 characters")

        elif not value:
            raise ValueError("multimodal content must not be empty")

        return value


class Usage(BaseModel):
    prompt_tokens: int | None = Field(default=None)
    completion_tokens: int | None = Field(default=None)
    total_tokens: int | None = Field(default=None)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Привет! Помоги оформить заявку на доступ к VPN.",
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 256,
                    "user_id": "employee-123",
                    "session_id": "session-001",
                },
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Ты — ИИ-ассистент внутренней техподдержки.",
                        },
                        {
                            "role": "user",
                            "content": "Кратко объясни, что делать, если не работает корпоративная почта.",
                        },
                    ],
                    "temperature": 0,
                    "max_tokens": 256,
                },
            ]
        }
    )

    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="История сообщений для LLM",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Deprecated compatibility field. The server always uses its configured "
            "model; a client value does not change model selection."
        ),
        json_schema_extra={"deprecated": True},
    )
    temperature: float = Field(
        default=0.2,
        ge=0,
        le=2,
        description="Температура генерации",
    )
    max_tokens: int = Field(
        default=500,
        ge=1,
        le=16000,
        description=(
            "Deprecated compatibility hint. The server replaces it with the "
            "configured `CHAT_MAX_TOKENS` limit (256 by default)."
        ),
        json_schema_extra={
            "deprecated": True,
            "default": 256,
            "maximum": 4096,
        },
    )
    user_id: str | None = Field(
        default=None,
        description="ID пользователя для логов и будущего rate-limit",
    )
    session_id: str | None = Field(
        default=None,
        description="ID сессии для мультиходового диалога",
    )


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Для подключения откройте корпоративный VPN-клиент.",
                "model": "openai/gpt-5.4-mini",
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 18,
                    "total_tokens": 60,
                },
                "finish_reason": "stop",
                "cached": False,
            }
        }
    )

    content: str
    model: str
    usage: Usage | None = None
    finish_reason: str | None = None
    cached: bool = False

    @classmethod
    def from_openai(cls, response, cached: bool = False) -> "ChatResponse":
        choice = response.choices[0]
        message = choice.message

        usage = None
        if response.usage is not None:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return cls(
            content=message.content or "",
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason,
            cached=cached,
        )


class ChatDelta(BaseModel):
    content: str | None = None
    usage: Usage | None = None
