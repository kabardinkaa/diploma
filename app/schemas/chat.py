from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Роль сообщения в диалоге",
    )
    content: str = Field(
    ...,
    min_length=1,
    max_length=4000,
    repr=False,
    description="Текст сообщения. Не выводится в repr, чтобы случайно не светить PII.",
    )


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
                    "max_tokens": 500,
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
                    "model": "openrouter/free",
                    "temperature": 0,
                    "max_tokens": 300,
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
        description="Модель. Если не указана, берётся default_model из настроек",
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
        description="Максимальное число токенов ответа",
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