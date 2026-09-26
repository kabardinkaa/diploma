from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


UserRole = Literal["read-only", "write-with-approve", "full"]


class AgentMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=8000)


class AgentInput(BaseModel):
    messages: list[AgentMessage] = Field(min_length=1)


class AgentStreamRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "thread_id": "demo-vpn-question",
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Найди инструкцию по подключению VPN.",
                            }
                        ]
                    },
                    "user_role": "read-only",
                },
                {
                    "thread_id": "admin-confirmation-demo",
                    "resume": True,
                    "user_role": "write-with-approve",
                },
            ]
        }
    )

    thread_id: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Client conversation label. It is not an owner credential: public "
            "checkpoint ownership is isolated by a server-issued session identity."
        ),
    )
    input: AgentInput | None = None
    resume: bool | None = None
    user_role: UserRole = Field(
        default="read-only",
        description=(
            "Deprecated client hint. Public requests are always `read-only`; only "
            "a valid `X-Admin-Token` grants `write-with-approve`."
        ),
        json_schema_extra={"deprecated": True},
    )

    @model_validator(mode="after")
    def validate_invocation(self) -> "AgentStreamRequest":
        supplied = int(self.input is not None) + int(self.resume is not None)
        if supplied != 1:
            raise ValueError("exactly one of input or resume must be provided")
        return self
