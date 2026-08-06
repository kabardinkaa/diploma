from typing import Literal

from pydantic import BaseModel, Field, model_validator


UserRole = Literal["read-only", "write-with-approve", "full"]


class AgentMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=8000)


class AgentInput(BaseModel):
    messages: list[AgentMessage] = Field(min_length=1)


class AgentStreamRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    input: AgentInput | None = None
    resume: bool | None = None
    user_role: UserRole = "write-with-approve"

    @model_validator(mode="after")
    def validate_invocation(self) -> "AgentStreamRequest":
        supplied = int(self.input is not None) + int(self.resume is not None)
        if supplied != 1:
            raise ValueError("exactly one of input or resume must be provided")
        return self
