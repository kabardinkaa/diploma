from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


HandoffStatus = Literal["active", "paused_for_human"]
FeedbackValue = Literal["up", "down"]
BroadcastStatus = Literal["pending", "done", "failed"]


class Chat(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    owner_external_id: str
    interface: str
    system_prompt: str | None = None
    handoff_status: HandoffStatus = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatMessage(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: Literal["user", "assistant", "system"]
    content: str
    media_refs: dict | None = None
    sources: list[dict] = Field(default_factory=list)
    prompt_id: UUID | None = None
    tokens: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Feedback(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    message_id: UUID
    owner_external_id: str
    value: FeedbackValue
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BroadcastTask(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    message: str
    interface_filter: str | None = None
    status: BroadcastStatus = "pending"
    sent: int = 0
    failed: int = 0
    recipients: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None


class SystemPrompt(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    version: str
    body: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    active: bool = True
    traffic_pct: int = Field(ge=0, le=100)
    notes: str | None = None


class AdminStats(BaseModel):
    total_messages: int
    active_users: int
    avg_latency_ms: float | None = None
    moderation_block_rate: float
    feedback_up_ratio: float | None = None


class AdminUser(BaseModel):
    owner_external_id: str
    chats_count: int
    last_seen_at: datetime
