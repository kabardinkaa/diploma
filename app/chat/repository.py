from typing import Protocol
from uuid import UUID

from app.chat.domain import AdminStats, AdminUser, BroadcastTask, Chat, ChatMessage, Feedback, FeedbackValue, HandoffStatus, SystemPrompt


class ChatRepository(Protocol):
    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        ...

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        ...

    async def append_message(
        self,
        chat_id: UUID,
        message: ChatMessage,
    ) -> ChatMessage:
        ...

    async def list_messages(
        self,
        chat_id: UUID,
        limit: int = 50,
    ) -> list[ChatMessage]:
        ...

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        ...

    async def save_feedback(
        self,
        chat_id: UUID,
        message_id: UUID,
        owner_external_id: str,
        value: FeedbackValue,
    ) -> Feedback | None:
        ...

    async def set_handoff_status(
        self,
        chat_id: UUID,
        status: HandoffStatus,
    ) -> Chat | None:
        ...

    async def admin_stats(self) -> AdminStats:
        ...

    async def list_admin_users(self, limit: int = 50) -> list[AdminUser]:
        ...

    async def create_broadcast(
        self,
        message: str,
        interface_filter: str | None = None,
    ) -> BroadcastTask:
        ...

    async def list_pending_broadcasts(self, limit: int = 10) -> list[BroadcastTask]:
        ...

    async def update_broadcast_result(
        self,
        task_id: UUID,
        sent: int,
        failed: int,
        status: str = "done",
    ) -> BroadcastTask | None:
        ...

    async def list_active_prompts(self) -> list[SystemPrompt]:
        ...
