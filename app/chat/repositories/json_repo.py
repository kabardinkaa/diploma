import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import aiofiles

from app.chat.domain import (
    AdminStats,
    AdminUser,
    BroadcastTask,
    Chat,
    ChatMessage,
    Feedback,
    FeedbackValue,
    HandoffStatus,
    SystemPrompt,
)


DEFAULT_PROMPTS = [
    SystemPrompt(
        version="A",
        body=(
            "Ты — ИИ-ассистент технической поддержки. Отвечай по-русски, "
            "коротко, вежливо и по делу."
        ),
        active=True,
        traffic_pct=50,
        notes="Базовая краткая версия",
    ),
    SystemPrompt(
        version="B",
        body=(
            "Ты — ИИ-ассистент внутренней технической поддержки. Отвечай "
            "по-русски, структурно, безопасно и при необходимости уточняй детали."
        ),
        active=True,
        traffic_pct=50,
        notes="Более структурная версия",
    ),
]


class JsonChatRepository:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.chats_dir = self.base_dir / "chats"

    def _chat_dir(self, chat_id: UUID) -> Path:
        return self.chats_dir / str(chat_id)

    def _chat_path(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "chat.json"

    def _messages_path(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "messages.jsonl"

    def _feedback_path(self) -> Path:
        return self.base_dir / "feedback.jsonl"

    def _broadcasts_path(self) -> Path:
        return self.base_dir / "broadcasts.jsonl"

    def _prompts_path(self) -> Path:
        return self.base_dir / "system_prompts.jsonl"

    async def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []

        async with aiofiles.open(path, "r", encoding="utf-8") as file:
            lines = await file.readlines()

        return [
            json.loads(line)
            for line in lines
            if line.strip()
        ]

    async def _rewrite_jsonl(self, path: Path, items: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(path, "w", encoding="utf-8") as file:
            for item in items:
                await file.write(json.dumps(item, ensure_ascii=False))
                await file.write("\n")

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        chat = Chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )

        chat_dir = self._chat_dir(chat.id)
        chat_dir.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(self._chat_path(chat.id), "w", encoding="utf-8") as file:
            await file.write(chat.model_dump_json())

        messages_path = self._messages_path(chat.id)
        if not messages_path.exists():
            async with aiofiles.open(messages_path, "w", encoding="utf-8"):
                pass

        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        chat_path = self._chat_path(chat_id)

        if not chat_path.exists():
            return None

        async with aiofiles.open(chat_path, "r", encoding="utf-8") as file:
            raw_chat = await file.read()

        return Chat.model_validate_json(raw_chat)

    async def append_message(
        self,
        chat_id: UUID,
        message: ChatMessage,
    ) -> ChatMessage:
        chat_dir = self._chat_dir(chat_id)
        chat_dir.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(self._messages_path(chat_id), "a", encoding="utf-8") as file:
            await file.write(message.model_dump_json())
            await file.write("\n")

        return message

    async def list_messages(
        self,
        chat_id: UUID,
        limit: int = 50,
    ) -> list[ChatMessage]:
        messages_path = self._messages_path(chat_id)

        if not messages_path.exists():
            return []

        async with aiofiles.open(messages_path, "r", encoding="utf-8") as file:
            lines = await file.readlines()

        messages: list[ChatMessage] = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            raw_item = json.loads(line)

            if raw_item.get("type") == "soft_delete":
                messages = []
                continue

            messages.append(ChatMessage.model_validate(raw_item))

        return messages[-limit:]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        chat_dir = self._chat_dir(chat_id)
        chat_dir.mkdir(parents=True, exist_ok=True)

        marker = {
            "type": "soft_delete",
            "at": datetime.now(UTC).isoformat(),
        }

        async with aiofiles.open(self._messages_path(chat_id), "a", encoding="utf-8") as file:
            await file.write(json.dumps(marker, ensure_ascii=False))
            await file.write("\n")

    async def save_feedback(
        self,
        chat_id: UUID,
        message_id: UUID,
        owner_external_id: str,
        value: FeedbackValue,
    ) -> Feedback | None:
        feedback_path = self._feedback_path()
        items = await self._read_jsonl(feedback_path)

        for item in items:
            if (
                item.get("owner_external_id") == owner_external_id
                and item.get("message_id") == str(message_id)
            ):
                return None

        feedback = Feedback(
            chat_id=chat_id,
            message_id=message_id,
            owner_external_id=owner_external_id,
            value=value,
        )

        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(feedback_path, "a", encoding="utf-8") as file:
            await file.write(feedback.model_dump_json())
            await file.write("\n")

        return feedback

    async def set_handoff_status(
        self,
        chat_id: UUID,
        status: HandoffStatus,
    ) -> Chat | None:
        chat = await self.get_chat(chat_id)

        if chat is None:
            return None

        chat.handoff_status = status

        async with aiofiles.open(self._chat_path(chat_id), "w", encoding="utf-8") as file:
            await file.write(chat.model_dump_json())

        return chat

    async def _list_chats(self) -> list[Chat]:
        if not self.chats_dir.exists():
            return []

        chats: list[Chat] = []

        for chat_path in self.chats_dir.glob("*/chat.json"):
            async with aiofiles.open(chat_path, "r", encoding="utf-8") as file:
                chats.append(Chat.model_validate_json(await file.read()))

        return chats

    async def admin_stats(self) -> AdminStats:
        now = datetime.now(UTC)
        since = now.timestamp() - 24 * 60 * 60
        chats = await self._list_chats()
        owners_by_chat = {
            chat.id: chat.owner_external_id
            for chat in chats
        }
        active_owners: set[str] = set()
        total_messages = 0

        for chat in chats:
            for message in await self.list_messages(chat.id, limit=100_000):
                if message.created_at.timestamp() < since:
                    continue

                total_messages += 1
                owner = owners_by_chat.get(message.chat_id)
                if owner:
                    active_owners.add(owner)

        feedback_items = [
            Feedback.model_validate(item)
            for item in await self._read_jsonl(self._feedback_path())
        ]
        recent_feedback = [
            item
            for item in feedback_items
            if item.created_at.timestamp() >= since
        ]
        up_count = len([item for item in recent_feedback if item.value == "up"])
        feedback_up_ratio = (
            up_count / len(recent_feedback)
            if recent_feedback
            else None
        )

        return AdminStats(
            total_messages=total_messages,
            active_users=len(active_owners),
            avg_latency_ms=None,
            moderation_block_rate=0.0,
            feedback_up_ratio=feedback_up_ratio,
        )

    async def list_admin_users(self, limit: int = 50) -> list[AdminUser]:
        chats = await self._list_chats()
        grouped: dict[str, AdminUser] = {}

        for chat in chats:
            existing = grouped.get(chat.owner_external_id)

            if existing is None:
                grouped[chat.owner_external_id] = AdminUser(
                    owner_external_id=chat.owner_external_id,
                    chats_count=1,
                    last_seen_at=chat.created_at,
                )
            else:
                existing.chats_count += 1
                if chat.created_at > existing.last_seen_at:
                    existing.last_seen_at = chat.created_at

            messages = await self.list_messages(chat.id, limit=1)
            if messages and messages[-1].created_at > grouped[chat.owner_external_id].last_seen_at:
                grouped[chat.owner_external_id].last_seen_at = messages[-1].created_at

        return sorted(
            grouped.values(),
            key=lambda item: item.last_seen_at,
            reverse=True,
        )[:limit]

    async def create_broadcast(
        self,
        message: str,
        interface_filter: str | None = None,
    ) -> BroadcastTask:
        recipients = sorted(
            {
                chat.owner_external_id
                for chat in await self._list_chats()
                if interface_filter is None or chat.interface == interface_filter
            }
        )
        task = BroadcastTask(
            message=message,
            interface_filter=interface_filter,
            recipients=recipients,
        )
        path = self._broadcasts_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(path, "a", encoding="utf-8") as file:
            await file.write(task.model_dump_json())
            await file.write("\n")

        return task

    async def list_pending_broadcasts(self, limit: int = 10) -> list[BroadcastTask]:
        items = [
            BroadcastTask.model_validate(item)
            for item in await self._read_jsonl(self._broadcasts_path())
        ]

        return [
            item
            for item in items
            if item.status == "pending"
        ][:limit]

    async def update_broadcast_result(
        self,
        task_id: UUID,
        sent: int,
        failed: int,
        status: str = "done",
    ) -> BroadcastTask | None:
        path = self._broadcasts_path()
        items = await self._read_jsonl(path)
        updated: BroadcastTask | None = None

        for item in items:
            if item.get("id") == str(task_id):
                item["sent"] = sent
                item["failed"] = failed
                item["status"] = status
                item["updated_at"] = datetime.now(UTC).isoformat()
                updated = BroadcastTask.model_validate(item)

        await self._rewrite_jsonl(path, items)
        return updated

    async def list_active_prompts(self) -> list[SystemPrompt]:
        path = self._prompts_path()
        items = await self._read_jsonl(path)

        if not items:
            await self._rewrite_jsonl(
                path,
                [
                    prompt.model_dump(mode="json")
                    for prompt in DEFAULT_PROMPTS
                ],
            )
            items = await self._read_jsonl(path)

        prompts = [
            SystemPrompt.model_validate(item)
            for item in items
        ]
        active = [
            prompt
            for prompt in prompts
            if prompt.active
        ]

        return active
