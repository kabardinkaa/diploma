import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import aiofiles

from app.chat.domain import Chat, ChatMessage


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