from types import SimpleNamespace

import pytest

from app.chat.domain import ChatMessage
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.service import ChatService, MODERATION_SAFE_MESSAGE
from app.schemas.chat import ChatDelta


class FakeLLMService:
    def __init__(self, text: str = "Ответ") -> None:
        self.openai = None
        self.text = text

    async def stream(self, request):
        yield ChatDelta(content=self.text)


@pytest.mark.asyncio
async def test_feedback_deduplicates_by_owner_and_message(tmp_path) -> None:
    repository = JsonChatRepository(base_dir=tmp_path)
    chat = await repository.create_chat("42", "telegram")
    message = await repository.append_message(
        chat.id,
        ChatMessage(
            chat_id=chat.id,
            role="assistant",
            content="ok",
        ),
    )

    first = await repository.save_feedback(chat.id, message.id, "42", "up")
    second = await repository.save_feedback(chat.id, message.id, "42", "down")

    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_prompt_split_is_deterministic(tmp_path) -> None:
    repository = JsonChatRepository(base_dir=tmp_path)
    service = ChatService(repository, FakeLLMService())

    first = await service.select_prompt("owner-1")
    second = await service.select_prompt("owner-1")

    assert first is not None
    assert second is not None
    assert first.id == second.id


@pytest.mark.asyncio
async def test_prompt_id_saved_on_assistant_message(tmp_path) -> None:
    repository = JsonChatRepository(base_dir=tmp_path)
    chat = await repository.create_chat("owner-2", "telegram")
    service = ChatService(repository, FakeLLMService("Готово"))

    chunks = [
        chunk
        async for chunk in service.send_message(chat.id, "Привет")
    ]
    messages = await repository.list_messages(chat.id)

    assert chunks == ["Готово"]
    assert messages[-1].role == "assistant"
    assert messages[-1].prompt_id is not None


@pytest.mark.asyncio
async def test_blocked_output_is_replaced(tmp_path) -> None:
    repository = JsonChatRepository(base_dir=tmp_path)
    chat = await repository.create_chat("owner-3", "telegram")
    service = ChatService(repository, FakeLLMService("keylogger code"))

    chunks = [
        chunk
        async for chunk in service.send_message(chat.id, "Привет")
    ]

    assert chunks == [MODERATION_SAFE_MESSAGE]
