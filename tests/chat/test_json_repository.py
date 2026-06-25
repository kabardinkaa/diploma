from uuid import uuid4

import pytest

from app.chat.domain import ChatMessage
from app.chat.repositories.json_repo import JsonChatRepository


@pytest.mark.asyncio
async def test_create_chat_and_get_chat(tmp_path):
    repository = JsonChatRepository(base_dir=tmp_path)

    chat = await repository.create_chat(
        owner_external_id="test-user",
        interface="cli",
        system_prompt="Ты тестовый ассистент.",
    )

    loaded_chat = await repository.get_chat(chat.id)

    assert loaded_chat is not None
    assert loaded_chat.id == chat.id
    assert loaded_chat.owner_external_id == "test-user"
    assert loaded_chat.interface == "cli"
    assert loaded_chat.system_prompt == "Ты тестовый ассистент."


@pytest.mark.asyncio
async def test_append_and_list_messages_chronological_order(tmp_path):
    repository = JsonChatRepository(base_dir=tmp_path)
    chat = await repository.create_chat("test-user", "cli")

    first = ChatMessage(
        chat_id=chat.id,
        role="user",
        content="Первое сообщение",
    )
    second = ChatMessage(
        chat_id=chat.id,
        role="assistant",
        content="Второе сообщение",
    )

    await repository.append_message(chat.id, first)
    await repository.append_message(chat.id, second)

    messages = await repository.list_messages(chat.id)

    assert [message.content for message in messages] == [
        "Первое сообщение",
        "Второе сообщение",
    ]


@pytest.mark.asyncio
async def test_list_messages_limit_returns_last_messages(tmp_path):
    repository = JsonChatRepository(base_dir=tmp_path)
    chat = await repository.create_chat("test-user", "cli")

    for index in range(5):
        await repository.append_message(
            chat.id,
            ChatMessage(
                chat_id=chat.id,
                role="user",
                content=f"message-{index}",
            ),
        )

    messages = await repository.list_messages(chat.id, limit=2)

    assert [message.content for message in messages] == [
        "message-3",
        "message-4",
    ]


@pytest.mark.asyncio
async def test_soft_delete_hides_old_messages_but_new_messages_are_visible(tmp_path):
    repository = JsonChatRepository(base_dir=tmp_path)
    chat = await repository.create_chat("test-user", "cli")

    await repository.append_message(
        chat.id,
        ChatMessage(
            chat_id=chat.id,
            role="user",
            content="Старое сообщение",
        ),
    )

    await repository.soft_delete_messages(chat.id)

    messages_after_delete = await repository.list_messages(chat.id)
    assert messages_after_delete == []

    await repository.append_message(
        chat.id,
        ChatMessage(
            chat_id=chat.id,
            role="user",
            content="Новое сообщение",
        ),
    )

    messages_after_new_append = await repository.list_messages(chat.id)

    assert len(messages_after_new_append) == 1
    assert messages_after_new_append[0].content == "Новое сообщение"


@pytest.mark.asyncio
async def test_unknown_chat_returns_none_and_empty_messages(tmp_path):
    repository = JsonChatRepository(base_dir=tmp_path)
    unknown_chat_id = uuid4()

    chat = await repository.get_chat(unknown_chat_id)
    messages = await repository.list_messages(unknown_chat_id)

    assert chat is None
    assert messages == []