from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from bot.handlers.admin import IsAdmin
from bot.handlers.feedback import handle_feedback
from bot.handlers.operator import cmd_operator
from bot.handlers.text import send_backend_error
from bot.keyboards.feedback import feedback_kb


@pytest.mark.asyncio
async def test_moderation_403_is_translated_for_user() -> None:
    message = SimpleNamespace(answer=AsyncMock())

    await send_backend_error(message, RuntimeError("moderation_blocked"))

    message.answer.assert_awaited_once_with(
        "Запрос нарушает правила. Попробуйте переформулировать."
    )


@pytest.mark.asyncio
async def test_admin_filter_rejects_non_admin(monkeypatch) -> None:
    from bot.config import get_bot_settings

    monkeypatch.setenv("BOT_ADMIN_IDS", "1")
    get_bot_settings.cache_clear()
    message = SimpleNamespace(from_user=SimpleNamespace(id=2))

    try:
        assert await IsAdmin()(message) is False
    finally:
        get_bot_settings.cache_clear()


def test_feedback_buttons_are_formed() -> None:
    message_id = uuid4()
    markup = feedback_kb(message_id)
    buttons = markup.inline_keyboard[0]

    assert buttons[0].callback_data == f"fb:up:{message_id}"
    assert buttons[1].callback_data == f"fb:down:{message_id}"


@pytest.mark.asyncio
async def test_feedback_callback_parses_and_sends_feedback() -> None:
    message_id = uuid4()
    backend = SimpleNamespace(
        get_or_create_chat=AsyncMock(return_value=uuid4()),
        send_feedback=AsyncMock(),
    )
    callback = SimpleNamespace(
        data=f"fb:up:{message_id}",
        from_user=SimpleNamespace(id=123),
        message=SimpleNamespace(edit_reply_markup=AsyncMock()),
        answer=AsyncMock(),
    )

    await handle_feedback(callback, backend)

    backend.send_feedback.assert_awaited_once()
    callback.message.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
    callback.answer.assert_awaited_once_with("Спасибо!")


@pytest.mark.asyncio
async def test_operator_command_sets_handoff() -> None:
    chat_id = uuid4()
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=123),
        chat=SimpleNamespace(id=123),
        answer=AsyncMock(),
    )
    backend = SimpleNamespace(
        get_or_create_chat=AsyncMock(return_value=chat_id),
        set_handoff=AsyncMock(),
    )

    await cmd_operator(message, backend)

    backend.set_handoff.assert_awaited_once_with(chat_id)
    message.answer.assert_awaited_once_with("Передаю запрос оператору, ожидайте.")
