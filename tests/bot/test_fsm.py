from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.handlers.fsm import choose_topic
from bot.states import AskFlow


@pytest.mark.asyncio
async def test_choose_topic_moves_to_waiting_for_question_and_saves_topic():
    state = AsyncMock()
    message = SimpleNamespace(edit_text=AsyncMock())
    callback = SimpleNamespace(
        data="topic:access",
        message=message,
        answer=AsyncMock(),
    )

    await choose_topic(callback=callback, state=state)

    state.update_data.assert_awaited_once_with(
        topic_slug="access",
        topic_title="Доступы",
    )
    state.set_state.assert_awaited_once_with(AskFlow.waiting_for_question)
    message.edit_text.assert_awaited_once_with(
        "Тема: Доступы\n\nТеперь напиши вопрос."
    )
    callback.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_choose_topic_cancel_clears_state():
    state = AsyncMock()
    message = SimpleNamespace(edit_text=AsyncMock())
    callback = SimpleNamespace(
        data="topic:cancel",
        message=message,
        answer=AsyncMock(),
    )

    await choose_topic(callback=callback, state=state)

    state.clear.assert_awaited_once()
    message.edit_text.assert_awaited_once_with("Сценарий отменён.")
    callback.answer.assert_awaited_once()