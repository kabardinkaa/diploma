import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import monotonic

from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message


EDIT_INTERVAL_SECONDS = 0.8
TELEGRAM_MESSAGE_LIMIT = 4096


@dataclass
class StreamResult:
    text: str
    message: Message

    def __bool__(self) -> bool:
        return bool(self.text.strip())


async def safe_edit_message(
    message: Message,
    text: str,
) -> None:
    if not text.strip():
        return

    safe_text = text[:TELEGRAM_MESSAGE_LIMIT]

    try:
        await message.edit_text(
            safe_text,
            parse_mode=None,
        )

    except TelegramRetryAfter as exc:
        await asyncio.sleep(float(exc.retry_after))

        await message.edit_text(
            safe_text,
            parse_mode=None,
        )

    except TelegramBadRequest as exc:
        error_text = str(exc).lower()

        if "message is not modified" in error_text:
            return

        raise


async def stream_to_chat(
    message: Message,
    tokens: AsyncIterator[str],
) -> StreamResult:
    """
    Показывает поток ответа через редактирование
    обычного сообщения Telegram.
    """

    answer_message = await message.answer(
        "Обрабатываю запрос…",
        parse_mode=None,
    )

    buffer = ""
    last_edit_at = 0.0

    try:
        async for delta in tokens:
            if not delta:
                continue

            buffer += delta

            if not buffer.strip():
                continue

            now = monotonic()

            if now - last_edit_at < EDIT_INTERVAL_SECONDS:
                continue

            await safe_edit_message(
                answer_message,
                buffer,
            )

            last_edit_at = now

    except Exception:
        if buffer.strip():
            await safe_edit_message(
                answer_message,
                buffer,
            )
        else:
            await safe_edit_message(
                answer_message,
                "Не удалось получить ответ.",
            )

        raise

    if not buffer.strip():
        await safe_edit_message(
            answer_message,
            "Backend не вернул текст ответа.",
        )
        return StreamResult(text="", message=answer_message)

    await safe_edit_message(
        answer_message,
        buffer,
    )

    return StreamResult(text=buffer, message=answer_message)
