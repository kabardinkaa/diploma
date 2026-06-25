import httpx
from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from bot.handlers.commands import get_owner_external_id
from bot.services.backend_client import BackendClient


router = Router()


async def safe_edit_text(message: Message, text: str) -> None:
    try:
        await message.edit_text(text)
    except TelegramBadRequest:
        # Telegram может ругаться, если текст не изменился.
        pass


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text_message(
    message: Message,
    backend: BackendClient,
) -> None:
    if not message.text:
        return

    try:
        chat_id = await backend.get_or_create_chat(
            owner_external_id=get_owner_external_id(message),
            interface="telegram",
        )

        answer_message = await message.answer("Думаю...")
        buffer = ""

        async for chunk in backend.send_message(chat_id, message.text):
            buffer += chunk

            if buffer.strip():
                await safe_edit_text(answer_message, buffer)

        if not buffer.strip():
            await safe_edit_text(
                answer_message,
                "Backend вернул пустой ответ.",
            )

    except httpx.HTTPError:
        await message.answer(
            "Не удалось получить ответ от backend. Проверь, что chat-сервис запущен."
        )
    except RuntimeError as exc:
        await message.answer(f"Backend вернул ошибку: {exc}")