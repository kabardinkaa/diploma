import httpx
from aiogram import F, Router
from aiogram.types import Message

from bot.handlers.commands import get_owner_external_id
from bot.keyboards.feedback import feedback_kb
from bot.services.backend_client import BackendClient
from bot.services.streaming import safe_edit_message, stream_to_chat, with_sources


router = Router()


async def send_backend_error(
    message: Message,
    error: Exception,
) -> None:
    if isinstance(error, RuntimeError) and str(error) == "moderation_blocked":
        text = "Запрос нарушает правила. Попробуйте переформулировать."

    elif isinstance(error, httpx.ConnectError):
        text = "Сервис сейчас недоступен. Попробуйте позже."

    elif isinstance(error, httpx.ReadTimeout):
        text = "Ответ занимает слишком долго. Попробуйте ещё раз."

    elif isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code

        if status_code == 429:
            text = "Слишком много запросов. Подождите минуту."
        elif status_code >= 500:
            text = "Внутренняя ошибка сервиса."
        else:
            text = "Не удалось обработать запрос."

    else:
        text = f"Не удалось получить ответ: {error}"

    await message.answer(text)


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

        tokens = backend.send_message(
            chat_id=chat_id,
            content=message.text,
        )

        result = await stream_to_chat(
            message=message,
            tokens=tokens,
        )

        if result and backend.last_sources:
            result.text = with_sources(result.text, backend.last_sources)
            await safe_edit_message(result.message, result.text)

        if result and backend.last_message_id is not None:
            await result.message.edit_reply_markup(
                reply_markup=feedback_kb(backend.last_message_id)
            )

        if not result:
            await message.answer(
                "Backend не вернул текст ответа."
            )

    except (
        httpx.HTTPError,
        RuntimeError,
    ) as exc:
        await send_backend_error(message, exc)
