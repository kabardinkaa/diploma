from uuid import UUID

import httpx
from aiogram import F, Router
from aiogram.types import CallbackQuery

from bot.services.backend_client import BackendClient


router = Router()


@router.callback_query(F.data.startswith("fb:"))
async def handle_feedback(
    callback: CallbackQuery,
    backend: BackendClient,
) -> None:
    parts = (callback.data or "").split(":")

    if len(parts) != 3 or parts[1] not in {"up", "down"}:
        await callback.answer("Некорректная оценка", show_alert=True)
        return

    value = parts[1]

    try:
        message_id = UUID(parts[2])
    except ValueError:
        await callback.answer("Некорректная оценка", show_alert=True)
        return

    if callback.message is None:
        await callback.answer()
        return

    try:
        chat_id = await backend.get_or_create_chat(
            owner_external_id=str(callback.from_user.id),
            interface="telegram",
        )
        await backend.send_feedback(
            chat_id=chat_id,
            message_id=message_id,
            value=value,
        )
        await callback.message.edit_reply_markup(reply_markup=None)
        await callback.answer(
            "Спасибо!" if value == "up" else "Учтём, разберёмся."
        )
    except (httpx.HTTPError, RuntimeError):
        await callback.answer("Не удалось сохранить оценку", show_alert=True)
