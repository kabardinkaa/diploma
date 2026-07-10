import httpx
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.handlers.commands import get_owner_external_id
from bot.services.backend_client import BackendClient


router = Router()


@router.message(Command("operator"))
async def cmd_operator(
    message: Message,
    backend: BackendClient,
) -> None:
    try:
        chat_id = await backend.get_or_create_chat(
            owner_external_id=get_owner_external_id(message),
            interface="telegram",
        )
        await backend.set_handoff(chat_id)
    except httpx.HTTPError:
        await message.answer("Не удалось передать диалог оператору. Попробуйте позже.")
        return

    await message.answer("Передаю запрос оператору, ожидайте.")
