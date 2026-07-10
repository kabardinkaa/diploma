import html

import httpx
from aiogram import Router
from aiogram.filters import BaseFilter, Command
from aiogram.types import Message

from bot.config import get_bot_settings
from bot.services.backend_client import BackendClient


router = Router()


class IsAdmin(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        if message.from_user is None:
            return False

        return message.from_user.id in get_bot_settings().bot_admin_ids


router.message.filter(IsAdmin())


async def send_admin_error(message: Message, error: Exception) -> None:
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 403:
        await message.answer("Нет доступа к admin API.")
        return

    await message.answer("Не удалось выполнить admin-команду.")


@router.message(Command("stats"))
async def admin_stats(message: Message, backend: BackendClient) -> None:
    try:
        stats = await backend.admin_stats()
    except (httpx.HTTPError, RuntimeError) as exc:
        await send_admin_error(message, exc)
        return

    await message.answer(
        "<b>Статистика за 24 часа</b>\n"
        f"Сообщений: <b>{stats.get('total_messages', 0)}</b>\n"
        f"Активных пользователей: <b>{stats.get('active_users', 0)}</b>\n"
        f"Средняя latency: <b>{stats.get('avg_latency_ms') or 0}</b> ms\n"
        f"Moderation block rate: <b>{stats.get('moderation_block_rate', 0)}</b>\n"
        f"Feedback up ratio: <b>{stats.get('feedback_up_ratio') or 0}</b>"
    )


@router.message(Command("users"))
async def admin_users(message: Message, backend: BackendClient) -> None:
    try:
        users = await backend.admin_users(limit=10)
    except (httpx.HTTPError, RuntimeError) as exc:
        await send_admin_error(message, exc)
        return

    if not users:
        await message.answer("Пользователей пока нет.")
        return

    lines = ["<b>Последние пользователи</b>"]
    for item in users[:10]:
        lines.append(
            f"{html.escape(str(item.get('owner_external_id')))} "
            f"чаты: {item.get('chats_count', 0)} "
            f"last_seen: {html.escape(str(item.get('last_seen_at')))}"
        )

    await message.answer("\n".join(lines))


@router.message(Command("broadcast"))
async def admin_broadcast(message: Message, backend: BackendClient) -> None:
    text = (message.text or "").partition(" ")[2].strip()

    if not text:
        await message.answer("Использование: /broadcast текст рассылки")
        return

    try:
        task = await backend.create_broadcast(
            message=text,
            interface_filter="telegram",
        )
    except (httpx.HTTPError, RuntimeError) as exc:
        await send_admin_error(message, exc)
        return

    await message.answer(
        "Рассылка создана.\n"
        f"id: <code>{html.escape(str(task.get('id')))}</code>\n"
        f"получателей: <b>{len(task.get('recipients', []))}</b>"
    )
