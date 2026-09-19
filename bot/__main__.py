import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import get_bot_settings
from bot.handlers import router as handlers_router
from bot.services.broadcast import broadcast_worker
from bot.services.backend_client import BackendClient, build_http_client
from bot.services.quota import BotUserQuota

PLACEHOLDER_BOT_TOKEN = "change-me-telegram-bot-token"
PLACEHOLDER_INTERNAL_TOKEN = "change-me-internal-token"


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    settings = get_bot_settings()

    bot_token = settings.bot_token.strip()
    if not bot_token or bot_token == PLACEHOLDER_BOT_TOKEN:
        logging.warning(
            "Telegram bot is disabled: set a real BOT_TOKEN to enable polling"
        )
        return

    bot = Bot(
        token=bot_token,
        default=DefaultBotProperties(parse_mode="HTML"),
    )

    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(handlers_router)

    http_client = build_http_client()
    internal_token = settings.internal_token.strip()
    backend = BackendClient(
        http_client=http_client,
        base_url=settings.backend_url,
        admin_token=settings.admin_token,
        internal_token=internal_token,
        user_quota=BotUserQuota(
            requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
            daily_quota=settings.daily_quota,
        ),
    )

    # aiogram будет прокидывать backend в handlers по имени параметра:
    # async def handler(message: Message, backend: BackendClient)
    dispatcher["backend"] = backend

    worker_task: asyncio.Task | None = None

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        if internal_token and internal_token != PLACEHOLDER_INTERNAL_TOKEN:
            worker_task = asyncio.create_task(
                broadcast_worker(bot=bot, backend=backend)
            )
        else:
            logging.info(
                "Broadcast worker is disabled: set a real INTERNAL_TOKEN to enable it"
            )
        await dispatcher.start_polling(bot)
    finally:
        if worker_task is not None:
            worker_task.cancel()
        await http_client.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
