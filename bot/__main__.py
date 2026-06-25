import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import get_bot_settings
from bot.handlers import router as handlers_router
from bot.services.backend_client import BackendClient, build_http_client


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    settings = get_bot_settings()

    if not settings.bot_token:
        raise RuntimeError(
            "BOT_TOKEN is not set. Create Telegram bot via @BotFather "
            "and put token into .env"
        )

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode="HTML"),
    )

    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(handlers_router)

    http_client = build_http_client()
    backend = BackendClient(
        http_client=http_client,
        base_url=settings.backend_url,
    )

    # aiogram будет прокидывать backend в handlers по имени параметра:
    # async def handler(message: Message, backend: BackendClient)
    dispatcher["backend"] = backend

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dispatcher.start_polling(bot)
    finally:
        await http_client.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())