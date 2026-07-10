import asyncio
import logging

import httpx
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from bot.services.backend_client import BackendClient


logger = logging.getLogger(__name__)


async def broadcast_worker(
    bot: Bot,
    backend: BackendClient,
    poll_interval_seconds: float = 5.0,
) -> None:
    while True:
        try:
            tasks = await backend.pending_broadcasts()

            for task in tasks:
                sent = 0
                failed = 0

                for recipient in task.get("recipients", []):
                    try:
                        await bot.send_message(
                            chat_id=int(recipient),
                            text=task["message"],
                        )
                        sent += 1
                    except (ValueError, TelegramAPIError):
                        failed += 1

                    await asyncio.sleep(0.04)

                await backend.update_broadcast_result(
                    task_id=task["id"],
                    sent=sent,
                    failed=failed,
                    status="done",
                )

        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.info("broadcast worker skipped cycle: %s", exc)

        await asyncio.sleep(poll_interval_seconds)
