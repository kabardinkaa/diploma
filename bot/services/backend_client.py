import json
from collections.abc import AsyncIterator
from uuid import UUID

import httpx


class BackendClient:
    """Асинхронный клиент Telegram-бота для работы с chat-service."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
    ) -> None:
        self.http_client = http_client
        self.base_url = base_url.rstrip("/")
        self._chat_cache: dict[tuple[str, str], UUID] = {}

    async def get_or_create_chat(
        self,
        owner_external_id: str,
        interface: str,
    ) -> UUID:
        cache_key = (owner_external_id, interface)

        if cache_key in self._chat_cache:
            return self._chat_cache[cache_key]

        response = await self.http_client.post(
            f"{self.base_url}/chats",
            json={
                "owner_external_id": owner_external_id,
                "interface": interface,
                "system_prompt": (
                    "Ты — ИИ-ассистент технической поддержки. "
                    "Отвечай понятно, вежливо и по существу."
                ),
            },
        )
        response.raise_for_status()

        payload = response.json()
        chat_id = UUID(payload["chat_id"])

        self._chat_cache[cache_key] = chat_id
        return chat_id

    async def send_message(
        self,
        chat_id: UUID,
        content: str,
        media: bytes | None = None,
        mime: str | None = None,
    ) -> AsyncIterator[str]:
        multipart_parts: dict[str, tuple] = {
            "content": (None, content),
        }

        if media is not None:
            multipart_parts["media"] = (
                "file.bin",
                media,
                mime or "application/octet-stream",
            )

        async with self.http_client.stream(
            "POST",
            f"{self.base_url}/chats/{chat_id}/messages",
            files=multipart_parts,
            timeout=httpx.Timeout(
                connect=10.0,
                read=600.0,
                write=60.0,
                pool=10.0,
            ),
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                raw_payload = line.removeprefix("data: ").strip()

                if not raw_payload:
                    continue

                payload = json.loads(raw_payload)
                event_type = payload.get("type")

                if event_type == "token":
                    delta = payload.get("delta", "")

                    if delta:
                        yield delta

                elif event_type == "done":
                    return

                elif event_type == "error":
                    message = payload.get(
                        "message",
                        "Backend returned an unknown error",
                    )
                    raise RuntimeError(message)

    async def clear_messages(self, chat_id: UUID) -> None:
        response = await self.http_client.delete(
            f"{self.base_url}/chats/{chat_id}/messages",
        )
        response.raise_for_status()


def build_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0,
            read=600.0,
            write=60.0,
            pool=10.0,
        ),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30.0,
        ),
        headers={
            "X-Client": "telegram-bot/1.0",
        },
    )