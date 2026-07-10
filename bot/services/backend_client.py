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
        admin_token: str = "",
        internal_token: str = "",
    ) -> None:
        self.http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.admin_token = admin_token
        self.internal_token = internal_token
        self._chat_cache: dict[tuple[str, str], UUID] = {}
        self.last_message_id: UUID | None = None

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
            if response.status_code == 403:
                raise RuntimeError("moderation_blocked")

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
                    message_id = payload.get("message_id")
                    self.last_message_id = UUID(message_id) if message_id else None
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

    async def send_feedback(
        self,
        chat_id: UUID,
        message_id: UUID,
        value: str,
    ) -> None:
        response = await self.http_client.post(
            f"{self.base_url}/chats/{chat_id}/messages/{message_id}/feedback",
            json={"value": value},
        )
        response.raise_for_status()

    async def set_handoff(self, chat_id: UUID) -> None:
        response = await self.http_client.post(
            f"{self.base_url}/chats/{chat_id}/handoff",
        )
        response.raise_for_status()

    def _admin_headers(self) -> dict[str, str]:
        return {"X-Admin-Token": self.admin_token}

    def _internal_headers(self) -> dict[str, str]:
        return {"X-Internal-Token": self.internal_token}

    async def admin_stats(self) -> dict:
        response = await self.http_client.get(
            f"{self.base_url}/chats/admin/stats",
            headers=self._admin_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def admin_users(self, limit: int = 10) -> list[dict]:
        response = await self.http_client.get(
            f"{self.base_url}/chats/admin/users",
            params={"limit": limit},
            headers=self._admin_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def create_broadcast(self, message: str, interface_filter: str = "telegram") -> dict:
        response = await self.http_client.post(
            f"{self.base_url}/chats/admin/broadcast",
            json={
                "message": message,
                "interface_filter": interface_filter,
            },
            headers=self._admin_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def pending_broadcasts(self) -> list[dict]:
        response = await self.http_client.get(
            f"{self.base_url}/chats/admin/internal/broadcasts/pending",
            headers=self._internal_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def update_broadcast_result(
        self,
        task_id: str,
        sent: int,
        failed: int,
        status: str = "done",
    ) -> None:
        response = await self.http_client.post(
            f"{self.base_url}/chats/admin/internal/broadcasts/{task_id}/result",
            json={
                "sent": sent,
                "failed": failed,
                "status": status,
            },
            headers=self._internal_headers(),
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
