from collections.abc import AsyncIterator
from uuid import UUID

import httpx


class BackendClient:
    """
    Тонкий async-клиент к chat-service из Б4.1.

    Бот не знает про LLM и не хранит историю.
    Он только вызывает backend:
    - POST /chats
    - POST /chats/{chat_id}/messages
    - DELETE /chats/{chat_id}/messages
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
    ):
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
                    "Ты ИИ-ассистент внутренней техподдержки. "
                    "Отвечай кратко, понятно и на русском языке."
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
    ) -> AsyncIterator[str]:
        async with self.http_client.stream(
            "POST",
            f"{self.base_url}/chats/{chat_id}/messages",
            json={"content": content},
        ) as response:
            response.raise_for_status()

            event_lines: list[str] = []

            async for line in response.aiter_lines():
                if line == "":
                    if not event_lines:
                        continue

                    data_parts: list[str] = []

                    for event_line in event_lines:
                        if event_line.startswith("data:"):
                            data = event_line.removeprefix("data:")

                            # SSE обычно пишет "data: text".
                            # Убираем только один служебный пробел после "data:",
                            # но сохраняем пробелы внутри самого LLM-чанка.
                            if data.startswith(" "):
                                data = data[1:]

                            data_parts.append(data)
                        else:
                            # Защита от backend-чанков с переносами строк:
                            # если строка не начинается с data:, не теряем её.
                            data_parts.append(event_line)

                    event_lines.clear()
                    data = "\n".join(data_parts)

                    if data == "[DONE]":
                        continue

                    if not data:
                        continue

                    if data.startswith("ERROR:"):
                        raise RuntimeError(data)

                    yield data
                    continue

                event_lines.append(line)

    async def clear_messages(self, chat_id: UUID) -> None:
        response = await self.http_client.delete(
            f"{self.base_url}/chats/{chat_id}/messages",
        )
        response.raise_for_status()


def build_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            timeout=30.0,
            connect=10.0,
            read=30.0,
        ),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
        ),
    )