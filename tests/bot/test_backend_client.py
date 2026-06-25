import json
from uuid import UUID, uuid4

import httpx
import pytest

from bot.services.backend_client import BackendClient


@pytest.mark.asyncio
async def test_get_or_create_chat_returns_uuid_and_uses_cache():
    chat_id = uuid4()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        assert request.method == "POST"
        assert request.url.path == "/chats"

        payload = json.loads(request.content.decode("utf-8"))
        assert payload["owner_external_id"] == "telegram-user-1"
        assert payload["interface"] == "telegram"

        return httpx.Response(
            status_code=200,
            json={"chat_id": str(chat_id)},
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        backend = BackendClient(
            http_client=http_client,
            base_url="http://backend.test",
        )

        first_result = await backend.get_or_create_chat(
            owner_external_id="telegram-user-1",
            interface="telegram",
        )
        second_result = await backend.get_or_create_chat(
            owner_external_id="telegram-user-1",
            interface="telegram",
        )

    assert first_result == chat_id
    assert second_result == chat_id
    assert isinstance(first_result, UUID)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_send_message_parses_sse_frames():
    chat_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/chats/{chat_id}/messages"

        payload = json.loads(request.content.decode("utf-8"))
        assert payload == {"content": "Привет"}

        return httpx.Response(
            status_code=200,
            content=(
                "data: При\n\n"
                "data: вет\n\n"
                "data: \n\n"
                "data: [DONE]\n\n"
            ).encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        backend = BackendClient(
            http_client=http_client,
            base_url="http://backend.test",
        )

        chunks = [
            chunk
            async for chunk in backend.send_message(chat_id, "Привет")
        ]

    assert chunks == ["При", "вет"]


@pytest.mark.asyncio
async def test_clear_messages_sends_delete_to_backend():
    chat_id = uuid4()
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True

        assert request.method == "DELETE"
        assert request.url.path == f"/chats/{chat_id}/messages"

        return httpx.Response(status_code=200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        backend = BackendClient(
            http_client=http_client,
            base_url="http://backend.test",
        )

        await backend.clear_messages(chat_id)

    assert called is True

@pytest.mark.asyncio
async def test_send_message_preserves_spaces_and_multiline_chunks():
    chat_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/chats/{chat_id}/messages"

        return httpx.Response(
            status_code=200,
            content=(
                "data: Привет,\n\n"
                "data:  Диана\n\n"
                "data: Строка 1\nСтрока 2\n\n"
                "data: [DONE]\n\n"
            ).encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        backend = BackendClient(
            http_client=http_client,
            base_url="http://backend.test",
        )

        chunks = [
            chunk
            async for chunk in backend.send_message(chat_id, "Привет")
        ]

    assert chunks == [
        "Привет,",
        " Диана",
        "Строка 1\nСтрока 2",
    ]