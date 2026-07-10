import json
from uuid import uuid4

import httpx
import pytest

from bot.services.backend_client import BackendClient


@pytest.mark.asyncio
async def test_get_or_create_chat_caches_chat_id() -> None:
    chat_id = uuid4()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1

        assert request.method == "POST"
        assert request.url.path == "/chats"

        payload = json.loads(request.content.decode("utf-8"))
        assert payload["owner_external_id"] == "123"
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

        first = await backend.get_or_create_chat(
            owner_external_id="123",
            interface="telegram",
        )
        second = await backend.get_or_create_chat(
            owner_external_id="123",
            interface="telegram",
        )

    assert first == chat_id
    assert second == chat_id
    assert calls == 1


@pytest.mark.asyncio
async def test_clear_messages_calls_delete() -> None:
    chat_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == f"/chats/{chat_id}/messages"

        return httpx.Response(status_code=200)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        backend = BackendClient(
            http_client=http_client,
            base_url="http://backend.test",
        )

        await backend.clear_messages(chat_id)


@pytest.mark.asyncio
async def test_send_message_parses_json_sse_frames() -> None:
    chat_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/chats/{chat_id}/messages"

        content_type = request.headers["content-type"]
        assert content_type.startswith("multipart/form-data; boundary=")

        body = request.content
        assert b'name="content"' in body
        assert "Привет".encode("utf-8") in body

        return httpx.Response(
            status_code=200,
            content=(
                'data: {"type":"token","delta":"При"}\n\n'
                'data: {"type":"token","delta":"вет"}\n\n'
                'data: {"type":"done"}\n\n'
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
            async for chunk in backend.send_message(
                chat_id,
                "Привет",
            )
        ]

    assert chunks == ["При", "вет"]


@pytest.mark.asyncio
async def test_send_message_preserves_spaces_and_multiline_chunks() -> None:
    chat_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/chats/{chat_id}/messages"

        return httpx.Response(
            status_code=200,
            content=(
                'data: {"type":"token","delta":"Привет, "}\n\n'
                'data: {"type":"token","delta":"Диана\\n"}\n\n'
                'data: {"type":"token","delta":"Строка 1\\nСтрока 2"}\n\n'
                'data: {"type":"done"}\n\n'
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
            async for chunk in backend.send_message(
                chat_id,
                "Привет",
            )
        ]

    assert chunks == [
        "Привет, ",
        "Диана\n",
        "Строка 1\nСтрока 2",
    ]


@pytest.mark.asyncio
async def test_send_message_sends_media_as_multipart() -> None:
    chat_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/chats/{chat_id}/messages"

        content_type = request.headers["content-type"]
        assert content_type.startswith("multipart/form-data; boundary=")

        body = request.content

        assert b'name="content"' in body
        assert "Опиши изображение".encode("utf-8") in body
        assert b'name="media"; filename="file.bin"' in body
        assert b"Content-Type: image/png" in body
        assert b"fake-png-data" in body

        return httpx.Response(
            status_code=200,
            content=(
                'data: {"type":"token","delta":"Готово"}\n\n'
                'data: {"type":"done"}\n\n'
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
            async for chunk in backend.send_message(
                chat_id=chat_id,
                content="Опиши изображение",
                media=b"fake-png-data",
                mime="image/png",
            )
        ]

    assert chunks == ["Готово"]