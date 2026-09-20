from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bot.handlers import media as media_handlers


def _message(**media_fields):
    return SimpleNamespace(
        bot=SimpleNamespace(),
        answer=AsyncMock(),
        caption=None,
        **media_fields,
    )


def _backend():
    return SimpleNamespace(
        send_message=Mock(return_value=object()),
        last_message_id=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["photo", "document", "voice", "audio"])
async def test_telegram_media_size_boundary_is_enforced(
    kind,
    monkeypatch,
) -> None:
    limit = 10
    monkeypatch.setattr(
        media_handlers,
        "get_bot_settings",
        lambda: SimpleNamespace(photo_max_bytes=limit, media_max_bytes=limit),
    )
    download = AsyncMock(return_value=b"valid-data")
    monkeypatch.setattr(media_handlers, "download_telegram_file", download)
    monkeypatch.setattr(
        media_handlers,
        "get_chat_id",
        AsyncMock(return_value="chat-id"),
    )
    monkeypatch.setattr(media_handlers, "stream_to_chat", AsyncMock(return_value=None))

    if kind == "photo":
        item = SimpleNamespace(file_id="photo", file_size=limit, width=1, height=1)
        message = _message(photo=[item])
        handler = media_handlers.handle_photo
    elif kind == "document":
        item = SimpleNamespace(
            file_id="document",
            file_size=limit,
            file_name="guide.pdf",
            mime_type="application/pdf",
        )
        message = _message(document=item)
        handler = media_handlers.handle_document
    elif kind == "voice":
        item = SimpleNamespace(file_id="voice", file_size=limit)
        message = _message(voice=item)
        handler = media_handlers.handle_voice
    else:
        item = SimpleNamespace(
            file_id="audio",
            file_size=limit,
            file_name="sample.mp3",
            mime_type="audio/mpeg",
        )
        message = _message(audio=item)
        handler = media_handlers.handle_audio

    backend = _backend()
    await handler(message, backend)
    download.assert_awaited_once()

    item.file_size = limit + 1
    download.reset_mock()
    await handler(message, backend)
    download.assert_not_awaited()
    assert message.answer.await_count >= 1


@pytest.mark.asyncio
async def test_telegram_download_stops_when_actual_size_exceeds_limit() -> None:
    class FakeBot:
        async def get_file(self, _):
            return SimpleNamespace(file_path="media/file")

        async def download_file(self, _, destination):
            destination.write(b"12345")
            destination.write(b"678901")

    message = SimpleNamespace(bot=FakeBot())

    with pytest.raises(media_handlers.TelegramMediaTooLargeError):
        await media_handlers.download_telegram_file(
            message,
            "file-id",
            max_bytes=10,
        )
