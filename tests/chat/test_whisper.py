from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import UploadFile

from app.chat.media import media_to_part


@pytest.mark.asyncio
async def test_media_to_part_transcribes_audio_locally() -> None:
    media = UploadFile(
        filename="voice.ogg",
        file=BytesIO(b"fake-ogg-data"),
        headers={"content-type": "audio/ogg"},
    )

    with patch(
        "app.chat.media.whisper_transcribe",
        new=AsyncMock(
            return_value="Проверка голосового сообщения"
        ),
    ) as transcribe_mock:
        result = await media_to_part(media)

    assert result == {
        "type": "text",
        "text": (
            "[Распознанный текст голосового сообщения]\n"
            "Проверка голосового сообщения"
        ),
    }

    transcribe_mock.assert_awaited_once_with(
        b"fake-ogg-data",
        "voice.ogg",
    )