from io import BytesIO

import httpx
from aiogram import F, Router
from aiogram.types import Message, PhotoSize

from bot.handlers.commands import get_owner_external_id
from bot.handlers.text import send_backend_error
from bot.config import get_bot_settings
from bot.keyboards.feedback import feedback_kb
from bot.services.backend_client import BackendClient
from bot.services.streaming import stream_to_chat


router = Router()

SUPPORTED_DOCUMENT_EXTENSIONS = (".pdf", ".docx")


class TelegramMediaTooLargeError(RuntimeError):
    pass


class LimitedBytesIO(BytesIO):
    def __init__(self, max_bytes: int) -> None:
        super().__init__()
        self.max_bytes = max_bytes

    def write(self, data: bytes) -> int:
        if self.tell() + len(data) > self.max_bytes:
            raise TelegramMediaTooLargeError()
        return super().write(data)


def _size_label(max_bytes: int) -> str:
    return f"{max_bytes / (1024 * 1024):g} МБ"


async def _answer_too_large(message: Message, max_bytes: int) -> None:
    await message.answer(
        f"Файл слишком большой. Максимальный размер — {_size_label(max_bytes)}."
    )


async def download_telegram_file(
    message: Message,
    file_id: str,
    *,
    max_bytes: int,
) -> bytes:
    telegram_file = await message.bot.get_file(file_id)

    if not telegram_file.file_path:
        raise RuntimeError("Telegram не вернул путь к файлу")

    buffer = LimitedBytesIO(max_bytes)

    await message.bot.download_file(
        telegram_file.file_path,
        destination=buffer,
    )

    return buffer.getvalue()


async def get_chat_id(
    message: Message,
    backend: BackendClient,
):
    return await backend.get_or_create_chat(
        owner_external_id=get_owner_external_id(message),
        interface="telegram",
    )


def select_photo(
    photos: list[PhotoSize],
    *,
    max_bytes: int,
) -> PhotoSize | None:
    suitable = [
        photo
        for photo in photos
        if photo.file_size is None
        or photo.file_size <= max_bytes
    ]

    if not suitable:
        return None

    return max(
        suitable,
        key=lambda photo: photo.width * photo.height,
    )


@router.message(F.photo)
async def handle_photo(
    message: Message,
    backend: BackendClient,
) -> None:
    max_bytes = get_bot_settings().photo_max_bytes
    photo = select_photo(message.photo, max_bytes=max_bytes)

    if photo is None:
        await message.answer(
            f"Фото слишком большое. Максимальный размер — {_size_label(max_bytes)}."
        )
        return

    try:
        media_bytes = await download_telegram_file(
            message,
            photo.file_id,
            max_bytes=max_bytes,
        )

        chat_id = await get_chat_id(message, backend)

        tokens = backend.send_message(
            chat_id=chat_id,
            content=message.caption or "Опиши изображение",
            media=media_bytes,
            mime="image/jpeg",
            filename="photo.jpg",
        )

        result = await stream_to_chat(message, tokens)
        if result and backend.last_message_id is not None:
            await result.message.edit_reply_markup(
                reply_markup=feedback_kb(backend.last_message_id)
            )

    except TelegramMediaTooLargeError:
        await _answer_too_large(message, max_bytes)
    except (
        httpx.HTTPError,
        RuntimeError,
    ) as exc:
        await send_backend_error(message, exc)


@router.message(F.voice)
async def handle_voice(
    message: Message,
    backend: BackendClient,
) -> None:
    if message.voice is None:
        return

    max_bytes = get_bot_settings().media_max_bytes
    if message.voice.file_size is not None and message.voice.file_size > max_bytes:
        await _answer_too_large(message, max_bytes)
        return

    try:
        media_bytes = await download_telegram_file(
            message,
            message.voice.file_id,
            max_bytes=max_bytes,
        )

        chat_id = await get_chat_id(message, backend)

        tokens = backend.send_message(
            chat_id=chat_id,
            content="Обработай голосовое сообщение",
            media=media_bytes,
            mime="audio/ogg",
            filename="voice.ogg",
        )

        result = await stream_to_chat(message, tokens)
        if result and backend.last_message_id is not None:
            await result.message.edit_reply_markup(
                reply_markup=feedback_kb(backend.last_message_id)
            )

    except TelegramMediaTooLargeError:
        await _answer_too_large(message, max_bytes)
    except (
        httpx.HTTPError,
        RuntimeError,
    ) as exc:
        await send_backend_error(message, exc)


@router.message(F.audio)
async def handle_audio(
    message: Message,
    backend: BackendClient,
) -> None:
    if message.audio is None:
        return

    max_bytes = get_bot_settings().media_max_bytes
    if message.audio.file_size is not None and message.audio.file_size > max_bytes:
        await _answer_too_large(message, max_bytes)
        return

    try:
        media_bytes = await download_telegram_file(
            message,
            message.audio.file_id,
            max_bytes=max_bytes,
        )

        chat_id = await get_chat_id(message, backend)

        tokens = backend.send_message(
            chat_id=chat_id,
            content=message.caption or "Обработай аудиофайл",
            media=media_bytes,
            mime=message.audio.mime_type or "audio/mpeg",
            filename=message.audio.file_name or "audio.mp3",
        )

        result = await stream_to_chat(message, tokens)
        if result and backend.last_message_id is not None:
            await result.message.edit_reply_markup(
                reply_markup=feedback_kb(backend.last_message_id)
            )

    except TelegramMediaTooLargeError:
        await _answer_too_large(message, max_bytes)
    except (
        httpx.HTTPError,
        RuntimeError,
    ) as exc:
        await send_backend_error(message, exc)


@router.message(F.document)
async def handle_document(
    message: Message,
    backend: BackendClient,
) -> None:
    document = message.document

    if document is None:
        return

    filename = document.file_name or ""
    normalized_filename = filename.lower()

    if not normalized_filename.endswith(
        SUPPORTED_DOCUMENT_EXTENSIONS
    ):
        await message.answer(
            "Поддерживаются только PDF и DOCX."
        )
        return

    max_bytes = get_bot_settings().media_max_bytes
    if (
        document.file_size is not None
        and document.file_size > max_bytes
    ):
        await _answer_too_large(message, max_bytes)
        return

    if normalized_filename.endswith(".pdf"):
        mime = "application/pdf"
    else:
        mime = (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        )

    try:
        media_bytes = await download_telegram_file(
            message,
            document.file_id,
            max_bytes=max_bytes,
        )

        chat_id = await get_chat_id(message, backend)

        tokens = backend.send_message(
            chat_id=chat_id,
            content=message.caption or "Изучи документ",
            media=media_bytes,
            mime=document.mime_type or mime,
            filename=filename,
        )

        result = await stream_to_chat(message, tokens)
        if result and backend.last_message_id is not None:
            await result.message.edit_reply_markup(
                reply_markup=feedback_kb(backend.last_message_id)
            )

    except TelegramMediaTooLargeError:
        await _answer_too_large(message, max_bytes)
    except (
        httpx.HTTPError,
        RuntimeError,
    ) as exc:
        await send_backend_error(message, exc)
