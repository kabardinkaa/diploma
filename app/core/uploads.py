from __future__ import annotations

import os
import tempfile
import unicodedata
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath

from fastapi import UploadFile

from app.core.exceptions import (
    InvalidDocumentError,
    PayloadTooLargeError,
    UnsupportedFileTypeError,
    UnsafeFilenameError,
)


UPLOAD_CHUNK_SIZE = 64 * 1024
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_DOCUMENT_MIME_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
    ".html": {"text/html", "text/plain", "application/octet-stream"},
    ".htm": {"text/html", "text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
}


def normalize_safe_filename(filename: str) -> str:
    normalized = unicodedata.normalize("NFKC", filename).strip()
    path = Path(normalized)
    stem = path.stem.lower()
    if (
        not normalized
        or normalized in {".", ".."}
        or normalized.startswith(".")
        or normalized.endswith((".", " "))
        or len(normalized) > 255
        or path.name != normalized
        or "/" in normalized
        or "\\" in normalized
        or ":" in normalized
        or stem in _WINDOWS_RESERVED_NAMES
        or any(ord(character) < 32 for character in normalized)
    ):
        raise UnsafeFilenameError()
    return normalized


async def read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    declared_size = upload.size
    if declared_size is not None and declared_size > max_bytes:
        raise PayloadTooLargeError()

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLargeError()
        chunks.append(chunk)
    return b"".join(chunks)


async def save_upload_limited(
    upload: UploadFile,
    directory: Path,
    *,
    suffix: str,
    max_bytes: int,
) -> Path:
    declared_size = upload.size
    if declared_size is not None and declared_size > max_bytes:
        raise PayloadTooLargeError()

    directory.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".upload-",
        suffix=suffix,
        dir=directory,
    )
    temporary_path = Path(temporary_name)
    total = 0
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise PayloadTooLargeError()
                output.write(chunk)
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def publish_without_overwrite(temporary_path: Path, target: Path) -> None:
    try:
        os.link(temporary_path, target)
    except FileExistsError:
        from app.core.exceptions import FileConflictError

        raise FileConflictError() from None
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_mime(extension: str, content_type: str | None) -> None:
    normalized_mime = (content_type or "application/octet-stream").split(";", 1)[0]
    normalized_mime = normalized_mime.strip().lower()
    allowed = _DOCUMENT_MIME_TYPES.get(extension)
    if allowed is None:
        raise UnsupportedFileTypeError()
    if normalized_mime not in allowed:
        raise InvalidDocumentError()


def _validate_archive_member(name: str) -> None:
    normalized = name.replace("\\", "/")
    member_path = PurePosixPath(normalized)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise InvalidDocumentError()


def validate_docx_archive(
    source: Path | BytesIO,
    *,
    max_entries: int,
    max_uncompressed_bytes: int,
    max_compression_ratio: float,
) -> None:
    try:
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries:
                raise InvalidDocumentError()

            total_uncompressed = 0
            names: set[str] = set()
            for entry in entries:
                _validate_archive_member(entry.filename)
                names.add(entry.filename.replace("\\", "/"))
                total_uncompressed += entry.file_size
                if total_uncompressed > max_uncompressed_bytes:
                    raise InvalidDocumentError()
                if entry.file_size:
                    if entry.compress_size == 0:
                        raise InvalidDocumentError()
                    if entry.file_size / entry.compress_size > max_compression_ratio:
                        raise InvalidDocumentError()

            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise InvalidDocumentError()
    except (OSError, zipfile.BadZipFile, RuntimeError):
        raise InvalidDocumentError() from None


def validate_document_file(
    path: Path,
    *,
    extension: str,
    content_type: str | None,
    archive_max_entries: int,
    archive_max_uncompressed_bytes: int,
    archive_max_compression_ratio: float,
) -> None:
    _validate_mime(extension, content_type)
    with path.open("rb") as source:
        prefix = source.read(64 * 1024)
    if not prefix:
        raise InvalidDocumentError()

    if extension == ".pdf":
        if not prefix.startswith(b"%PDF-"):
            raise InvalidDocumentError()
        return

    if extension == ".docx":
        if not prefix.startswith(b"PK"):
            raise InvalidDocumentError()
        validate_docx_archive(
            path,
            max_entries=archive_max_entries,
            max_uncompressed_bytes=archive_max_uncompressed_bytes,
            max_compression_ratio=archive_max_compression_ratio,
        )
        return

    try:
        text = prefix.decode("utf-8")
    except UnicodeDecodeError:
        raise InvalidDocumentError() from None
    if "\x00" in text:
        raise InvalidDocumentError()
    if extension in {".html", ".htm"}:
        lowered = text.lstrip("\ufeff \t\r\n").lower()
        markers = ("<!doctype html", "<html", "<head", "<body", "<section", "<article")
        if not lowered.startswith(markers):
            raise InvalidDocumentError()


def validate_media_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: str | None,
    archive_max_entries: int,
    archive_max_uncompressed_bytes: int,
    archive_max_compression_ratio: float,
) -> None:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    extension = Path(filename).suffix.lower()
    if not data:
        raise InvalidDocumentError()

    image_signatures = {
        "image/png": lambda value: value.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": lambda value: value.startswith(b"\xff\xd8\xff"),
        "image/webp": lambda value: value.startswith(b"RIFF") and value[8:12] == b"WEBP",
    }
    if mime in image_signatures:
        if not image_signatures[mime](data):
            raise InvalidDocumentError()
        return

    audio_signatures = {
        "audio/ogg": lambda value: value.startswith(b"OggS"),
        "application/ogg": lambda value: value.startswith(b"OggS"),
        "audio/mpeg": lambda value: value.startswith(b"ID3") or value[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"},
        "audio/wav": lambda value: value.startswith(b"RIFF") and value[8:12] == b"WAVE",
        "audio/x-wav": lambda value: value.startswith(b"RIFF") and value[8:12] == b"WAVE",
        "audio/mp4": lambda value: len(value) >= 12 and value[4:8] == b"ftyp",
        "audio/x-m4a": lambda value: len(value) >= 12 and value[4:8] == b"ftyp",
    }
    if mime in audio_signatures:
        if not audio_signatures[mime](data):
            raise InvalidDocumentError()
        return

    if mime == "application/pdf" and extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise InvalidDocumentError()
        return

    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if mime == docx_mime and extension == ".docx":
        if not data.startswith(b"PK"):
            raise InvalidDocumentError()
        validate_docx_archive(
            BytesIO(data),
            max_entries=archive_max_entries,
            max_uncompressed_bytes=archive_max_uncompressed_bytes,
            max_compression_ratio=archive_max_compression_ratio,
        )
        return

    raise UnsupportedFileTypeError()
