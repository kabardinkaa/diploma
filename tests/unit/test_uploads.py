from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from app.core.exceptions import InvalidDocumentError, UnsupportedFileTypeError
from app.core.uploads import validate_media_bytes


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _docx_bytes(extra_entries: dict[str, bytes] | None = None) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr("word/document.xml", b"<document/>")
        for name, data in (extra_entries or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _validate(data: bytes, **overrides) -> None:
    options = {
        "filename": "guide.docx",
        "content_type": DOCX_MIME,
        "archive_max_entries": 10,
        "archive_max_uncompressed_bytes": 1024,
        "archive_max_compression_ratio": 100.0,
        **overrides,
    }
    validate_media_bytes(data, **options)


def test_docx_archive_rejects_path_traversal() -> None:
    with pytest.raises(InvalidDocumentError):
        _validate(_docx_bytes({"../payload.bin": b"x"}))


def test_docx_archive_rejects_entry_count_and_uncompressed_size() -> None:
    data = _docx_bytes({"word/extra.xml": b"x" * 20})

    with pytest.raises(InvalidDocumentError):
        _validate(data, archive_max_entries=2)
    with pytest.raises(InvalidDocumentError):
        _validate(data, archive_max_uncompressed_bytes=10)


def test_docx_archive_rejects_extreme_compression_ratio() -> None:
    data = _docx_bytes({"word/large.xml": b"0" * 500})

    with pytest.raises(InvalidDocumentError):
        _validate(data, archive_max_compression_ratio=2.0)


def test_direct_zip_upload_is_not_supported() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_media_bytes(
            _docx_bytes(),
            filename="archive.zip",
            content_type="application/zip",
            archive_max_entries=10,
            archive_max_uncompressed_bytes=1024,
            archive_max_compression_ratio=100.0,
        )
