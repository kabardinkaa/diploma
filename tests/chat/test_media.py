from io import BytesIO

import pytest
from docx import Document
from fastapi import UploadFile
from pypdf import PdfWriter

from app.chat.media import media_to_part
from app.core.exceptions import InvalidDocumentError, PayloadTooLargeError


@pytest.mark.asyncio
async def test_media_to_part_returns_image_url_for_png() -> None:
    media = UploadFile(
        filename="sample.png",
        file=BytesIO(b"\x89PNG\r\n\x1a\nsample"),
        headers={"content-type": "image/png"},
    )

    result = await media_to_part(media)

    assert result["type"] == "image_url"
    assert result["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.asyncio
async def test_media_to_part_returns_text_for_pdf() -> None:
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(buffer)

    media = UploadFile(
        filename="sample.pdf",
        file=BytesIO(buffer.getvalue()),
        headers={"content-type": "application/pdf"},
    )

    result = await media_to_part(media)

    assert result["type"] == "text"
    assert result["text"].startswith(
        "[Содержимое PDF-документа]\n"
    )


@pytest.mark.asyncio
async def test_media_to_part_returns_text_for_docx() -> None:
    buffer = BytesIO()
    document = Document()
    document.add_paragraph("Тестовый документ")
    document.save(buffer)

    media = UploadFile(
        filename="sample.docx",
        file=BytesIO(buffer.getvalue()),
        headers={
            "content-type": (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            )
        },
    )

    result = await media_to_part(media)

    assert result["type"] == "text"
    assert "[Содержимое DOCX-документа]" in result["text"]
    assert "Тестовый документ" in result["text"]


@pytest.mark.asyncio
async def test_media_exactly_at_limit_is_accepted() -> None:
    payload = b"\x89PNG\r\n\x1a\n"
    media = UploadFile(
        filename="sample.png",
        file=BytesIO(payload),
        headers={"content-type": "image/png"},
    )

    result = await media_to_part(media, max_bytes=len(payload))

    assert result["type"] == "image_url"


@pytest.mark.asyncio
async def test_media_above_limit_is_rejected() -> None:
    payload = b"\x89PNG\r\n\x1a\nextra"
    media = UploadFile(
        filename="sample.png",
        file=BytesIO(payload),
        headers={"content-type": "image/png"},
    )

    with pytest.raises(PayloadTooLargeError):
        await media_to_part(media, max_bytes=len(payload) - 1)


@pytest.mark.asyncio
async def test_media_mime_content_mismatch_is_rejected() -> None:
    media = UploadFile(
        filename="sample.png",
        file=BytesIO(b"not a png"),
        headers={"content-type": "image/png"},
    )

    with pytest.raises(InvalidDocumentError):
        await media_to_part(media)


@pytest.mark.asyncio
async def test_pdf_parser_failure_is_controlled() -> None:
    media = UploadFile(
        filename="broken.pdf",
        file=BytesIO(b"%PDF-invalid"),
        headers={"content-type": "application/pdf"},
    )

    with pytest.raises(InvalidDocumentError):
        await media_to_part(media)
