import asyncio
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.admin.deps import require_admin
from app.core.exceptions import InvalidDocumentError, SafeInputError
from app.core.uploads import UPLOAD_CHUNK_SIZE, read_upload_limited
from app.routers import documents


def make_client(
    tmp_path: Path,
    monkeypatch,
    *,
    upload_limit: int = 1024,
    reindex_max_files: int = 10,
    reindex_max_total_bytes: int = 4096,
) -> tuple[TestClient, AsyncMock, AsyncMock]:
    api = FastAPI()
    api.include_router(documents.router)
    api.dependency_overrides[require_admin] = lambda: None

    @api.exception_handler(SafeInputError)
    async def safe_input_handler(_, exc: SafeInputError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    settings = SimpleNamespace(
        rag_data_dir=tmp_path,
        document_upload_max_bytes=upload_limit,
        document_archive_max_entries=100,
        document_archive_max_uncompressed_bytes=4096,
        document_archive_max_compression_ratio=20.0,
        reindex_max_files=reindex_max_files,
        reindex_max_total_bytes=reindex_max_total_bytes,
    )
    monkeypatch.setattr(documents, "get_settings", lambda: settings)
    parse = AsyncMock()
    monkeypatch.setattr(documents, "_validate_with_parser", parse)
    ingest = AsyncMock()

    async def ingest_and_release(path: Path, root: Path) -> None:
        try:
            await ingest(path, root)
        finally:
            documents._release_ingestion()

    monkeypatch.setattr(documents, "_ingest_uploaded", ingest_and_release)
    return TestClient(api), parse, ingest


def test_upload_accepts_file_exactly_at_limit(tmp_path, monkeypatch) -> None:
    payload = b"# VPN\nx"
    client, parse, ingest = make_client(
        tmp_path,
        monkeypatch,
        upload_limit=len(payload),
    )

    response = client.post(
        "/documents/upload",
        files={"file": ("guide.md", payload, "text/markdown")},
        data={"category": "vpn"},
    )

    assert response.status_code == 202
    assert (tmp_path / "vpn" / "guide.md").read_bytes() == payload
    parse.assert_awaited_once()
    ingest.assert_awaited_once()


def test_upload_rejects_file_above_limit_without_partial_file(
    tmp_path,
    monkeypatch,
) -> None:
    client, _, _ = make_client(tmp_path, monkeypatch, upload_limit=8)

    response = client.post(
        "/documents/upload",
        files={"file": ("guide.md", b"x" * 9, "text/markdown")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert not list(tmp_path.rglob("*.md"))
    assert not list(tmp_path.rglob(".upload-*"))


@pytest.mark.asyncio
async def test_chunked_upload_above_limit_stops_at_hard_limit() -> None:
    payload_size = UPLOAD_CHUNK_SIZE + 1
    upload = UploadFile(
        filename="guide.md",
        file=BytesIO(b"a" * payload_size),
        size=None,
    )

    with pytest.raises(SafeInputError) as raised:
        await read_upload_limited(upload, max_bytes=UPLOAD_CHUNK_SIZE)

    assert raised.value.code == "payload_too_large"
    assert upload.file.tell() == payload_size


@pytest.mark.parametrize(
    ("filename", "payload", "mime", "expected_status"),
    [
        ("data.json", b"{}", "application/json", 415),
        ("guide.pdf", b"not a pdf", "application/pdf", 422),
        ("guide.pdf", b"%PDF-1.4", "text/plain", 422),
        ("../../secret.md", b"x", "text/markdown", 400),
        (".hidden.md", b"x", "text/markdown", 400),
    ],
)
def test_upload_rejects_unsafe_type_signature_and_filename(
    tmp_path,
    monkeypatch,
    filename,
    payload,
    mime,
    expected_status,
) -> None:
    client, _, _ = make_client(tmp_path, monkeypatch)

    response = client.post(
        "/documents/upload",
        files={"file": (filename, payload, mime)},
    )

    assert response.status_code == expected_status
    assert "error" in response.json()


def test_existing_filename_returns_conflict_without_overwrite(
    tmp_path,
    monkeypatch,
) -> None:
    existing = tmp_path / "uploads" / "guide.md"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"original")
    client, _, ingest = make_client(tmp_path, monkeypatch)

    response = client.post(
        "/documents/upload",
        files={"file": ("guide.md", b"replacement", "text/markdown")},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "file_conflict"
    assert existing.read_bytes() == b"original"
    ingest.assert_not_awaited()


def test_parser_failure_is_safe_and_leaves_no_corpus_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    client, parse, ingest = make_client(tmp_path, monkeypatch)
    parse.side_effect = InvalidDocumentError()

    response = client.post(
        "/documents/upload",
        files={"file": ("broken.md", b"# broken", "text/markdown")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_document"
    assert "secret" not in response.text
    assert not list(tmp_path.rglob("broken.md"))
    assert not list(tmp_path.rglob(".upload-*"))
    ingest.assert_not_awaited()


def test_parallel_reindex_returns_busy(tmp_path, monkeypatch) -> None:
    client, _, _ = make_client(tmp_path, monkeypatch)
    asyncio.run(documents._reserve_ingestion())
    try:
        response = client.post(
            "/documents/reindex",
            json={"mode": "incremental"},
        )
    finally:
        documents._release_ingestion()

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ingestion_busy"


def test_upload_during_full_reindex_returns_busy(tmp_path, monkeypatch) -> None:
    client, _, _ = make_client(tmp_path, monkeypatch)
    asyncio.run(documents._reserve_ingestion())
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("guide.md", b"# VPN", "text/markdown")},
        )
    finally:
        documents._release_ingestion()

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ingestion_busy"
    assert not list(tmp_path.rglob("guide.md"))


def test_reindex_scope_is_bounded(tmp_path, monkeypatch) -> None:
    for name in ("one.md", "two.md"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    client, _, _ = make_client(tmp_path, monkeypatch, reindex_max_files=1)

    response = client.post(
        "/documents/reindex",
        json={"mode": "incremental"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "reindex_limit_exceeded"


def test_reindex_files_requires_non_empty_list(tmp_path, monkeypatch) -> None:
    client, _, _ = make_client(tmp_path, monkeypatch)
    response = client.post(
        "/documents/reindex",
        json={"mode": "files", "files": []},
    )
    assert response.status_code == 422
