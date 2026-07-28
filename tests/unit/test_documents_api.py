from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import documents


def make_client(tmp_path: Path, monkeypatch) -> tuple[TestClient, AsyncMock]:
    api = FastAPI()
    api.include_router(documents.router)
    settings = SimpleNamespace(rag_data_dir=tmp_path)
    monkeypatch.setattr(documents, "get_settings", lambda: settings)
    ingest = AsyncMock()
    monkeypatch.setattr(documents, "_ingest_uploaded", ingest)
    return TestClient(api), ingest


def test_upload_returns_202_and_rejects_unsafe_paths(tmp_path, monkeypatch) -> None:
    client, ingest = make_client(tmp_path, monkeypatch)
    accepted = client.post(
        "/documents/upload",
        files={"file": ("guide.md", b"# VPN", "text/markdown")},
        data={"category": "vpn"},
    )
    traversal = client.post(
        "/documents/upload",
        files={"file": ("../../secret.md", b"x", "text/markdown")},
        data={"category": "vpn"},
    )
    unsupported = client.post(
        "/documents/upload",
        files={"file": ("data.json", b"{}", "application/json")},
    )

    assert accepted.status_code == 202
    assert accepted.json()["status"] == "accepted"
    assert (tmp_path / "vpn" / "guide.md").exists()
    assert ingest.await_count == 1
    assert traversal.status_code == 400
    assert unsupported.status_code == 415


def test_reindex_files_requires_non_empty_list(tmp_path, monkeypatch) -> None:
    client, _ = make_client(tmp_path, monkeypatch)
    response = client.post(
        "/documents/reindex",
        json={"mode": "files", "files": []},
    )
    assert response.status_code == 422
