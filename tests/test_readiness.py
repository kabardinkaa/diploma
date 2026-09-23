from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import health
from app.services.readiness import ReadinessService


def _readiness_client(
    *,
    postgres_error: Exception | None = None,
    qdrant_error: Exception | None = None,
    collections: tuple[str, ...] = ("corporate_rag",),
) -> tuple[TestClient, AsyncMock, AsyncMock]:
    connection = SimpleNamespace(
        fetchval=AsyncMock(return_value=1),
        close=AsyncMock(),
    )
    connect = AsyncMock(return_value=connection)
    if postgres_error is not None:
        connect.side_effect = postgres_error

    qdrant = SimpleNamespace(get_collections=AsyncMock())
    qdrant.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name=name) for name in collections]
    )
    if qdrant_error is not None:
        qdrant.get_collections.side_effect = qdrant_error

    settings = SimpleNamespace(
        database_url="postgresql://user:secret@postgres/diploma",
        rag_production_collection="corporate_rag",
    )
    api = FastAPI()
    api.state.readiness_probe = ReadinessService(
        settings,
        qdrant,
        postgres_connect=connect,
    )
    api.include_router(health.router)
    return TestClient(api), connect, qdrant.get_collections


def test_live_is_independent_from_failed_dependencies() -> None:
    client, connect, qdrant = _readiness_client(
        postgres_error=RuntimeError("postgres secret details"),
        qdrant_error=RuntimeError("qdrant internal URL"),
    )

    assert client.get("/health/live").status_code == 200
    assert client.get("/health").status_code == 200
    connect.assert_not_awaited()
    qdrant.assert_not_awaited()


def test_ready_returns_200_when_required_dependencies_are_available() -> None:
    client, _, _ = _readiness_client()

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "dependencies": {
            "postgres": {"status": "ok"},
            "qdrant": {"status": "ok"},
            "corporate_rag": {"status": "ok"},
        },
    }


def test_ready_returns_safe_503_when_postgres_is_unavailable() -> None:
    client, _, _ = _readiness_client(
        postgres_error=RuntimeError(
            "could not connect to postgresql://user:secret@postgres/diploma"
        )
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"]["postgres"] == {
        "status": "unavailable"
    }
    assert "secret" not in response.text
    assert "postgresql://" not in response.text


def test_ready_returns_safe_503_when_qdrant_is_unavailable() -> None:
    client, _, _ = _readiness_client(
        qdrant_error=RuntimeError("http://qdrant:6333 token=secret")
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"]["qdrant"] == {
        "status": "unavailable"
    }
    assert response.json()["dependencies"]["corporate_rag"] == {
        "status": "unavailable"
    }
    assert "qdrant:6333" not in response.text
    assert "secret" not in response.text


def test_ready_returns_503_when_corporate_collection_is_missing() -> None:
    client, _, _ = _readiness_client(collections=("another_collection",))

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"]["qdrant"] == {"status": "ok"}
    assert response.json()["dependencies"]["corporate_rag"] == {
        "status": "missing"
    }


def test_ready_reuses_lifespan_postgres_pool() -> None:
    connection = SimpleNamespace(fetchval=AsyncMock(return_value=1))

    class Pool:
        acquire_count = 0

        @asynccontextmanager
        async def acquire(self, *, timeout: float):
            self.acquire_count += 1
            yield connection

    pool = Pool()
    connect = AsyncMock()
    qdrant = SimpleNamespace(
        get_collections=AsyncMock(
            return_value=SimpleNamespace(
                collections=[SimpleNamespace(name="corporate_rag")]
            )
        )
    )
    settings = SimpleNamespace(
        database_url="postgresql://redacted",
        rag_production_collection="corporate_rag",
    )
    api = FastAPI()
    api.state.readiness_probe = ReadinessService(
        settings,
        qdrant,
        postgres_pool=pool,
        postgres_connect=connect,
    )
    api.include_router(health.router)

    response = TestClient(api).get("/health/ready")

    assert response.status_code == 200
    assert pool.acquire_count == 1
    connect.assert_not_awaited()
