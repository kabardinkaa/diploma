from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.chat.repositories.postgres_repo import (
    PostgresChatRepository,
    SCHEMA_SQL,
    create_postgres_pool,
)
from app.core.exceptions import DatabaseInfrastructureError


def test_postgres_repository_exposes_required_contract() -> None:
    required_methods = [
        "create_chat",
        "get_chat",
        "append_message",
        "list_messages",
        "soft_delete_messages",
        "save_feedback",
        "admin_stats",
        "list_admin_users",
        "create_broadcast",
        "list_pending_broadcasts",
        "set_handoff_status",
        "list_active_prompts",
    ]

    for method in required_methods:
        assert hasattr(PostgresChatRepository, method)


def test_postgres_schema_contains_required_tables() -> None:
    for table in (
        "chats",
        "chat_messages",
        "feedback",
        "broadcast_tasks",
        "system_prompts",
    ):
        assert table in SCHEMA_SQL


class FakePool:
    def __init__(self, connection) -> None:
        self.connection = connection
        self.acquire_count = 0

    @asynccontextmanager
    async def acquire(self, *, timeout: float):
        assert timeout > 0
        self.acquire_count += 1
        yield self.connection


@pytest.mark.asyncio
async def test_schema_bootstrap_runs_once_and_operations_reuse_pool() -> None:
    connection = SimpleNamespace(
        execute=AsyncMock(),
        fetchval=AsyncMock(side_effect=[1, 2, 3]),
    )
    pool = FakePool(connection)
    repository = PostgresChatRepository(pool)

    await repository.initialize()
    await repository.initialize()
    first = await repository._run(lambda conn: conn.fetchval("SELECT 2"))
    second = await repository._run(lambda conn: conn.fetchval("SELECT 3"))

    assert (first, second) == (2, 3)
    assert pool.acquire_count == 3
    assert connection.execute.await_count == 1


@pytest.mark.asyncio
async def test_pool_acquire_timeout_is_safe_domain_error() -> None:
    class TimeoutPool:
        @asynccontextmanager
        async def acquire(self, *, timeout: float):
            raise TimeoutError("postgresql://user:secret@postgres/diploma")
            yield  # pragma: no cover

    repository = PostgresChatRepository(TimeoutPool())

    with pytest.raises(DatabaseInfrastructureError) as raised:
        await repository.initialize()

    assert "secret" not in raised.value.message


@pytest.mark.asyncio
async def test_pool_factory_uses_configured_bounds(monkeypatch) -> None:
    expected_pool = object()
    create_pool = AsyncMock(return_value=expected_pool)
    monkeypatch.setattr(
        "app.chat.repositories.postgres_repo.asyncpg.create_pool",
        create_pool,
    )
    settings = SimpleNamespace(
        database_url="postgresql://user:secret@postgres/diploma",
        db_pool_min_size=2,
        db_pool_max_size=7,
    )

    assert await create_postgres_pool(settings) is expected_pool
    create_pool.assert_awaited_once_with(
        dsn=settings.database_url,
        min_size=2,
        max_size=7,
        timeout=5.0,
        command_timeout=30.0,
    )
