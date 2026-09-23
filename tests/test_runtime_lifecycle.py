from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from app import main


@pytest.mark.asyncio
async def test_lifespan_creates_shared_resources_once_and_closes_them(monkeypatch) -> None:
    settings = SimpleNamespace(
        rag_tracing_enabled=False,
        llm=SimpleNamespace(
            api_key=SecretStr("test-key"),
            request_timeout=1,
            max_retries=0,
            base_url=None,
        ),
        llm_cache_max_entries=4,
        llm_cache_ttl_seconds=30,
        llm_cache_enabled=True,
        chat_repository="postgres",
        chat_storage_dir="unused",
    )
    pool = SimpleNamespace(close=AsyncMock(), terminate=Mock())
    openai_client = SimpleNamespace(close=AsyncMock())
    rag_service = SimpleNamespace(
        async_client=object(),
        build=AsyncMock(),
        close=AsyncMock(),
    )
    repository = SimpleNamespace(initialize=AsyncMock())
    create_pool = AsyncMock(return_value=pool)
    flush_tracing = Mock(return_value=True)
    shutdown_tracing = Mock(return_value=True)

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "setup_tracing", lambda **_kwargs: None)
    monkeypatch.setattr(main, "force_flush_tracing", flush_tracing)
    monkeypatch.setattr(main, "shutdown_tracing", shutdown_tracing)
    monkeypatch.setattr(main, "AsyncOpenAI", lambda **_kwargs: openai_client)
    monkeypatch.setattr(main, "create_postgres_pool", create_pool)
    monkeypatch.setattr(
        main,
        "PostgresChatRepository",
        lambda selected_pool: repository if selected_pool is pool else None,
    )
    monkeypatch.setattr(
        main,
        "RAGService",
        lambda _settings, *, openai_client: rag_service,
    )

    @asynccontextmanager
    async def fake_agent_lifespan(_settings, *, rag_service):
        yield "agent"

    monkeypatch.setattr(main, "agent_lifespan", fake_agent_lifespan)

    api = FastAPI()
    async with main.lifespan(api):
        assert api.state.db_pool is pool
        assert api.state.chat_repository is repository
        assert api.state.openai is openai_client
        assert api.state.rag_service is rag_service
        assert api.state.persistent_agent == "agent"

    create_pool.assert_awaited_once_with(settings)
    repository.initialize.assert_awaited_once_with()
    rag_service.build.assert_awaited_once_with()
    rag_service.close.assert_awaited_once_with()
    openai_client.close.assert_awaited_once_with()
    pool.close.assert_awaited_once_with()
    pool.terminate.assert_not_called()
    flush_tracing.assert_called_once_with(5_000)
    shutdown_tracing.assert_called_once_with()
