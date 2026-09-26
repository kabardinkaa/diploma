from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from app.core.config import LLMSettings, Settings, get_settings
from app.chat.deps import get_chat_service
from app.chat.routes import router as chat_history_router
from app.core.exceptions import PublicGenerationControlError
from app.core.sse import public_stream_error
from app.deps.providers import get_llm_service
from app.routers import chat, health, rag
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Message
from app.security.public_budget import PublicGenerationBudget
from app.security.rate_limit import PublicRateLimitMiddleware
from app.services.llm import LLMService
from app.services.retention import RetentionManager


class FakeLLMService:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ChatResponse(content="ok", model=request.model or "server-model")

    async def stream(self, request):
        self.calls += 1
        yield ChatDelta(content="ok")

    async def batch(self, requests):
        self.calls += len(requests)
        return [ChatResponse(content="ok", model="server-model") for _ in requests]


class FakeRAGService:
    def __init__(self) -> None:
        self.calls = 0

    async def answer(self, question: str) -> dict:
        self.calls += 1
        return {
            "answer": f"answer: {question}",
            "sources": [],
            "top_score": 0.0,
            "confident": False,
        }


def _budget_api(requests: int = 2):
    api = FastAPI()
    budget = PublicGenerationBudget(
        enabled=True,
        requests=requests,
        window_seconds=60,
    )
    llm = FakeLLMService()
    rag_service = FakeRAGService()
    api.state.public_generation_budget = budget
    api.state.rag_service = rag_service
    api.include_router(health.router)
    api.include_router(chat.router)
    api.include_router(rag.router)
    api.dependency_overrides[get_llm_service] = lambda: llm
    api.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        environment="dev",
        llm=SimpleNamespace(default_model="server-model"),
        chat_max_tokens=256,
        admin_token=SecretStr("admin-secret"),
    )

    @api.exception_handler(PublicGenerationControlError)
    async def generation_error(
        _: Request,
        exc: PublicGenerationControlError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    return api, budget, llm, rag_service


@pytest.mark.asyncio
async def test_public_budget_exhaustion_and_window_reset() -> None:
    now = [100.0]
    budget = PublicGenerationBudget(
        enabled=True,
        requests=2,
        window_seconds=10,
        clock=lambda: now[0],
    )

    await budget.consume()
    await budget.consume()
    with pytest.raises(PublicGenerationControlError) as exc_info:
        await budget.consume()
    assert exc_info.value.code == "public_quota_exhausted"

    now[0] = 111.0
    await budget.consume()
    assert budget.used == 1


@pytest.mark.asyncio
async def test_public_budget_is_atomic_for_parallel_requests() -> None:
    budget = PublicGenerationBudget(enabled=True, requests=3, window_seconds=60)

    results = await asyncio.gather(
        *(budget.consume() for _ in range(10)),
        return_exceptions=True,
    )

    assert len([item for item in results if item is None]) == 3
    assert len(
        [item for item in results if isinstance(item, PublicGenerationControlError)]
    ) == 7


def test_public_endpoints_share_budget_and_health_does_not_consume_it() -> None:
    api, budget, llm, rag_service = _budget_api(requests=2)
    client = TestClient(api)

    assert client.get("/health/live").status_code == 200
    assert budget.used == 0
    assert client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 9999,
            "public_generation_budget_requests": 999999,
        },
    ).status_code == 200
    assert client.post("/rag/query", json={"question": "unknown"}).status_code == 200

    exhausted = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "again"}]},
    )
    assert exhausted.status_code == 429
    assert exhausted.json()["error"]["code"] == "public_quota_exhausted"
    assert llm.calls == 1
    assert rag_service.calls == 1


def test_telegram_end_user_generation_shares_public_budget() -> None:
    class FakeHistoryService:
        last_sources: list[dict] = []
        last_rag_meta: dict = {}
        last_assistant_message_id = None
        llm_service = SimpleNamespace(openai=AsyncMock())
        calls = 0

        async def check_user_content(self, _: str) -> None:
            return None

        async def send_message(self, **_):
            self.calls += 1
            yield "ok"

    api, budget, _, _ = _budget_api(requests=1)
    service = FakeHistoryService()
    api.include_router(chat_history_router)
    api.dependency_overrides[get_chat_service] = lambda: service
    client = TestClient(api)

    assert client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "web"}]},
    ).status_code == 200
    response = client.post(
        f"/chats/{uuid4()}/messages",
        files={"content": (None, "telegram")},
    )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "public_quota_exhausted"
    assert budget.used == 1
    assert service.calls == 0


def test_public_generation_can_be_disabled_without_calling_provider() -> None:
    api, _, llm, _ = _budget_api(requests=2)
    api.state.public_generation_budget = PublicGenerationBudget(
        enabled=False,
        requests=2,
        window_seconds=60,
    )

    response = TestClient(api).post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "public_generation_disabled"
    assert llm.calls == 0


def test_admin_batch_does_not_consume_public_budget() -> None:
    api, budget, llm, _ = _budget_api(requests=1)
    response = TestClient(api).post(
        "/chat/batch",
        headers={"X-Admin-Token": "admin-secret"},
        json={"requests": [{"messages": [{"role": "user", "content": "x"}]}]},
    )

    assert response.status_code == 200
    assert budget.used == 0
    assert llm.calls == 1


def test_stream_quota_error_is_safe_and_releases_concurrency_slot() -> None:
    api, budget, llm, _ = _budget_api(requests=1)
    limited = PublicRateLimitMiddleware(
        api,
        requests=10,
        window_seconds=60,
        max_concurrent=1,
        paths={("POST", "/chat"), ("POST", "/chat/stream")},
    )
    client = TestClient(limited)
    payload = {"messages": [{"role": "user", "content": "hello"}]}

    assert client.post("/chat", json=payload).status_code == 200
    response = client.post("/chat/stream", json=payload)

    assert response.status_code == 200
    assert 'event: error' in response.text
    assert '"code":"public_quota_exhausted"' in response.text
    assert 'event: done' in response.text
    assert limited.limiter.active_requests == 0
    assert budget.used == 1
    assert llm.calls == 1


def _public_settings(**overrides) -> Settings:
    values = {
        "APP_ENV": "public",
        "PUBLIC_DOMAIN": "demo.example.com",
        "PUBLIC_PROXY_IP": "172.31.250.10",
        "TRUSTED_PROXY_CIDRS": "172.31.250.10/32",
        "PUBLIC_SESSION_SECRET": "public-session-secret-with-at-least-32-chars",
        "ADMIN_TOKEN": "safe-admin-token",
        "INTERNAL_TOKEN": "safe-internal-token",
        "QDRANT_API_KEY": "safe-qdrant-token",
        "CORS_ORIGINS": ["https://demo.example.com"],
        "PUBLIC_GENERATION_BUDGET_REQUESTS": 100,
        "PUBLIC_DATA_RETENTION_DAYS": 30,
        "TRACING_CAPTURE_CONTENT": False,
        "LOG_PROMPT_PREVIEW_ENABLED": False,
        "llm": LLMSettings(
            _env_file=None,
            OPENROUTER_API_KEY="safe-provider-token",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
        ),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize(
    "origins",
    [
        ["*"],
        [],
        ["http://demo.example.com"],
        ["https://localhost:3000"],
        ["https://user:password@demo.example.com"],
        ["https://demo.example.com/"],
    ],
)
def test_public_cors_rejects_unsafe_origins(origins: list[str]) -> None:
    with pytest.raises(ValueError, match="CORS_ORIGINS"):
        _public_settings(CORS_ORIGINS=origins)


def test_public_cors_accepts_explicit_https_allowlist() -> None:
    settings = _public_settings(
        CORS_ORIGINS=["https://demo.example.com", "https://review.example.org"]
    )
    assert settings.cors_origins == [
        "https://demo.example.com",
        "https://review.example.org",
    ]


def test_dev_keeps_wildcard_cors_and_unlimited_budget_defaults() -> None:
    settings = Settings(_env_file=None, APP_ENV="dev")
    assert settings.cors_origins == ["*"]
    assert settings.public_generation_budget_requests == 0
    assert settings.public_data_retention_days == 0


def test_public_config_requires_budget_retention_and_private_observability() -> None:
    for override, setting in (
        ({"PUBLIC_GENERATION_BUDGET_REQUESTS": 0}, "PUBLIC_GENERATION"),
        ({"PUBLIC_DATA_RETENTION_DAYS": 0}, "PUBLIC_DATA_RETENTION_DAYS"),
        ({"TRACING_CAPTURE_CONTENT": True}, "TRACING_CAPTURE_CONTENT"),
        ({"LOG_PROMPT_PREVIEW_ENABLED": True}, "LOG_PROMPT_PREVIEW_ENABLED"),
    ):
        with pytest.raises(ValueError, match=setting):
            _public_settings(**override)


def test_config_error_does_not_echo_secret_values() -> None:
    secret = "unique-provider-secret-must-not-leak"
    with pytest.raises(ValueError) as exc_info:
        _public_settings(
            CORS_ORIGINS=["*"],
            llm=LLMSettings(_env_file=None, OPENROUTER_API_KEY=secret),
        )
    assert secret not in str(exc_info.value)


class RecordingRepository:
    def __init__(self) -> None:
        self.protected_calls: list[set[str]] = []

    async def cleanup_expired(self, *, before, protected_chat_ids) -> int:
        self.protected_calls.append(set(protected_chat_ids))
        return 0 if "active-chat" in protected_chat_ids else 1


class AsyncContext:
    def __init__(self, value=None) -> None:
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_):
        return False


class RecordingCheckpointConnection:
    def __init__(self) -> None:
        self.deletes: list[tuple[str, list[str]]] = []

    def transaction(self) -> AsyncContext:
        return AsyncContext()

    async def fetch(self, *_):
        return [
            {"thread_id": "active-thread"},
            {"thread_id": "expired-thread"},
        ]

    async def execute(self, query: str, thread_ids: list[str]):
        table = query.split("DELETE FROM ", 1)[1].split(" ", 1)[0]
        self.deletes.append((table, thread_ids))
        return "DELETE 1"


class RecordingCheckpointPool:
    def __init__(self, connection: RecordingCheckpointConnection) -> None:
        self.connection = connection

    def acquire(self, *, timeout: float) -> AsyncContext:
        assert timeout == 5.0
        return AsyncContext(self.connection)


@pytest.mark.asyncio
async def test_retention_preserves_active_session_then_cleans_it_later() -> None:
    repository = RecordingRepository()
    manager = RetentionManager(
        repository=repository,
        postgres_pool=None,
        retention_days=30,
        cleanup_interval_seconds=3600,
    )

    async with manager.protect("chat:active-chat"):
        protected_result = await manager.run_once()
    expired_result = await manager.run_once()

    assert repository.protected_calls == [{"active-chat"}, set()]
    assert protected_result["chat_records"] == 0
    assert expired_result["chat_records"] == 1


@pytest.mark.asyncio
async def test_retention_never_deletes_active_agent_checkpoint_thread() -> None:
    repository = RecordingRepository()
    connection = RecordingCheckpointConnection()
    manager = RetentionManager(
        repository=repository,
        postgres_pool=RecordingCheckpointPool(connection),
        retention_days=30,
        cleanup_interval_seconds=3600,
    )

    async with manager.protect("agent:active-thread"):
        result = await manager.run_once()

    assert result["checkpoint_threads"] == 1
    assert connection.deletes == [
        ("checkpoint_writes", ["expired-thread"]),
        ("checkpoint_blobs", ["expired-thread"]),
        ("checkpoints", ["expired-thread"]),
    ]


@pytest.mark.asyncio
async def test_public_logs_omit_prompt_preview_when_disabled(mocker) -> None:
    settings = SimpleNamespace(
        llm=SimpleNamespace(default_model="server-model"),
        log_prompt_preview_enabled=False,
    )
    request = ChatRequest(messages=[Message(role="user", content="private prompt")])
    cache: dict[str, str] = {}
    service = LLMService(openai_client=AsyncMock(), cache=cache, settings=settings)
    cache[service._cache_key(request)] = ChatResponse(
        content="cached",
        model="server-model",
    ).model_dump_json()
    logged = mocker.patch("app.services.llm.logger.info")

    await service.complete(request)

    assert logged.call_args.kwargs["prompt_preview"] is None
    assert "private prompt" not in str(logged.call_args)


def test_public_quota_error_never_exposes_internal_exception_text() -> None:
    payload = public_stream_error(RuntimeError("token=super-secret provider-body"))
    assert payload == {
        "type": "error",
        "code": "stream_error",
        "message": "Stream terminated unexpectedly",
    }
