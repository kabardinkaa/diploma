from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.core.exceptions import SafeInputError
from app.admin.deps import require_internal_in_public
from app.deps.providers import get_llm_service
from app.routers import agent, chat, documents, health
from app.schemas.chat import ChatDelta, ChatResponse
from app.security.rate_limit import InMemoryRequestLimiter, PublicRateLimitMiddleware
from bot.services.quota import BotQuotaExceeded, BotUserQuota


class FakeLLMService:
    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ChatResponse(content="ok", model=request.model or "missing")

    async def stream(self, request):
        self.requests.append(request)
        yield ChatDelta(content="ok")

    async def batch(self, requests):
        self.requests.extend(requests)
        return [ChatResponse(content="ok", model=item.model or "missing") for item in requests]


def _chat_client(admin_token: str = "test-admin-token") -> tuple[TestClient, FakeLLMService]:
    api = FastAPI()
    api.include_router(chat.router)
    service = FakeLLMService()
    settings = SimpleNamespace(
        llm=SimpleNamespace(default_model="server-model"),
        chat_max_tokens=256,
        admin_token=SecretStr(admin_token),
    )
    api.dependency_overrides[get_llm_service] = lambda: service
    api.dependency_overrides[get_settings] = lambda: settings
    return TestClient(api), service


def test_public_chat_ignores_client_model_and_token_budget() -> None:
    client, service = _chat_client()

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "model": "attacker/expensive-model",
            "max_tokens": 16000,
        },
    )

    assert response.status_code == 200
    assert service.requests[0].model == "server-model"
    assert service.requests[0].max_tokens == 256


def test_batch_requires_real_admin_token_and_rejects_placeholder() -> None:
    client, _ = _chat_client()
    payload = {"requests": [{"messages": [{"role": "user", "content": "hello"}]}]}

    assert client.post("/chat/batch", json=payload).status_code == 403
    assert (
        client.post(
            "/chat/batch",
            json=payload,
            headers={"X-Admin-Token": "test-admin-token"},
        ).status_code
        == 200
    )

    placeholder_client, _ = _chat_client("change-me-admin-token")
    assert (
        placeholder_client.post(
            "/chat/batch",
            json=payload,
            headers={"X-Admin-Token": "change-me-admin-token"},
        ).status_code
        == 403
    )


def test_document_upload_requires_admin_token() -> None:
    api = FastAPI()
    api.include_router(documents.router)

    @api.exception_handler(SafeInputError)
    async def safe_input_handler(_, exc: SafeInputError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    api.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        admin_token=SecretStr("test-admin-token")
    )
    client = TestClient(api)
    files = {"file": ("unsupported.json", b"{}", "application/json")}

    assert client.post("/documents/upload", files=files).status_code == 403
    assert (
        client.post(
            "/documents/upload",
            files=files,
            headers={"X-Admin-Token": "test-admin-token"},
        ).status_code
        == 415
    )


def test_public_settings_fail_fast_on_placeholder_tokens() -> None:
    with pytest.raises(ValueError, match="non-placeholder secrets"):
        Settings(
            _env_file=None,
            APP_ENV="public",
            PUBLIC_DOMAIN="demo.example.com",
            PUBLIC_PROXY_IP="172.31.250.10",
            TRUSTED_PROXY_CIDRS="172.31.250.10/32",
            PUBLIC_GENERATION_BUDGET_REQUESTS=100,
            PUBLIC_DATA_RETENTION_DAYS=30,
            TRACING_CAPTURE_CONTENT=False,
            LOG_PROMPT_PREVIEW_ENABLED=False,
            CORS_ORIGINS=["https://demo.example.com"],
            ADMIN_TOKEN="change-me-admin-token",
            INTERNAL_TOKEN="internal-secret",
        )


def test_public_agent_cannot_escalate_role_from_request_body() -> None:
    captured = {}

    class FakeGraph:
        async def astream(self, graph_input, *, config, stream_mode):
            captured["input"] = graph_input
            captured["config"] = config
            yield "updates", {"safe": True}

    api = FastAPI()
    api.state.persistent_agent = FakeGraph()
    api.include_router(agent.router)
    api.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        admin_token=SecretStr("test-admin-token")
    )

    response = TestClient(api).post(
        "/agent/stream",
        json={
            "thread_id": "shared-name",
            "user_role": "full",
            "input": {"messages": [{"role": "user", "content": "hello"}]},
        },
    )

    assert response.status_code == 200
    assert captured["input"]["user_role"] == "read-only"
    assert captured["config"]["configurable"]["user_role"] == "read-only"
    effective_thread = captured["config"]["configurable"]["thread_id"]
    assert effective_thread.startswith("public:")
    assert effective_thread.endswith(":shared-name")


def test_internal_state_endpoint_requires_token_in_public_mode() -> None:
    api = FastAPI()

    @api.post("/state", dependencies=[Depends(require_internal_in_public)])
    async def mutate_state():
        return {"id": str(uuid4())}

    api.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        environment="public",
        internal_token=SecretStr("internal-secret"),
    )
    client = TestClient(api)

    assert client.post("/state").status_code == 403
    assert (
        client.post(
            "/state",
            headers={"X-Internal-Token": "internal-secret"},
        ).status_code
        == 200
    )


def test_rate_limit_applies_only_to_expensive_paths() -> None:
    api = FastAPI()

    @api.post("/chat")
    async def expensive():
        return {"ok": True}

    api.include_router(health.router)
    limited = PublicRateLimitMiddleware(
        api,
        requests=1,
        window_seconds=60,
        max_concurrent=2,
        paths={("POST", "/chat")},
    )
    client = TestClient(limited)

    assert client.post("/chat").status_code == 200
    assert client.post("/chat").status_code == 429
    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200


@pytest.mark.asyncio
async def test_concurrency_slot_is_released_after_request() -> None:
    limiter = InMemoryRequestLimiter(
        requests=10,
        window_seconds=60,
        max_concurrent=1,
    )

    first = await limiter.acquire("one")
    blocked = await limiter.acquire("two")
    assert first.allowed is True
    assert blocked.reason == "concurrency_limit_exceeded"

    await limiter.release()
    after_release = await limiter.acquire("two")
    assert after_release.allowed is True
    await limiter.release()
    assert limiter.active_requests == 0


def test_middleware_releases_streaming_slot_after_completion_and_error() -> None:
    api = FastAPI()

    @api.post("/chat/stream")
    async def stream():
        async def body():
            yield "data: ok\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    @api.post("/chat")
    async def fail():
        raise RuntimeError("provider failed")

    limited = PublicRateLimitMiddleware(
        api,
        requests=10,
        window_seconds=60,
        max_concurrent=1,
        paths={("POST", "/chat"), ("POST", "/chat/stream")},
    )
    client = TestClient(limited, raise_server_exceptions=False)

    assert client.post("/chat/stream").status_code == 200
    assert limited.limiter.active_requests == 0
    assert client.post("/chat").status_code == 500
    assert limited.limiter.active_requests == 0


@pytest.mark.asyncio
async def test_bot_user_rate_and_daily_quotas() -> None:
    rate_quota = BotUserQuota(requests=1, window_seconds=60, daily_quota=10)
    await rate_quota.check("user-1")
    with pytest.raises(BotQuotaExceeded, match="bot_rate_limit"):
        await rate_quota.check("user-1")

    daily_quota = BotUserQuota(requests=10, window_seconds=60, daily_quota=1)
    await daily_quota.check("user-2")
    with pytest.raises(BotQuotaExceeded, match="bot_daily_quota"):
        await daily_quota.check("user-2")


@pytest.mark.asyncio
async def test_bot_quota_state_is_bounded() -> None:
    quota = BotUserQuota(
        requests=10,
        window_seconds=60,
        daily_quota=10,
        max_users=2,
        state_ttl_seconds=3600,
    )

    for user_id in ("user-1", "user-2", "user-3"):
        await quota.check(user_id)

    assert len(quota._recent) == 2
    assert len(quota._daily) == 2
