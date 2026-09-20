from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.chat.deps import get_chat_service
from app.chat.routes import router as chat_history_router
from app.core.config import get_settings
from app.deps.providers import get_llm_service
from app.routers import agent, chat
from app.security.rate_limit import PublicRateLimitMiddleware
from app.services.llm import LLMService


class FailingProviderStream:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.sent_chunk = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.sent_chunk:
            self.sent_chunk = True
            return SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))],
            )
        raise self.error


def _status_error(status: int) -> openai.OpenAIError:
    request = httpx.Request("POST", "https://provider.test/chat/completions")
    response = httpx.Response(status, request=request)
    kwargs = {
        "message": "raw provider body: key=super-secret",
        "response": response,
        "body": {"error": {"message": "super-secret internal response"}},
    }
    if status == 401:
        return openai.AuthenticationError(**kwargs)
    if status == 429:
        return openai.RateLimitError(**kwargs)
    return openai.APIStatusError(**kwargs)


def _chat_stream_client(error: Exception) -> tuple[TestClient, PublicRateLimitMiddleware]:
    openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=FailingProviderStream(error))
            )
        )
    )
    settings = SimpleNamespace(
        llm=SimpleNamespace(default_model="server-model"),
        chat_max_tokens=256,
    )
    service = LLMService(openai_client, {}, settings)
    api = FastAPI()
    api.include_router(chat.router)
    api.dependency_overrides[get_settings] = lambda: settings
    api.dependency_overrides[get_llm_service] = lambda: service
    limited = PublicRateLimitMiddleware(
        api,
        requests=10,
        window_seconds=60,
        max_concurrent=1,
        paths={("POST", "/chat/stream")},
    )
    return TestClient(limited), limited


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (_status_error(401), "llm_unavailable"),
        (_status_error(402), "llm_unavailable"),
        (_status_error(429), "llm_rate_limit"),
        (_status_error(500), "llm_unavailable"),
        (
            openai.APITimeoutError(
                request=httpx.Request(
                    "POST",
                    "https://provider.test/chat/completions",
                )
            ),
            "llm_timeout",
        ),
    ],
    ids=["401", "402", "429", "500", "timeout"],
)
def test_chat_stream_provider_errors_are_safe_and_release_slot(
    error: Exception,
    expected_code: str,
) -> None:
    client, limited = _chat_stream_client(error)

    response = client.post(
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert "data: partial" in response.text
    assert "event: error" in response.text
    assert f'"code":"{expected_code}"' in response.text
    assert "event: done" in response.text
    assert "super-secret" not in response.text
    assert "provider.test" not in response.text
    assert limited.limiter.active_requests == 0


def test_agent_stream_hides_raw_graph_exception_and_finishes() -> None:
    class FailingGraph:
        async def astream(self, *_args, **_kwargs):
            yield "updates", {"started": True}
            raise RuntimeError(
                "postgresql://user:secret@postgres/db http://internal.service"
            )

    api = FastAPI()
    api.state.persistent_agent = FailingGraph()
    api.include_router(agent.router)
    api.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        admin_token=SecretStr("test-admin-token")
    )
    limited = PublicRateLimitMiddleware(
        api,
        requests=10,
        window_seconds=60,
        max_concurrent=1,
        paths={("POST", "/agent/stream")},
    )

    response = TestClient(limited).post(
        "/agent/stream",
        json={
            "thread_id": "safe-error",
            "input": {"messages": [{"role": "user", "content": "hello"}]},
        },
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert '"code":"stream_error"' in response.text
    assert "event: done" in response.text
    assert "postgresql://" not in response.text
    assert "internal.service" not in response.text
    assert limited.limiter.active_requests == 0


def test_chat_history_stream_uses_same_safe_error_contract() -> None:
    class FailingChatService:
        last_sources = []
        last_rag_meta = {}
        last_assistant_message_id = None

        async def check_user_content(self, _content):
            return None

        async def send_message(self, **_kwargs):
            if False:
                yield ""
            raise RuntimeError("http://qdrant:6333 api-key=super-secret")

    api = FastAPI()
    api.include_router(chat_history_router)
    api.dependency_overrides[get_chat_service] = lambda: FailingChatService()
    api.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        environment="dev"
    )

    response = TestClient(api).post(
        f"/chats/{uuid4()}/messages",
        data={"content": "hello"},
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert '"code":"stream_error"' in response.text
    assert "event: done" in response.text
    assert "qdrant:6333" not in response.text
    assert "super-secret" not in response.text
