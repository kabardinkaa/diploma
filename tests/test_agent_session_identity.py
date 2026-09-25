from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import get_settings
from app.routers import agent
from app.security.identity import (
    PUBLIC_AGENT_SESSION_COOKIE,
    issue_public_agent_identity,
    verify_public_agent_cookie,
)


PAYLOAD = {
    "thread_id": "demo",
    "input": {"messages": [{"role": "user", "content": "hello"}]},
    "user_role": "full",
}


class CapturingGraph:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def astream(self, graph_input, *, config, stream_mode):
        self.calls.append(
            {
                "input": graph_input,
                "config": config,
                "stream_mode": stream_mode,
            }
        )
        yield "updates", {"safe": True}


def _settings(*, environment: str = "public") -> SimpleNamespace:
    return SimpleNamespace(
        environment=environment,
        admin_token=SecretStr("test-admin-token"),
        public_session_secret=SecretStr(
            "test-public-session-secret-with-at-least-32-characters"
        ),
        public_session_ttl_seconds=3600,
    )


def _app(settings: SimpleNamespace) -> tuple[FastAPI, CapturingGraph]:
    api = FastAPI()
    graph = CapturingGraph()
    api.state.persistent_agent = graph
    api.include_router(agent.router)
    api.dependency_overrides[get_settings] = lambda: settings
    return api, graph


def _thread_id(call: dict) -> str:
    return call["config"]["configurable"]["thread_id"]


def test_two_public_clients_get_isolated_namespaces_behind_same_proxy_and_nat() -> None:
    api, graph = _app(_settings())
    forwarded_headers = {
        "X-Forwarded-For": "203.0.113.7",
        "X-Forwarded-Proto": "https",
    }

    with TestClient(api, base_url="https://testserver") as first:
        first_response = first.post(
            "/agent/stream",
            json=PAYLOAD,
            headers=forwarded_headers,
        )
    with TestClient(api, base_url="https://testserver") as second:
        second_response = second.post(
            "/agent/stream",
            json=PAYLOAD,
            headers=forwarded_headers,
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_thread, second_thread = (_thread_id(call) for call in graph.calls)
    assert first_thread != second_thread
    assert first_thread.endswith(":demo")
    assert second_thread.endswith(":demo")
    assert graph.calls[0]["input"]["user_role"] == "read-only"
    assert graph.calls[1]["input"]["user_role"] == "read-only"


def test_public_client_keeps_identity_between_requests() -> None:
    api, graph = _app(_settings())

    with TestClient(api, base_url="https://testserver") as client:
        first = client.post("/agent/stream", json=PAYLOAD)
        second = client.post("/agent/stream", json=PAYLOAD)

    assert _thread_id(graph.calls[0]) == _thread_id(graph.calls[1])
    assert PUBLIC_AGENT_SESSION_COOKIE in first.cookies
    assert "set-cookie" not in second.headers


def test_public_cookie_has_safe_attributes_and_is_not_exposed_in_stream() -> None:
    api, _ = _app(_settings())

    with TestClient(api, base_url="https://testserver") as client:
        response = client.post("/agent/stream", json=PAYLOAD)

    cookie = response.cookies[PUBLIC_AGENT_SESSION_COOKIE]
    set_cookie = response.headers["set-cookie"].lower()
    assert "httponly" in set_cookie
    assert "samesite=lax" in set_cookie
    assert "secure" in set_cookie
    assert "path=/agent" in set_cookie
    assert "max-age=3600" in set_cookie
    assert cookie not in response.text


def test_tampered_public_cookie_cannot_select_existing_principal() -> None:
    settings = _settings()
    api, graph = _app(settings)

    with TestClient(api, base_url="https://testserver") as owner:
        owner_response = owner.post("/agent/stream", json=PAYLOAD)
    owner_token = owner_response.cookies[PUBLIC_AGENT_SESSION_COOKIE]
    tampered = owner_token[:-1] + ("A" if owner_token[-1] != "A" else "B")

    with TestClient(api, base_url="https://testserver") as attacker:
        attacker.cookies.set(
            PUBLIC_AGENT_SESSION_COOKIE,
            tampered,
            path="/agent",
        )
        attacker_response = attacker.post("/agent/stream", json=PAYLOAD)

    assert attacker_response.status_code == 200
    assert _thread_id(graph.calls[0]) != _thread_id(graph.calls[1])
    assert attacker_response.cookies[PUBLIC_AGENT_SESSION_COOKIE] != tampered


def test_expired_public_cookie_is_rejected() -> None:
    settings = _settings()
    issued = issue_public_agent_identity(settings, now=1_000)

    assert issued.cookie_value is not None
    assert verify_public_agent_cookie(
        issued.cookie_value,
        settings,
        now=1_000 + settings.public_session_ttl_seconds + 1,
    ) is None


def test_admin_namespace_and_role_are_unchanged_and_no_public_cookie_is_set() -> None:
    api, graph = _app(_settings())

    with TestClient(api, base_url="https://testserver") as client:
        response = client.post(
            "/agent/stream",
            json=PAYLOAD,
            headers={"X-Admin-Token": "test-admin-token"},
        )

    assert response.status_code == 200
    assert _thread_id(graph.calls[0]) == "admin:demo"
    assert graph.calls[0]["input"]["user_role"] == "write-with-approve"
    assert "set-cookie" not in response.headers
