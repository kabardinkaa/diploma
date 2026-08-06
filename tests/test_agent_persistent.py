from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.services import agent_persistent
from app.services.agent_persistent import agent_lifespan, build_agent
from app.routers.agent import router as agent_router


class SendRequestModel:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, config=None):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="Действие обработано.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "send_telegram_message",
                    "args": {"chat_id": "support-1", "text": "Нужна помощь"},
                    "id": "call-send-1",
                    "type": "tool_call",
                }
            ],
        )


def _input(role: str = "write-with-approve") -> dict:
    return {
        "messages": [("user", "Отправь сообщение в поддержку")],
        "user_role": role,
        "tool_results": [],
    }


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.asyncio
async def test_agent_lifespan_calls_setup_once(monkeypatch, tmp_path) -> None:
    saver = SimpleNamespace(setup=AsyncMock())

    @asynccontextmanager
    async def fake_saver_context():
        yield saver

    monkeypatch.setattr(
        agent_persistent.AsyncSqliteSaver,
        "from_conn_string",
        lambda _path: fake_saver_context(),
    )
    monkeypatch.setattr(
        agent_persistent,
        "build_agent",
        lambda checkpointer: ("compiled", checkpointer),
    )
    settings = SimpleNamespace(
        agent_checkpointer="sqlite",
        agent_sqlite_path=tmp_path / "lifespan.db",
    )

    async with agent_lifespan(settings) as graph:
        assert graph == ("compiled", saver)

    saver.setup.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_graph_reaches_interrupt(tmp_path) -> None:
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "interrupt.db")) as saver:
        await saver.setup()
        graph = build_agent(saver, model=SendRequestModel(), send_handler=AsyncMock())
        config = _config("interrupt-thread")

        result = await graph.ainvoke(_input(), config=config)
        snapshot = await graph.aget_state(config)

        assert "__interrupt__" in result
        assert snapshot.next == ("confirm_and_execute_send_telegram_message",)
        assert snapshot.values["pending_send"]["chat_id"] == "support-1"
        assert snapshot.values["sent"] is False


@pytest.mark.asyncio
async def test_resume_true_executes_side_effect_once(tmp_path) -> None:
    send = AsyncMock(return_value="Сообщение отправлено")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "approve.db")) as saver:
        await saver.setup()
        graph = build_agent(saver, model=SendRequestModel(), send_handler=send)
        config = _config("approve-thread")

        await graph.ainvoke(_input(), config=config)
        result = await graph.ainvoke(Command(resume=True), config=config)

        assert result["sent"] is True
        assert result["approval_decision"] is True
        send.assert_awaited_once_with(
            {"chat_id": "support-1", "text": "Нужна помощь"}
        )


@pytest.mark.asyncio
async def test_resume_false_does_not_execute_side_effect(tmp_path) -> None:
    send = AsyncMock(return_value="must not happen")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "reject.db")) as saver:
        await saver.setup()
        graph = build_agent(saver, model=SendRequestModel(), send_handler=send)
        config = _config("reject-thread")

        await graph.ainvoke(_input(), config=config)
        result = await graph.ainvoke(Command(resume=False), config=config)

        assert result["sent"] is False
        assert result["approval_decision"] is False
        send.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_only_role_rejects_without_interrupt(tmp_path) -> None:
    send = AsyncMock()
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "readonly.db")) as saver:
        await saver.setup()
        graph = build_agent(saver, model=SendRequestModel(), send_handler=send)
        config = _config("readonly-thread")

        result = await graph.ainvoke(_input("read-only"), config=config)

        assert "__interrupt__" not in result
        assert result["sent"] is False
        send.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_role_executes_without_interrupt(tmp_path) -> None:
    send = AsyncMock(return_value="Сообщение отправлено")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "full.db")) as saver:
        await saver.setup()
        graph = build_agent(saver, model=SendRequestModel(), send_handler=send)
        config = _config("full-thread")

        result = await graph.ainvoke(_input("full"), config=config)

        assert "__interrupt__" not in result
        assert result["sent"] is True
        send.assert_awaited_once()


def test_stream_request_requires_exactly_one_input_or_resume() -> None:
    api = FastAPI()
    api.include_router(agent_router)

    with TestClient(api) as client:
        neither = client.post(
            "/agent/stream",
            json={"thread_id": "thread-1", "user_role": "read-only"},
        )
        both = client.post(
            "/agent/stream",
            json={
                "thread_id": "thread-1",
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "resume": True,
            },
        )

    assert neither.status_code == 422
    assert both.status_code == 422


def test_sse_uses_updates_and_messages_and_serializes_interrupt() -> None:
    class FakeGraph:
        async def astream(self, graph_input, *, config, stream_mode):
            assert stream_mode == ["updates", "messages"]
            assert config["configurable"]["thread_id"] == "stream-thread"
            yield "updates", {
                "__interrupt__": [
                    {
                        "type": "approve_send_telegram_message",
                        "preview": {"chat_id": "1", "text": "hello"},
                    }
                ]
            }
            yield "messages", (AIMessage(content="готово"), {"node": "call_model"})

    api = FastAPI()
    api.state.persistent_agent = FakeGraph()
    api.include_router(agent_router)

    with TestClient(api) as client:
        response = client.post(
            "/agent/stream",
            json={
                "thread_id": "stream-thread",
                "input": {
                    "messages": [{"role": "user", "content": "hello"}]
                },
                "user_role": "write-with-approve",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "interrupt"' in response.text
    assert '"type": "message"' in response.text
    assert '"type": "done"' in response.text


def _sse_payloads(response: httpx.Response) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_real_sqlite_sse_interrupt_and_two_resume_branches(tmp_path) -> None:
    send = AsyncMock(return_value="Сообщение отправлено")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "sse.db")) as saver:
        await saver.setup()
        graph = build_agent(saver, model=SendRequestModel(), send_handler=send)
        api = FastAPI()
        api.state.persistent_agent = graph
        api.include_router(agent_router)
        transport = httpx.ASGITransport(app=api)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            initial_payload = {
                "input": {
                    "messages": [
                        {"role": "user", "content": "Отправь сообщение в поддержку"}
                    ]
                },
                "user_role": "write-with-approve",
            }
            approve_start = await client.post(
                "/agent/stream",
                json={"thread_id": "sse-approve", **initial_payload},
            )
            reject_start = await client.post(
                "/agent/stream",
                json={"thread_id": "sse-reject", **initial_payload},
            )
            approve_resume = await client.post(
                "/agent/stream",
                json={
                    "thread_id": "sse-approve",
                    "resume": True,
                    "user_role": "write-with-approve",
                },
            )
            reject_resume = await client.post(
                "/agent/stream",
                json={
                    "thread_id": "sse-reject",
                    "resume": False,
                    "user_role": "write-with-approve",
                },
            )

        for response in (approve_start, reject_start, approve_resume, reject_resume):
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert all(isinstance(item, dict) for item in _sse_payloads(response))

        approve_events = _sse_payloads(approve_start)
        reject_events = _sse_payloads(reject_start)
        for events in (approve_events, reject_events):
            assert "update" in {item["type"] for item in events}
            assert "message" in {item["type"] for item in events}
            interrupt_event = next(item for item in events if item["type"] == "interrupt")
            assert interrupt_event["data"]["__interrupt__"][0]["type"] == (
                "approve_send_telegram_message"
            )
            assert events[-1]["type"] == "done"

        assert _sse_payloads(approve_resume)[-1]["type"] == "done"
        assert _sse_payloads(reject_resume)[-1]["type"] == "done"
        assert (await graph.aget_state(_config("sse-approve"))).values["sent"] is True
        assert (await graph.aget_state(_config("sse-reject"))).values["sent"] is False
        send.assert_awaited_once()
