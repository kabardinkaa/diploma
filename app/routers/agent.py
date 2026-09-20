from __future__ import annotations

import json
import hashlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from app.schemas.agent import AgentMessage, AgentStreamRequest
from app.core.sse import sse_error_events
from app.deps.providers import SettingsDep
from app.security.tokens import secret_matches


router = APIRouter(prefix="/agent", tags=["agent"])


def _message_from_request(message: AgentMessage) -> Any:
    message_type = {
        "user": HumanMessage,
        "assistant": AIMessage,
        "system": SystemMessage,
    }[message.role]
    return message_type(content=message.content)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseException):
        return "internal_error"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return str(value)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _agent_context(
    request: Request,
    payload: AgentStreamRequest,
    settings: SettingsDep,
    admin_token: str | None,
) -> tuple[str, str]:
    if admin_token is not None:
        if not secret_matches(admin_token, settings.admin_token):
            raise HTTPException(status_code=403, detail="Invalid admin token")
        return f"admin:{payload.thread_id}", "write-with-approve"

    client_host = request.client.host if request.client else "unknown"
    client_scope = hashlib.sha256(client_host.encode("utf-8")).hexdigest()[:16]
    return f"public:{client_scope}:{payload.thread_id}", "read-only"


async def _events(
    request: Request,
    payload: AgentStreamRequest,
    *,
    effective_thread_id: str,
    user_role: str,
) -> AsyncIterator[str]:
    graph = request.app.state.persistent_agent
    config = {
        "configurable": {
            "thread_id": effective_thread_id,
            "user_role": user_role,
        }
    }
    if payload.resume is not None:
        graph_input: Any = Command(resume=payload.resume)
    else:
        assert payload.input is not None
        graph_input = {
            "messages": [
                _message_from_request(message) for message in payload.input.messages
            ],
            "user_role": user_role,
            "tool_results": [],
        }

    try:
        async for mode, event in graph.astream(
            graph_input,
            config=config,
            stream_mode=["updates", "messages"],
        ):
            event_type = "message" if mode == "messages" else "update"
            if mode == "updates" and isinstance(event, dict) and "__interrupt__" in event:
                event_type = "interrupt"
            yield _sse({"type": event_type, "mode": mode, "data": _jsonable(event)})
        yield _sse({"type": "done", "thread_id": payload.thread_id})
    except Exception as exc:
        for event in sse_error_events(exc):
            yield event


@router.post("/stream")
async def agent_stream(
    payload: AgentStreamRequest,
    request: Request,
    settings: SettingsDep,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> StreamingResponse:
    effective_thread_id, user_role = _agent_context(
        request,
        payload,
        settings,
        x_admin_token,
    )
    return StreamingResponse(
        _events(
            request,
            payload,
            effective_thread_id=effective_thread_id,
            user_role=user_role,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
