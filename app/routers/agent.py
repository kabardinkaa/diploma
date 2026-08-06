from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from app.schemas.agent import AgentMessage, AgentStreamRequest


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


async def _events(request: Request, payload: AgentStreamRequest) -> AsyncIterator[str]:
    graph = request.app.state.persistent_agent
    config = {
        "configurable": {
            "thread_id": payload.thread_id,
            "user_role": payload.user_role,
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
            "user_role": payload.user_role,
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
        yield _sse({"type": "error", "message": str(exc)})


@router.post("/stream")
async def agent_stream(
    payload: AgentStreamRequest,
    request: Request,
) -> StreamingResponse:
    return StreamingResponse(
        _events(request, payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
