from __future__ import annotations

import inspect
import os
import operator
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict
from urllib.parse import quote

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from app.core.config import Settings, get_settings
from app.tools.graph_agent_tools import TOOLS, TOOLS_BY_NAME, send_telegram_message


WRITE_TOOL = "send_telegram_message"
SYSTEM_PROMPT = (
    "Ты агент внутренней техподдержки. Используй только доступные инструменты. "
    "Не утверждай, что пишущее действие выполнено, пока инструмент не вернул результат."
)
UserRole = Literal["read-only", "write-with-approve", "full"]
SendHandler = Callable[[dict[str, str]], Awaitable[Any] | Any]


class PersistentAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_role: UserRole
    pending_send: NotRequired[dict[str, Any] | None]
    approval_decision: NotRequired[bool | None]
    sent: NotRequired[bool | None]
    send_result: NotRequired[str | None]
    tool_results: Annotated[list[dict[str, Any]], operator.add]


def _build_model() -> ChatOpenAI:
    settings = get_settings().llm
    return ChatOpenAI(
        model="gpt-5.4-mini",
        temperature=0,
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )


async def _default_send(payload: dict[str, str]) -> Any:
    return await send_telegram_message.ainvoke(payload)


async def _call_handler(handler: SendHandler, payload: dict[str, str]) -> str:
    result = handler(payload)
    if inspect.isawaitable(result):
        result = await result
    return str(result)


def _last_ai_message(state: PersistentAgentState) -> AIMessage | None:
    message = state["messages"][-1] if state.get("messages") else None
    return message if isinstance(message, AIMessage) else None


def _send_call(state: PersistentAgentState) -> dict[str, Any] | None:
    message = _last_ai_message(state)
    if message is None:
        return None
    return next(
        (call for call in message.tool_calls if call.get("name") == WRITE_TOOL),
        None,
    )


def build_agent(
    checkpointer: Any,
    *,
    model: Any | None = None,
    send_handler: SendHandler | None = None,
) -> Any:
    """Compile the persistent graph without owning the checkpointer lifecycle."""

    configured_model = model or _build_model()
    model_with_tools = configured_model.bind_tools(TOOLS)
    execute_send = send_handler or _default_send

    async def call_model(
        state: PersistentAgentState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        if not any(isinstance(item, SystemMessage) for item in messages):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
        response = await model_with_tools.ainvoke(messages, config=config)
        configured_role = config.get("configurable", {}).get("user_role")
        role = configured_role or state.get("user_role", "write-with-approve")
        return {"messages": [response], "user_role": role}

    async def execute_safe_tools(
        state: PersistentAgentState,
    ) -> dict[str, Any]:
        message = _last_ai_message(state)
        if message is None:
            return {}
        output_messages: list[ToolMessage] = []
        output_results: list[dict[str, Any]] = []
        for call in message.tool_calls:
            name = str(call.get("name") or "")
            call_id = str(call.get("id") or "missing-tool-call-id")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            selected = TOOLS_BY_NAME.get(name)
            if selected is None or name == WRITE_TOOL:
                result = f"Инструмент '{name}' недоступен в этом узле"
                status = "error"
            else:
                try:
                    result = str(await selected.ainvoke(args))
                    status = "success"
                except Exception as exc:
                    result = f"Ошибка инструмента '{name}': {exc}"
                    status = "error"
            output_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=call_id,
                    name=name or None,
                    status=status,
                )
            )
            output_results.append(
                {"name": name, "args": args, "result": result, "status": status}
            )
        return {"messages": output_messages, "tool_results": output_results}

    async def prepare_send_telegram_message(
        state: PersistentAgentState,
    ) -> dict[str, Any]:
        call = _send_call(state)
        if call is None:
            return {"pending_send": None}
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        chat_id = args.get("chat_id")
        text = args.get("text")
        error = None
        if not isinstance(chat_id, str) or not chat_id.strip():
            error = "chat_id должен быть непустой строкой"
        elif not isinstance(text, str) or not text.strip():
            error = "text должен быть непустой строкой"
        return {
            "pending_send": {
                "chat_id": chat_id.strip() if isinstance(chat_id, str) else "",
                "text": text.strip() if isinstance(text, str) else "",
                "tool_call_id": str(call.get("id") or "missing-tool-call-id"),
                "error": error,
            },
            "sent": False,
            "approval_decision": None,
            "send_result": None,
        }

    async def reject_send_telegram_message(
        state: PersistentAgentState,
    ) -> dict[str, Any]:
        pending = state.get("pending_send") or {}
        reason = pending.get("error") or "роль read-only запрещает пишущие действия"
        return {
            "messages": [
                ToolMessage(
                    content=reason,
                    tool_call_id=str(pending.get("tool_call_id", "missing-tool-call-id")),
                    name=WRITE_TOOL,
                    status="error",
                )
            ],
            "approval_decision": False,
            "sent": False,
            "send_result": reason,
            "tool_results": [
                {"name": WRITE_TOOL, "result": reason, "status": "rejected"}
            ],
        }

    async def confirm_and_execute_send_telegram_message(
        state: PersistentAgentState,
    ) -> dict[str, Any]:
        pending = state.get("pending_send") or {}
        preview = {"chat_id": pending["chat_id"], "text": pending["text"]}
        role = state.get("user_role", "write-with-approve")
        approved = True
        if role == "write-with-approve":
            approved = bool(
                interrupt(
                    {"type": "approve_send_telegram_message", "preview": preview}
                )
            )

        if approved:
            result = await _call_handler(execute_send, preview)
            status = "success"
        else:
            result = "Отправка отклонена пользователем"
            status = "rejected"
        return {
            "messages": [
                ToolMessage(
                    content=result,
                    tool_call_id=str(pending["tool_call_id"]),
                    name=WRITE_TOOL,
                    status="success" if approved else "error",
                )
            ],
            "approval_decision": approved,
            "sent": approved,
            "send_result": result,
            "tool_results": [
                {"name": WRITE_TOOL, "result": result, "status": status}
            ],
        }

    def route_after_model(
        state: PersistentAgentState,
    ) -> Literal["prepare_send_telegram_message", "execute_safe_tools", "__end__"]:
        message = _last_ai_message(state)
        if message is None or not message.tool_calls:
            return "__end__"
        if _send_call(state) is not None:
            return "prepare_send_telegram_message"
        return "execute_safe_tools"

    def route_after_prepare(
        state: PersistentAgentState,
    ) -> Literal[
        "confirm_and_execute_send_telegram_message",
        "reject_send_telegram_message",
    ]:
        pending = state.get("pending_send") or {}
        if pending.get("error") or state.get("user_role") == "read-only":
            return "reject_send_telegram_message"
        return "confirm_and_execute_send_telegram_message"

    builder = StateGraph(PersistentAgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_safe_tools", execute_safe_tools)
    builder.add_node("prepare_send_telegram_message", prepare_send_telegram_message)
    builder.add_node(
        "confirm_and_execute_send_telegram_message",
        confirm_and_execute_send_telegram_message,
    )
    builder.add_node("reject_send_telegram_message", reject_send_telegram_message)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model)
    builder.add_edge("execute_safe_tools", "call_model")
    builder.add_conditional_edges("prepare_send_telegram_message", route_after_prepare)
    builder.add_edge("confirm_and_execute_send_telegram_message", "call_model")
    builder.add_edge("reject_send_telegram_message", "call_model")
    return builder.compile(checkpointer=checkpointer)


def _postgres_uri(settings: Settings) -> str:
    if settings.database_url:
        return settings.database_url
    user = quote(os.getenv("POSTGRES_USER", "postgres"), safe="")
    password = quote(os.getenv("POSTGRES_PASSWORD", "postgres"), safe="")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = quote(os.getenv("POSTGRES_DB", "diploma"), safe="")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@asynccontextmanager
async def agent_lifespan(
    settings: Settings | None = None,
) -> AsyncIterator[Any]:
    """Own one checkpointer and one compiled graph for the application lifespan."""

    configured = settings or get_settings()
    async with AsyncExitStack() as stack:
        if configured.agent_checkpointer == "memory":
            checkpointer: Any = InMemorySaver()
        elif configured.agent_checkpointer == "sqlite":
            sqlite_path = Path(configured.agent_sqlite_path)
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(sqlite_path))
            )
            await checkpointer.setup()
        else:
            checkpointer = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(_postgres_uri(configured))
            )
            await checkpointer.setup()
        yield build_agent(checkpointer)


__all__ = [
    "Command",
    "PersistentAgentState",
    "agent_lifespan",
    "build_agent",
]
