from __future__ import annotations

import argparse
import asyncio
import json
import operator
import sys
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.core.config import get_settings
from app.tools.graph_agent_tools import TOOLS, TOOLS_BY_NAME


MAX_ITERATIONS = 6
DEFAULT_MODEL = "gpt-5.4-mini"
SYSTEM_PROMPT = """Ты агент внутренней техподдержки. Используй доступные tools
только при необходимости и никогда не вызывай неизвестные инструменты. Не
выдумывай данные: если поиск ничего не нашёл или tools недостаточно, честно
сообщи об этом. Пишущие действия выполняй только после явного подтверждения
пользователя. Когда данных достаточно, верни финальный ответ без tool call. Не
раскрывай скрытые внутренние рассуждения или chain of thought."""


class AgentState(TypedDict):
    """Serializable graph state.

    messages is reducer-managed dialogue history, iteration_count is the
    replaced model-call counter, and tool_results is the accumulated tool trace.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict[str, Any]], operator.add]


def _build_model() -> ChatOpenAI:
    settings = get_settings().llm
    configured = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=0,
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )
    # Some built-in model profiles discard temperature even for compatible APIs.
    if configured.temperature != 0:
        configured = configured.model_copy(update={"temperature": 0.0})
    return configured


model = _build_model()
model_with_tools = model.bind_tools(TOOLS)


async def call_model(state: AgentState) -> dict[str, Any]:
    response = await model_with_tools.ainvoke(state["messages"])
    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1,
    }


def _tool_error(name: str, message: str) -> str:
    return f"Ошибка инструмента '{name}': {message}"


async def execute_tool(state: AgentState) -> dict[str, Any]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return {}

    new_messages: list[ToolMessage] = []
    new_results: list[dict[str, Any]] = []
    for tool_call in last_message.tool_calls:
        name = str(tool_call.get("name") or "")
        call_id = str(tool_call.get("id") or "missing-tool-call-id")
        raw_args = tool_call.get("args", {})
        args = raw_args if isinstance(raw_args, dict) else {}
        error: str | None = None
        selected_tool = TOOLS_BY_NAME.get(name)
        if selected_tool is None:
            available = ", ".join(sorted(TOOLS_BY_NAME))
            error = _tool_error(name, f"неизвестное имя. Доступны: {available}")
            result = error
        elif not isinstance(raw_args, dict):
            error = _tool_error(name, "аргументы должны быть JSON-объектом")
            result = error
        else:
            try:
                result = str(await selected_tool.ainvoke(args))
            except Exception as exc:
                error = _tool_error(name, str(exc))
                result = error

        new_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=call_id,
                name=name or None,
                status="error" if error else "success",
            )
        )
        new_results.append(
            {
                "name": name,
                "args": args,
                "result": result,
                "tool_call_id": call_id,
                "error": error,
            }
        )
    return {"messages": new_messages, "tool_results": new_results}


async def force_finish(state: AgentState) -> dict[str, Any]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return {"messages": [AIMessage(content="Агент не вернул финальный ответ")]}

    final_text = _message_text(last_message)
    if last_message.tool_calls and state["iteration_count"] >= MAX_ITERATIONS:
        messages: list[AnyMessage] = []
        results: list[dict[str, Any]] = []
        for tool_call in last_message.tool_calls:
            name = str(tool_call.get("name") or "")
            call_id = str(tool_call.get("id") or "missing-tool-call-id")
            raw_args = tool_call.get("args", {})
            args = raw_args if isinstance(raw_args, dict) else {}
            error = _tool_error(name, "не выполнен: превышен лимит итераций")
            messages.append(
                ToolMessage(
                    content=error,
                    tool_call_id=call_id,
                    name=name or None,
                    status="error",
                )
            )
            results.append(
                {
                    "name": name,
                    "args": args,
                    "result": error,
                    "tool_call_id": call_id,
                    "error": error,
                }
            )
        if not final_text:
            messages.append(AIMessage(content="Превышен лимит итераций"))
        return {"messages": messages, "tool_results": results}

    if final_text:
        return {}
    return {"messages": [AIMessage(content="Агент не вернул финальный ответ")]}


def route_after_model(
    state: AgentState,
) -> Literal["execute_tool", "force_finish"]:
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "force_finish"
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "execute_tool"
    return "force_finish"


builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("execute_tool", execute_tool)
builder.add_node("force_finish", force_finish)
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    route_after_model,
    {"execute_tool": "execute_tool", "force_finish": "force_finish"},
)
builder.add_edge("execute_tool", "call_model")
builder.add_edge("force_finish", END)
custom_graph = builder.compile()

def _build_prebuilt_graph(configured_model: ChatOpenAI) -> Any:
    # LangChain 1.x create_agent keeps the supplied model configuration intact.
    return create_agent(
        model=configured_model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


prebuilt_graph = _build_prebuilt_graph(model)


def _message_text(message: AnyMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _normalize_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    data = raw or {}
    prompt = int(data.get("input_tokens") or data.get("prompt_tokens") or 0)
    completion = int(
        data.get("output_tokens") or data.get("completion_tokens") or 0
    )
    total = int(data.get("total_tokens") or prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _message_usage(message: AIMessage) -> dict[str, int]:
    if message.usage_metadata:
        return _normalize_usage(dict(message.usage_metadata))
    metadata = message.response_metadata or {}
    raw = metadata.get("token_usage") or metadata.get("usage")
    return _normalize_usage(raw if isinstance(raw, dict) else None)


def _collect_usage(messages: list[AnyMessage]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen: set[str | int] = set()
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        marker: str | int = message.id or id(message)
        if marker in seen:
            continue
        seen.add(marker)
        current = _message_usage(message)
        for key in total:
            total[key] += current[key]
    return total


def _final_answer(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _message_text(message)
            if text:
                return text
    return "Агент не вернул финальный ответ"


def _count_model_steps(messages: list[AnyMessage]) -> int:
    return sum(isinstance(message, AIMessage) for message in messages)


def _tool_results_from_messages(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    results: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                call_id = str(call.get("id") or "")
                raw_args = call.get("args", {})
                calls[call_id] = (
                    str(call.get("name") or ""),
                    raw_args if isinstance(raw_args, dict) else {},
                )
        elif isinstance(message, ToolMessage):
            name, args = calls.get(message.tool_call_id, (message.name or "", {}))
            result = _message_text(message)
            error = result if message.status == "error" else None
            results.append(
                {
                    "name": name,
                    "args": args,
                    "result": result,
                    "tool_call_id": message.tool_call_id,
                    "error": error,
                }
            )
    return results


def _run_config(thread_id: str | None) -> dict[str, Any] | None:
    if thread_id is None:
        return None
    return {"configurable": {"thread_id": thread_id}}


async def run_custom_graph(
    task: str,
    *,
    thread_id: str | None = None,
) -> dict:
    initial: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=task),
        ],
        "iteration_count": 0,
        "tool_results": [],
    }
    state = await custom_graph.ainvoke(initial, config=_run_config(thread_id))
    messages = list(state["messages"])
    return {
        "answer": _final_answer(messages),
        "steps": _count_model_steps(messages),
        "messages": messages,
        "tool_results": list(state["tool_results"]),
        "usage": _collect_usage(messages),
    }


async def run_prebuilt_graph(
    task: str,
    *,
    thread_id: str | None = None,
) -> dict:
    state = await prebuilt_graph.ainvoke(
        {"messages": [HumanMessage(content=task)]},
        config=_run_config(thread_id),
    )
    messages = list(state["messages"])
    return {
        "answer": _final_answer(messages),
        "steps": _count_model_steps(messages),
        "messages": messages,
        "tool_results": _tool_results_from_messages(messages),
        "usage": _collect_usage(messages),
    }


def _trace_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": result["steps"],
        "usage": result["usage"],
        "tool_results": result["tool_results"],
        "message_types": [type(message).__name__ for message in result["messages"]],
    }


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


async def _async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LangGraph-агент блока 6.3")
    parser.add_argument("variant", choices=("custom", "prebuilt"))
    parser.add_argument("task", nargs="?", help="Задача для агента")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--thread-id")
    parser.add_argument("--mermaid", action="store_true")
    args = parser.parse_args(argv)

    graph = custom_graph if args.variant == "custom" else prebuilt_graph
    if args.mermaid:
        print(graph.get_graph().draw_mermaid())
        return 0
    if not args.task:
        parser.error("task обязателен без --mermaid")

    runner = run_custom_graph if args.variant == "custom" else run_prebuilt_graph
    result = await runner(args.task, thread_id=args.thread_id)
    print(result["answer"])
    if args.trace:
        print(json.dumps(_trace_payload(result), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
