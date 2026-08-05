from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages

from app.services import agent_graph
from app.eval.agent_scenarios import AGENT_SCENARIOS
from app.tools import graph_agent_tools
from scripts import bench_agents


class FakeTool:
    def __init__(self, result: str = "ok", error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(args)
        if self.error:
            raise self.error
        return self.result


class FakeGraph:
    def __init__(self, output: dict) -> None:
        self.output = output
        self.input = None
        self.config = None

    async def ainvoke(self, graph_input, config=None):
        self.input = graph_input
        self.config = config
        return self.output


class SequenceModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.histories: list[list] = []

    async def ainvoke(self, messages):
        self.histories.append(list(messages))
        return self.responses.pop(0)


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_agent_state_declares_required_reducers() -> None:
    hints = get_type_hints(agent_graph.AgentState, include_extras=True)

    assert set(hints) == {"messages", "iteration_count", "tool_results"}
    assert hints["messages"].__metadata__ == (add_messages,)
    assert hints["tool_results"].__metadata__ == (agent_graph.operator.add,)
    assert hints["iteration_count"] is int
    assert not ({"client", "model", "api_key", "base_url", "session", "logger"} & set(hints))
    serializable = {
        "messages": [HumanMessage(content="hello").model_dump(mode="json")],
        "iteration_count": 0,
        "tool_results": [],
    }
    json.dumps(serializable)


def test_build_model_uses_existing_endpoint_and_key_source(monkeypatch) -> None:
    api_key = object()
    llm = SimpleNamespace(
        api_key=api_key,
        base_url="http://local.test/v1",
        request_timeout=17,
        max_retries=2,
    )
    captured = {}

    class FakeChatOpenAI:
        temperature = 0.0

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(agent_graph, "get_settings", lambda: SimpleNamespace(llm=llm))
    monkeypatch.setattr(agent_graph, "ChatOpenAI", FakeChatOpenAI)

    configured = agent_graph._build_model()

    assert isinstance(configured, FakeChatOpenAI)
    assert captured == {
        "model": "gpt-5.4-mini",
        "temperature": 0,
        "api_key": api_key,
        "base_url": "http://local.test/v1",
        "timeout": 17,
        "max_retries": 2,
    }


def test_prebuilt_receives_the_same_configured_model(monkeypatch) -> None:
    configured_model = object()
    compiled = object()
    factory = Mock(return_value=compiled)
    monkeypatch.setattr(agent_graph, "create_agent", factory)

    assert agent_graph._build_prebuilt_graph(configured_model) is compiled
    assert factory.call_args.kwargs["model"] is configured_model
    assert factory.call_args.kwargs["tools"] is graph_agent_tools.TOOLS
    assert factory.call_args.kwargs["system_prompt"] == agent_graph.SYSTEM_PROMPT
    assert agent_graph.model_with_tools.bound is agent_graph.model


def test_graph_tools_are_shared_and_descriptive() -> None:
    assert set(graph_agent_tools.TOOLS_BY_NAME) == {
        "search_knowledge_base",
        "get_current_time",
        "send_telegram_message",
    }
    assert [item.name for item in graph_agent_tools.TOOLS] == list(
        graph_agent_tools.TOOLS_BY_NAME
    )
    for graph_tool in graph_agent_tools.TOOLS:
        assert graph_tool.description.count(".") >= 4


def test_graph_tool_reuses_existing_implementation(monkeypatch) -> None:
    implementation = Mock(return_value="fragment")
    monkeypatch.setattr(
        graph_agent_tools,
        "search_knowledge_base_impl",
        implementation,
    )

    result = graph_agent_tools.search_knowledge_base.invoke({"query": "VPN"})

    assert result == "fragment"
    implementation.assert_called_once_with("VPN")


@pytest.mark.asyncio
async def test_call_model_returns_update_without_mutating_state(monkeypatch) -> None:
    response = AIMessage(content="Финал")

    class FakeModel:
        async def ainvoke(self, messages):
            assert len(messages) == 1
            return response

    monkeypatch.setattr(agent_graph, "model_with_tools", FakeModel())
    state = {"messages": [HumanMessage(content="Задача")], "iteration_count": 2, "tool_results": []}

    update = await agent_graph.call_model(state)

    assert update == {"messages": [response], "iteration_count": 3}
    assert state["iteration_count"] == 2
    assert len(state["messages"]) == 1


@pytest.mark.asyncio
async def test_execute_tool_runs_every_call_with_matching_ids(monkeypatch) -> None:
    first = FakeTool("one")
    second = FakeTool("two")
    monkeypatch.setitem(agent_graph.TOOLS_BY_NAME, "first", first)
    monkeypatch.setitem(agent_graph.TOOLS_BY_NAME, "second", second)
    message = AIMessage(
        content="",
        tool_calls=[_call("first", {"x": 1}, "call-1"), _call("second", {"y": 2}, "call-2")],
    )

    update = await agent_graph.execute_tool(
        {"messages": [message], "iteration_count": 1, "tool_results": []}
    )

    assert [item.tool_call_id for item in update["messages"]] == ["call-1", "call-2"]
    assert [item.content for item in update["messages"]] == ["one", "two"]
    assert [item["name"] for item in update["tool_results"]] == ["first", "second"]
    assert first.calls == [{"x": 1}]
    assert second.calls == [{"y": 2}]


@pytest.mark.asyncio
async def test_unknown_tool_becomes_error_tool_message() -> None:
    message = AIMessage(content="", tool_calls=[_call("missing", {}, "call-x")])

    update = await agent_graph.execute_tool(
        {"messages": [message], "iteration_count": 1, "tool_results": []}
    )

    tool_message = update["messages"][0]
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "call-x"
    assert "неизвестное имя" in tool_message.content
    assert update["tool_results"][0]["error"] == tool_message.content


@pytest.mark.asyncio
async def test_invalid_args_become_error_tool_message(monkeypatch) -> None:
    selected = FakeTool()
    monkeypatch.setitem(agent_graph.TOOLS_BY_NAME, "known", selected)
    message = AIMessage(content="", tool_calls=[_call("known", {}, "call-x")])
    message.tool_calls[0]["args"] = "not-an-object"

    update = await agent_graph.execute_tool(
        {"messages": [message], "iteration_count": 1, "tool_results": []}
    )

    assert update["messages"][0].status == "error"
    assert "JSON-объектом" in update["messages"][0].content
    assert selected.calls == []


@pytest.mark.asyncio
async def test_tool_exception_becomes_observation(monkeypatch) -> None:
    monkeypatch.setitem(
        agent_graph.TOOLS_BY_NAME,
        "broken",
        FakeTool(error=RuntimeError("unavailable")),
    )
    message = AIMessage(content="", tool_calls=[_call("broken", {}, "call-x")])

    update = await agent_graph.execute_tool(
        {"messages": [message], "iteration_count": 1, "tool_results": []}
    )

    assert update["messages"][0].status == "error"
    assert "unavailable" in update["messages"][0].content


@pytest.mark.asyncio
async def test_force_finish_keeps_existing_final_answer() -> None:
    message = AIMessage(content="Готовый ответ")

    assert await agent_graph.force_finish(
        {"messages": [message], "iteration_count": 1, "tool_results": []}
    ) == {}


@pytest.mark.asyncio
async def test_force_finish_closes_tool_protocol_at_limit() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            _call("first", {"x": 1}, "call-1"),
            _call("second", {"y": 2}, "call-2"),
        ],
    )

    update = await agent_graph.force_finish(
        {"messages": [message], "iteration_count": 6, "tool_results": []}
    )

    assert [type(item) for item in update["messages"]] == [
        ToolMessage,
        ToolMessage,
        AIMessage,
    ]
    assert [item.tool_call_id for item in update["messages"][:2]] == ["call-1", "call-2"]
    assert update["messages"][-1].content == "Превышен лимит итераций"
    assert len(update["tool_results"]) == 2


@pytest.mark.asyncio
async def test_custom_graph_finishes_without_tool(monkeypatch) -> None:
    fake_model = SequenceModel([AIMessage(content="Прямой ответ")])
    monkeypatch.setattr(agent_graph, "model_with_tools", fake_model)

    result = await agent_graph.run_custom_graph("Ответь")

    assert result["answer"] == "Прямой ответ"
    assert result["steps"] == 1
    assert result["tool_results"] == []


@pytest.mark.asyncio
async def test_custom_graph_full_single_tool_path(monkeypatch) -> None:
    fake_model = SequenceModel(
        [
            AIMessage(
                content="",
                tool_calls=[_call("clock", {"timezone": "UTC"}, "call-1")],
            ),
            AIMessage(content="Сейчас 12:00"),
        ]
    )
    selected = FakeTool("12:00")
    monkeypatch.setattr(agent_graph, "model_with_tools", fake_model)
    monkeypatch.setitem(agent_graph.TOOLS_BY_NAME, "clock", selected)

    result = await agent_graph.run_custom_graph("Время")

    assert result["answer"] == "Сейчас 12:00"
    assert result["steps"] == 2
    assert result["tool_results"][0]["tool_call_id"] == "call-1"
    assert [type(item) for item in fake_model.histories[1][-2:]] == [
        AIMessage,
        ToolMessage,
    ]


@pytest.mark.asyncio
async def test_custom_graph_full_two_tool_path_is_protocol_valid(monkeypatch) -> None:
    calls = [
        _call("first", {"x": 1}, "call-1"),
        _call("second", {"y": 2}, "call-2"),
    ]
    fake_model = SequenceModel(
        [AIMessage(content="", tool_calls=calls), AIMessage(content="Финал")]
    )
    monkeypatch.setattr(agent_graph, "model_with_tools", fake_model)
    monkeypatch.setitem(agent_graph.TOOLS_BY_NAME, "first", FakeTool("one"))
    monkeypatch.setitem(agent_graph.TOOLS_BY_NAME, "second", FakeTool("two"))

    result = await agent_graph.run_custom_graph("Два действия")

    assert result["answer"] == "Финал"
    assert result["steps"] == 2
    assert [item["tool_call_id"] for item in result["tool_results"]] == [
        "call-1",
        "call-2",
    ]
    second_history = fake_model.histories[1]
    tool_messages = [item for item in second_history if isinstance(item, ToolMessage)]
    assert [item.tool_call_id for item in tool_messages] == ["call-1", "call-2"]


@pytest.mark.asyncio
async def test_custom_graph_full_unknown_tool_path(monkeypatch) -> None:
    fake_model = SequenceModel(
        [
            AIMessage(
                content="",
                tool_calls=[_call("unknown", {}, "call-x")],
            ),
            AIMessage(content="Инструмент недоступен"),
        ]
    )
    monkeypatch.setattr(agent_graph, "model_with_tools", fake_model)

    result = await agent_graph.run_custom_graph("Неизвестный tool")

    assert result["answer"] == "Инструмент недоступен"
    assert result["tool_results"][0]["error"]
    assert isinstance(fake_model.histories[1][-1], ToolMessage)


@pytest.mark.asyncio
async def test_custom_graph_force_finish_returns_explicit_answer(monkeypatch) -> None:
    responses = [
        AIMessage(
            content="",
            tool_calls=[_call("loop", {}, f"call-{number}")],
        )
        for number in range(1, agent_graph.MAX_ITERATIONS + 1)
    ]
    fake_model = SequenceModel(responses)
    monkeypatch.setattr(agent_graph, "model_with_tools", fake_model)
    monkeypatch.setitem(agent_graph.TOOLS_BY_NAME, "loop", FakeTool("again"))

    result = await agent_graph.run_custom_graph("Зациклись")

    assert result["answer"] == "Превышен лимит итераций"
    messages = result["messages"]
    call_ids = {
        call["id"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    result_ids = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    assert call_ids == result_ids
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "Превышен лимит итераций"


@pytest.mark.parametrize(
    ("iteration", "message", "route"),
    [
        (6, AIMessage(content="", tool_calls=[_call("x", {}, "1")]), "force_finish"),
        (1, AIMessage(content="", tool_calls=[_call("x", {}, "1")]), "execute_tool"),
        (1, AIMessage(content="Финал"), "force_finish"),
    ],
)
def test_router_is_deterministic(iteration, message, route) -> None:
    state = {"messages": [message], "iteration_count": iteration, "tool_results": []}

    assert agent_graph.route_after_model(state) == route
    assert state["iteration_count"] == iteration


def test_custom_and_prebuilt_graphs_are_compiled_once() -> None:
    custom = agent_graph.custom_graph.get_graph().draw_mermaid()
    prebuilt = agent_graph.prebuilt_graph.get_graph().draw_mermaid()

    assert all(name in custom for name in ("call_model", "execute_tool", "force_finish"))
    assert "model" in prebuilt
    assert "tools" in prebuilt


def test_usage_normalizes_metadata_and_deduplicates_messages() -> None:
    first = AIMessage(
        id="one",
        content="a",
        usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        response_metadata={
            "token_usage": {
                "prompt_tokens": 500,
                "completion_tokens": 200,
                "total_tokens": 700,
            }
        },
    )
    second = AIMessage(
        id="two",
        content="b",
        response_metadata={
            "token_usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        },
    )

    tool_message = ToolMessage(content="free", tool_call_id="call-x")

    assert agent_graph._collect_usage([first, first, tool_message, second]) == {
        "prompt_tokens": 8,
        "completion_tokens": 6,
        "total_tokens": 14,
    }


@pytest.mark.asyncio
async def test_run_custom_graph_builds_initial_state_and_config(monkeypatch) -> None:
    final = AIMessage(
        content="Готово",
        usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
    )
    fake = FakeGraph(
        {"messages": [SystemMessage(content="s"), HumanMessage(content="q"), final], "iteration_count": 1, "tool_results": []}
    )
    monkeypatch.setattr(agent_graph, "custom_graph", fake)

    result = await agent_graph.run_custom_graph("Задача", thread_id="thread-1")

    assert isinstance(fake.input["messages"][0], SystemMessage)
    assert isinstance(fake.input["messages"][1], HumanMessage)
    assert fake.input["iteration_count"] == 0
    assert fake.input["tool_results"] == []
    assert fake.config == {"configurable": {"thread_id": "thread-1"}}
    assert result["answer"] == "Готово"
    assert result["steps"] == 1
    assert result["usage"]["total_tokens"] == 6


@pytest.mark.asyncio
async def test_run_prebuilt_graph_normalizes_history(monkeypatch) -> None:
    call = _call("get_current_time", {"timezone": "UTC"}, "call-1")
    messages = [
        HumanMessage(content="Время"),
        AIMessage(
            content="",
            tool_calls=[call],
            usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        ),
        ToolMessage(content="12:00", tool_call_id="call-1", name="get_current_time"),
        AIMessage(
            content="Сейчас 12:00",
            usage_metadata={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
        ),
    ]
    fake = FakeGraph({"messages": messages})
    monkeypatch.setattr(agent_graph, "prebuilt_graph", fake)

    result = await agent_graph.run_prebuilt_graph("Время", thread_id="thread-2")

    assert result["answer"] == "Сейчас 12:00"
    assert result["steps"] == 2
    assert result["usage"]["total_tokens"] == 15
    assert result["tool_results"] == [
        {
            "name": "get_current_time",
            "args": {"timezone": "UTC"},
            "result": "12:00",
            "tool_call_id": "call-1",
            "error": None,
        }
    ]
    assert fake.config == {"configurable": {"thread_id": "thread-2"}}


@pytest.mark.asyncio
async def test_prebuilt_final_without_tool_and_without_usage(monkeypatch) -> None:
    messages = [HumanMessage(content="Вопрос"), AIMessage(content="Ответ")]
    monkeypatch.setattr(agent_graph, "prebuilt_graph", FakeGraph({"messages": messages}))

    result = await agent_graph.run_prebuilt_graph("Вопрос")

    assert result["answer"] == "Ответ"
    assert result["steps"] == 1
    assert result["tool_results"] == []
    assert result["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


@pytest.mark.asyncio
async def test_prebuilt_multiple_tools_and_response_metadata(monkeypatch) -> None:
    calls = [
        _call("first", {"x": 1}, "call-1"),
        _call("second", {"y": 2}, "call-2"),
    ]
    messages = [
        HumanMessage(content="Два действия"),
        AIMessage(
            content="",
            tool_calls=calls,
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                }
            },
        ),
        ToolMessage(content="one", tool_call_id="call-1", name="first"),
        ToolMessage(content="two", tool_call_id="call-2", name="second"),
        AIMessage(
            content="Финал",
            response_metadata={
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 3,
                    "total_tokens": 11,
                }
            },
        ),
    ]
    monkeypatch.setattr(agent_graph, "prebuilt_graph", FakeGraph({"messages": messages}))

    result = await agent_graph.run_prebuilt_graph("Два действия")

    assert result["answer"] == "Финал"
    assert result["steps"] == 2
    assert result["usage"] == {
        "prompt_tokens": 13,
        "completion_tokens": 5,
        "total_tokens": 18,
    }
    assert [item["tool_call_id"] for item in result["tool_results"]] == [
        "call-1",
        "call-2",
    ]


def test_total_steps_is_ai_message_count_for_shared_history() -> None:
    messages = [
        HumanMessage(content="Задача"),
        AIMessage(content="", tool_calls=[_call("tool", {}, "call-1")]),
        ToolMessage(content="result", tool_call_id="call-1"),
        AIMessage(content="Финал"),
    ]

    assert agent_graph._count_model_steps(messages) == 2


def test_system_prompt_contains_safety_rules() -> None:
    prompt = " ".join(agent_graph.SYSTEM_PROMPT.lower().split())

    assert "только при необходимости" in prompt
    assert "не выдумывай" in prompt
    assert "неизвестные инструменты" in prompt
    assert "явного подтверждения" in prompt
    assert "финальный ответ без tool call" in prompt
    assert "chain of thought" in prompt


def test_model_configuration_is_deterministic_and_uses_shared_tools() -> None:
    assert agent_graph.model.model_name == "gpt-5.4-mini"
    assert agent_graph.model.temperature == 0
    assert [item["function"]["name"] for item in agent_graph.model_with_tools.kwargs["tools"]] == [
        item.name for item in graph_agent_tools.TOOLS
    ]


@pytest.mark.asyncio
async def test_benchmark_contract_is_prepared_but_not_executed() -> None:
    assert len(bench_agents.TASKS) == 5
    assert set(bench_agents.RUNNERS) == {"react_6_2", "custom", "prebuilt"}
    with pytest.raises(ValueError, match="ровно 3"):
        await bench_agents.run_benchmark(2)


def test_benchmark_dry_run_has_45_unique_run_ids() -> None:
    matrix = bench_agents.dry_run_matrix()

    assert len(matrix) == 45
    assert len({item["run_id"] for item in matrix}) == 45
    assert matrix[0]["run_id"] == "bench-1-react_6_2-1"
    assert matrix[-1]["run_id"] == "bench-5-prebuilt-3"
    assert [scenario.task for scenario in AGENT_SCENARIOS] == bench_agents.TASKS


def test_full_benchmark_requires_loopback_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        bench_agents,
        "get_settings",
        lambda: SimpleNamespace(llm=SimpleNamespace(base_url="http://127.0.0.1:1235/v1")),
    )
    bench_agents._require_local_endpoint()

    monkeypatch.setattr(
        bench_agents,
        "get_settings",
        lambda: SimpleNamespace(llm=SimpleNamespace(base_url="https://example.com/v1")),
    )
    with pytest.raises(RuntimeError, match="только через локальный"):
        bench_agents._require_local_endpoint()


@pytest.mark.asyncio
async def test_react_benchmark_uses_15_second_iteration_timeout(monkeypatch) -> None:
    captured = {}

    def fake_run(task, **kwargs):
        captured.update(kwargs)
        return {"answer": "ok", "steps": 1, "trace": [], "usage": {}}

    monkeypatch.setattr(bench_agents, "run_react_with_reflection", fake_run)

    await bench_agents._run_react("Задача", "ignored")

    assert captured == {"timeout_per_iteration_sec": 15}


def test_benchmark_cli_has_separate_safe_timeout_defaults() -> None:
    args = bench_agents._build_parser().parse_args([])

    assert 5 <= args.react_timeout_per_iteration <= 15
    assert args.react_timeout_per_iteration == 15
    assert args.run_timeout == 60


def test_benchmark_cli_rejects_unsafe_react_timeout() -> None:
    parser = bench_agents._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--react-timeout-per-iteration", "16"])


@pytest.mark.asyncio
async def test_run_timeout_is_not_forwarded_to_react(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(task, **kwargs):
        captured.update(kwargs)
        return {"answer": "ok", "steps": 1, "trace": [], "usage": {}}

    monkeypatch.setattr(bench_agents, "run_react_with_reflection", fake_run)
    await bench_agents._run_one(
        AGENT_SCENARIOS[1],
        "react_6_2",
        1,
        60,
        raw_path=tmp_path / "react.json",
    )

    assert captured == {"timeout_per_iteration_sec": 15}


@pytest.mark.asyncio
@pytest.mark.parametrize("implementation", ["custom", "prebuilt"])
async def test_graph_runners_use_only_outer_run_timeout(
    monkeypatch,
    tmp_path,
    implementation,
) -> None:
    calls = []

    async def fake_runner(task, thread_id):
        calls.append((task, thread_id))
        await bench_agents.asyncio.sleep(0.02)
        return {"answer": "late"}

    monkeypatch.setitem(bench_agents.RUNNERS, implementation, fake_runner)
    row = await bench_agents._run_one(
        AGENT_SCENARIOS[1],
        implementation,
        1,
        0.001,
        raw_path=tmp_path / f"{implementation}.json",
    )

    assert len(calls) == 1
    assert row["timed_out"] is True


@pytest.mark.asyncio
async def test_benchmark_timeout_is_saved_without_retry(monkeypatch, tmp_path) -> None:
    calls = 0

    async def slow_runner(task, thread_id):
        nonlocal calls
        calls += 1
        await bench_agents.asyncio.sleep(0.05)
        return {"answer": "late"}

    monkeypatch.setitem(bench_agents.RUNNERS, "custom", slow_runner)
    monkeypatch.setattr(bench_agents, "RAW_DIR", tmp_path)

    row = await bench_agents._run_one(AGENT_SCENARIOS[1], "custom", 1, 0.001)

    assert calls == 1
    assert row["timed_out"] is True
    assert row["error"].startswith("Per-run timeout")
    assert Path(row["raw_file"]).is_file()
    raw = json.loads(Path(row["raw_file"]).read_text(encoding="utf-8"))
    assert raw["timed_out"] is True
    assert raw["run_id"] == "bench-2-custom-1"


@pytest.mark.asyncio
async def test_benchmark_continues_after_one_error(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    async def fake_runner(task, thread_id, **kwargs):
        calls.append(thread_id)
        if thread_id == "bench-1-react_6_2-1":
            raise RuntimeError("first failed")
        return {
            "answer": "HTTP и HTTPS",
            "steps": 2,
            "tool_results": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    for implementation in bench_agents.IMPLEMENTATIONS:
        monkeypatch.setitem(bench_agents.RUNNERS, implementation, fake_runner)
    monkeypatch.setattr(bench_agents, "RAW_DIR", tmp_path)
    report = tmp_path / "report.md"
    report.write_text(
        f"{bench_agents.START_MARKER}\n{bench_agents.END_MARKER}",
        encoding="utf-8",
    )
    monkeypatch.setattr(bench_agents, "REPORT_PATH", report)

    rows = await bench_agents.run_benchmark(run_timeout_sec=1)

    assert len(rows) == 45
    assert len(calls) == 45
    assert len(set(calls)) == 45
    assert sum(bool(row["error"]) for row in rows) == 1
    assert {row["total_steps"] for row in rows if not row["error"]} == {2}
    assert len(list(tmp_path.glob("bench-*-*-*.json"))) == 45
    assert (tmp_path / "benchmark-results.json").is_file()
    required = {
        "task_id",
        "implementation",
        "repeat",
        "answer",
        "correct",
        "correctness_reason",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "total_steps",
        "tool_calls_count",
        "tools_used",
        "error",
        "timed_out",
        "raw_file",
    }
    assert required <= set(rows[0])


def test_deterministic_correctness_rules() -> None:
    checks = [
        bench_agents._correctness(
            AGENT_SCENARIOS[0],
            "Инструкция",
            {"tool_results": [{"name": "search_knowledge_base"}]},
            None,
            False,
        )[0],
        bench_agents._correctness(
            AGENT_SCENARIOS[1],
            "12:00",
            {
                "tool_results": [
                    {
                        "name": "get_current_time",
                        "args": {"timezone": "Europe/Moscow"},
                    }
                ]
            },
            None,
            False,
        )[0],
        bench_agents._correctness(
            AGENT_SCENARIOS[2],
            "Черновик",
            {"tool_results": [{"name": "search_knowledge_base"}]},
            None,
            False,
        )[0],
        bench_agents._correctness(
            AGENT_SCENARIOS[3],
            "Отправлено",
            {
                "tool_results": [
                    {"name": "search_knowledge_base"},
                    {"name": "send_telegram_message"},
                ]
            },
            None,
            False,
        )[0],
        bench_agents._correctness(
            AGENT_SCENARIOS[4], "HTTP и HTTPS", {}, None, False
        )[0],
    ]

    assert checks == [None, True, None, None, True]
    assert bench_agents._correctness(
        AGENT_SCENARIOS[2],
        "Черновик",
        {
            "tool_results": [
                {"name": "search_knowledge_base"},
                {"name": "send_telegram_message"},
            ]
        },
        None,
        False,
    )[0] is False


def test_benchmark_table_averages_repeats() -> None:
    rows = [
        {
            "task_id": 1,
            "implementation": "custom",
            "answer": "ok",
            "latency_ms": 10.0,
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
            "total_steps": 1,
            "tool_calls_count": 1,
            "correct": True,
            "error": None,
            "timed_out": False,
        },
        {
            "task_id": 1,
            "implementation": "custom",
            "answer": "ok",
            "latency_ms": 20.0,
            "prompt_tokens": 6,
            "completion_tokens": 4,
            "total_tokens": 10,
            "total_steps": 3,
            "tool_calls_count": 3,
            "correct": False,
            "error": None,
            "timed_out": False,
        },
    ]

    table = bench_agents._benchmark_table(rows)

    assert (
        "| 1 | custom | 15.0 | 10.0 | 20.0 | 5.0 | 3.0 | 8.0 | "
        "2.0 | 2.0 | 2 | 1 | 0 | 0 |"
    ) in table


def _raw_payload(run_id: str) -> dict:
    match = bench_agents.RAW_RUN_PATTERN.fullmatch(f"{run_id}.json")
    assert match is not None
    task_id, implementation, repeat = match.group(2, 3, 4)
    return {
        "run_id": run_id,
        "task_id": int(task_id),
        "task": "task",
        "implementation": implementation,
        "repeat": int(repeat),
        "answer": "ok",
        "correct": True,
        "correctness_reason": "ok",
        "latency_ms": 1.0,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "total_steps": 1,
        "tool_calls_count": 0,
        "tools_used": [],
        "error": None,
        "timed_out": False,
        "raw_file": f"docs/agent-graph-results/{run_id}.json",
    }


def _set_benchmark_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bench_agents, "RAW_DIR", tmp_path)
    report = tmp_path / "report.md"
    report.write_text(
        f"before\n{bench_agents.START_MARKER}\nold\n"
        f"{bench_agents.END_MARKER}\nafter\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bench_agents, "REPORT_PATH", report)


def test_benchmark_filters_select_expected_matrix() -> None:
    assert len(bench_agents.select_matrix(task_id=2)) == 9
    assert len(bench_agents.select_matrix(implementation="custom")) == 15
    assert len(bench_agents.select_matrix(repeat=2)) == 15
    selected = bench_agents.select_matrix(
        task_id=2,
        implementation="custom",
        repeat=2,
    )
    assert [item["run_id"] for item in selected] == ["bench-2-custom-2"]


@pytest.mark.asyncio
async def test_resume_skips_existing_valid_run(monkeypatch, tmp_path) -> None:
    _set_benchmark_paths(monkeypatch, tmp_path)
    run_id = "bench-2-custom-1"
    (tmp_path / f"{run_id}.json").write_text(
        json.dumps(_raw_payload(run_id)),
        encoding="utf-8",
    )

    async def must_not_run(task, thread_id):
        raise AssertionError("runner must be skipped")

    monkeypatch.setitem(bench_agents.RUNNERS, "custom", must_not_run)
    rows = await bench_agents.run_benchmark(
        task_id=2,
        implementation="custom",
        repeat=1,
        resume=True,
    )
    assert rows == []


@pytest.mark.asyncio
async def test_resume_preserves_corrupt_raw_and_runs_missing(monkeypatch, tmp_path) -> None:
    _set_benchmark_paths(monkeypatch, tmp_path)
    raw = tmp_path / "bench-2-custom-1.json"
    raw.write_text("{broken", encoding="utf-8")
    calls = []

    async def fake_runner(task, thread_id):
        calls.append(thread_id)
        return {"answer": "ok", "steps": 1, "tool_results": [], "usage": {}}

    monkeypatch.setitem(bench_agents.RUNNERS, "custom", fake_runner)
    await bench_agents.run_benchmark(
        task_id=2,
        implementation="custom",
        repeat=1,
        resume=True,
    )
    assert calls == ["bench-2-custom-1"]
    assert raw.is_file()
    assert (tmp_path / "bench-2-custom-1.json.corrupt").read_text() == "{broken"


@pytest.mark.asyncio
async def test_runs_are_sequential_and_saved_incrementally(monkeypatch, tmp_path) -> None:
    _set_benchmark_paths(monkeypatch, tmp_path)
    active = 0
    max_active = 0
    calls = 0

    async def fake_runner(task, thread_id):
        nonlocal active, max_active, calls
        if calls:
            summary = json.loads(
                (tmp_path / "benchmark-results.json").read_text(encoding="utf-8")
            )
            assert len(summary) == calls
        calls += 1
        active += 1
        max_active = max(max_active, active)
        await bench_agents.asyncio.sleep(0)
        active -= 1
        return {"answer": "ok", "steps": 1, "tool_results": [], "usage": {}}

    monkeypatch.setitem(bench_agents.RUNNERS, "custom", fake_runner)
    rows = await bench_agents.run_benchmark(
        task_id=2,
        implementation="custom",
    )
    assert len(rows) == 3
    assert max_active == 1
    assert len(json.loads((tmp_path / "benchmark-results.json").read_text())) == 3


def test_aggregate_only_does_not_require_endpoint_or_runner(
    monkeypatch,
    tmp_path,
) -> None:
    _set_benchmark_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        bench_agents,
        "_require_local_endpoint",
        lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    monkeypatch.setattr(
        bench_agents.sys,
        "argv",
        ["bench_agents.py", "--aggregate-only"],
    )
    bench_agents.main()
    assert (tmp_path / "benchmark-results.json").is_file()


def _write_raw(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _failed_payload(run_id: str, error: str = "Timeout") -> dict:
    payload = _raw_payload(run_id)
    payload.update(answer="", correct=False, error=error, timed_out=error == "Timeout")
    return payload


def test_technical_failure_classification_ignores_semantic_correctness() -> None:
    timeout = _failed_payload("bench-1-custom-1")
    endpoint = _failed_payload("bench-1-custom-1", "APIConnectionError: Connection error")
    semantic = _raw_payload("bench-1-custom-1")
    semantic["correct"] = False

    assert bench_agents._technical_error_type(timeout) == "timeout"
    assert bench_agents._technical_error_type(endpoint) == "endpoint_connection"
    assert bench_agents._is_technical_success(semantic) is True


@pytest.mark.asyncio
async def test_rerun_failed_skips_technical_success(monkeypatch, tmp_path) -> None:
    _set_benchmark_paths(monkeypatch, tmp_path)
    run_id = "bench-1-custom-1"
    payload = _raw_payload(run_id)
    payload["correct"] = False
    _write_raw(tmp_path / f"{run_id}.json", payload)

    async def must_not_run(task, thread_id):
        raise AssertionError("semantic incorrect must not be retried")

    monkeypatch.setitem(bench_agents.RUNNERS, "custom", must_not_run)
    rows = await bench_agents.run_benchmark(
        task_id=1,
        implementation="custom",
        repeat=1,
        resume=True,
        rerun_failed=True,
    )
    assert rows == []


@pytest.mark.asyncio
async def test_rerun_archives_attempt_one_and_writes_attempt_two(
    monkeypatch,
    tmp_path,
) -> None:
    _set_benchmark_paths(monkeypatch, tmp_path)
    run_id = "bench-5-custom-1"
    source = tmp_path / f"{run_id}.json"
    _write_raw(source, _failed_payload(run_id))
    original = source.read_bytes()

    async def fake_runner(task, thread_id):
        return {
            "answer": "HTTP и HTTPS",
            "steps": 1,
            "tool_results": [],
            "usage": {},
        }

    monkeypatch.setitem(bench_agents.RUNNERS, "custom", fake_runner)
    await bench_agents.run_benchmark(
        task_id=5,
        implementation="custom",
        repeat=1,
        resume=True,
        rerun_failed=True,
    )

    attempt_one = tmp_path / f"{run_id}.attempt-1.json"
    attempt_two = tmp_path / f"{run_id}.attempt-2.json"
    assert not source.exists()
    assert attempt_one.read_bytes() == original
    assert attempt_two.is_file()
    summary = json.loads((tmp_path / "benchmark-results.json").read_text())
    assert summary[0]["selected_attempt"] == 2
    assert summary[0]["attempts_count"] == 2
    assert summary[0]["previous_errors"][0]["type"] == "timeout"


def test_canonical_attempt_rules(monkeypatch, tmp_path) -> None:
    _set_benchmark_paths(monkeypatch, tmp_path)
    success_id = "bench-1-custom-1"
    failed_id = "bench-1-custom-2"
    _write_raw(
        tmp_path / f"{success_id}.attempt-1.json",
        _failed_payload(success_id),
    )
    _write_raw(
        tmp_path / f"{success_id}.attempt-2.json",
        _raw_payload(success_id),
    )
    _write_raw(
        tmp_path / f"{failed_id}.attempt-1.json",
        _failed_payload(failed_id, "first error"),
    )
    _write_raw(
        tmp_path / f"{failed_id}.attempt-2.json",
        _failed_payload(failed_id, "second error"),
    )

    rows, _ = bench_agents.aggregate_results(update_report=False)
    by_id = {row["run_id"]: row for row in rows}
    assert by_id[success_id]["selected_attempt"] == 2
    assert by_id[success_id]["error"] is None
    assert by_id[failed_id]["selected_attempt"] == 2
    assert by_id[failed_id]["error"] == "second error"


@pytest.mark.asyncio
async def test_max_attempts_prevents_third_attempt(monkeypatch, tmp_path) -> None:
    _set_benchmark_paths(monkeypatch, tmp_path)
    run_id = "bench-1-custom-1"
    for attempt in (1, 2):
        _write_raw(
            tmp_path / f"{run_id}.attempt-{attempt}.json",
            _failed_payload(run_id),
        )

    async def must_not_run(task, thread_id):
        raise AssertionError("third attempt must not run")

    monkeypatch.setitem(bench_agents.RUNNERS, "custom", must_not_run)
    rows = await bench_agents.run_benchmark(
        task_id=1,
        implementation="custom",
        repeat=1,
        resume=True,
        rerun_failed=True,
        max_attempts=2,
    )
    assert rows == []
    assert len(list(tmp_path.glob(f"{run_id}.attempt-*.json"))) == 2


def test_plan_rerun_failed_does_not_call_endpoint(monkeypatch, tmp_path) -> None:
    _set_benchmark_paths(monkeypatch, tmp_path)
    run_id = "bench-1-custom-1"
    _write_raw(tmp_path / f"{run_id}.json", _failed_payload(run_id))
    monkeypatch.setattr(
        bench_agents,
        "_require_local_endpoint",
        lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    monkeypatch.setattr(
        bench_agents.sys,
        "argv",
        ["bench_agents.py", "--plan-rerun-failed"],
    )
    bench_agents.main()


def test_aggregate_returns_45_canonical_results(monkeypatch, tmp_path) -> None:
    _set_benchmark_paths(monkeypatch, tmp_path)
    for item in bench_agents.dry_run_matrix():
        _write_raw(
            tmp_path / f"{item['run_id']}.json",
            _raw_payload(item["run_id"]),
        )
    rows, problems = bench_agents.aggregate_results(update_report=False)
    assert len(rows) == 45
    assert len({row["run_id"] for row in rows}) == 45
    assert problems == []


def test_visualization_files_match_compiled_graphs() -> None:
    root = Path(agent_graph.__file__).resolve().parents[2]

    assert (root / "docs/agent-graph-custom.mmd").read_text(encoding="utf-8") == (
        agent_graph.custom_graph.get_graph().draw_mermaid().strip() + "\n"
    )
    assert (root / "docs/agent-graph-prebuilt.mmd").read_text(encoding="utf-8") == (
        agent_graph.prebuilt_graph.get_graph().draw_mermaid().strip() + "\n"
    )


def test_previous_agent_baselines_are_byte_identical() -> None:
    service_dir = Path(agent_graph.__file__).resolve().parent
    naive = (service_dir / "agent_naive.py").read_bytes()
    react = (service_dir / "agent_react.py").read_bytes()

    assert hashlib.sha256(naive).hexdigest() == (
        "db07cf8de5d2937eb41194fd4d50873c7db2c2e2659dd071d79265ad49910901"
    )
    assert hashlib.sha256(react).hexdigest() == (
        "e1f22e4ae8d134d545fe16d678e8c03ed99252a16e0e56fea9d2be56548ba729"
    )
