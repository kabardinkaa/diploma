from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from experiments import common, multi_agent_langgraph, single_agent_baseline


class FakeRAG:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    async def retrieve_contexts(self, query: str, *, top_k: int):
        self.queries.append((query, top_k))
        return [
            {"text": "Первый факт", "file_name": "vpn.md", "page": 2},
            {"text": "Второй факт", "file_name": "access.md", "page": None},
        ]


class FakeStreamApp:
    def __init__(self, events: list[dict]) -> None:
        self.events = events
        self.configs: list[dict] = []

    async def astream(self, graph_input, *, config, stream_mode):
        assert graph_input["messages"][0]["role"] == "user"
        assert stream_mode == "updates"
        self.configs.append(config)
        for event in self.events:
            yield event


class FakeSupervisorWorkflow:
    def __init__(self) -> None:
        self.edges = {
            ("researcher", "supervisor"),
            ("writer", "supervisor"),
        }
        self.nodes: dict[str, object] = {}
        self.compiled_with = None

    def add_node(self, name: str, node) -> None:
        self.nodes[name] = node

    def add_edge(self, start: str, end: str) -> None:
        self.edges.add((start, end))

    def compile(self, **kwargs):
        self.compiled_with = kwargs
        return "compiled"


@pytest.mark.asyncio
async def test_shared_search_tool_uses_existing_rag_and_numbers_sources() -> None:
    rag = FakeRAG()
    async with common.activate_search(rag) as trace:
        result = await common.SEARCH_KNOWLEDGE_BASE_TOOL.ainvoke({"query": "VPN"})

    assert rag.queries == [("VPN", 3)]
    assert "[1] Источник: vpn.md, страница 2" in result
    assert "[2] Источник: access.md" in result
    assert [item["id"] for item in trace.contexts] == [1, 2]


@pytest.mark.asyncio
async def test_sufficient_first_retrieval_is_reused_without_second_rag_call() -> None:
    rag = FakeRAG()
    async with common.activate_search(rag) as trace:
        first = await common.SEARCH_KNOWLEDGE_BASE_TOOL.ainvoke({"query": "VPN"})
        second = await common.SEARCH_KNOWLEDGE_BASE_TOOL.ainvoke(
            {"query": "корпоративный VPN"}
        )

    assert first == second
    assert rag.queries == [("VPN", 3)]
    assert trace.tool_calls == ["VPN", "корпоративный VPN"]
    assert trace.calls == ["VPN"]
    assert [item["id"] for item in trace.contexts] == [1, 2]
    assert second.count("[1] Источник:") == 1
    assert second.count("[2] Источник:") == 1


@pytest.mark.asyncio
async def test_empty_first_retrieval_allows_a_second_query() -> None:
    class EmptyThenFoundRAG(FakeRAG):
        async def retrieve_contexts(self, query: str, *, top_k: int):
            self.queries.append((query, top_k))
            if len(self.queries) == 1:
                return []
            return [{"text": "Факт", "file_name": "vpn.md", "page": None}]

    rag = EmptyThenFoundRAG()
    async with common.activate_search(rag) as trace:
        first = await common.SEARCH_KNOWLEDGE_BASE_TOOL.ainvoke({"query": "первый"})
        second = await common.SEARCH_KNOWLEDGE_BASE_TOOL.ainvoke({"query": "второй"})

    assert "не найдены" in first
    assert "[1] Источник: vpn.md" in second
    assert trace.calls == ["первый", "второй"]


def test_five_questions_have_required_categories() -> None:
    categories = [item.category for item in common.TEST_QUESTIONS]

    assert len(common.TEST_QUESTIONS) == 5
    assert categories.count("corpus") == 3
    assert categories.count("multi_step") == 1
    assert categories.count("out_of_scope") == 1


def test_both_implementations_receive_the_same_tool(monkeypatch) -> None:
    tool = object()
    single_factory = Mock(return_value="single")
    monkeypatch.setattr(single_agent_baseline, "create_agent", single_factory)

    assert single_agent_baseline.build_single_agent("model", search_tool=tool) == "single"
    assert single_factory.call_args.kwargs["tools"] == [tool]

    created = ["researcher", "writer"]
    multi_factory = Mock(side_effect=created)
    workflow = FakeSupervisorWorkflow()
    supervisor_factory = Mock(return_value=workflow)
    monkeypatch.setattr(multi_agent_langgraph, "create_agent", multi_factory)
    monkeypatch.setattr(multi_agent_langgraph, "create_supervisor", supervisor_factory)

    assert multi_agent_langgraph.build_multi_agent(
        "model", search_tool=tool, checkpointer="memory"
    ) == "compiled"
    assert multi_factory.call_args_list[0].kwargs["tools"] == [tool]
    assert multi_factory.call_args_list[1].kwargs["tools"] == []
    assert supervisor_factory.call_args.kwargs["output_mode"] == "last_message"
    assert supervisor_factory.call_args.kwargs["add_handoff_back_messages"] is False
    assert (
        supervisor_factory.call_args.kwargs["post_model_hook"]
        is multi_agent_langgraph.guard_supervisor_handoff
    )
    assert ("writer", "supervisor") not in workflow.edges
    assert ("writer", "supervisor_final") in workflow.edges
    assert ("supervisor_final", multi_agent_langgraph.END) in workflow.edges
    assert workflow.compiled_with["checkpointer"] == "memory"


def test_worker_prompts_enforce_role_boundaries() -> None:
    assert "Не формируй финальный ответ" in multi_agent_langgraph.RESEARCHER_PROMPT
    assert "нет инструментов" in multi_agent_langgraph.WRITER_PROMPT
    assert "не выполняй поиск" in multi_agent_langgraph.SUPERVISOR_PROMPT
    assert "ровно один раз" in multi_agent_langgraph.RESEARCHER_PROMPT


def test_supervisor_final_preserves_writer_answer_without_new_message() -> None:
    writer = AIMessage(
        id="writer-final",
        name="writer",
        content="Итоговый ответ [1] [2].",
        usage_metadata={"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
    )
    events = [
        {"writer": {"messages": [writer]}},
        {"supervisor_final": {}},
    ]

    assert multi_agent_langgraph.finish_after_writer({"messages": [writer]}) == {}
    assert common.final_answer(events) == writer.content
    assert common.collect_metrics(events)["llm_calls"] == 1


def test_empty_supervisor_stop_after_successful_researcher_falls_back_to_writer() -> None:
    researcher = AIMessage(
        id="researcher-facts",
        name="researcher",
        content="Факт о VPN [1]. Ещё один факт [2].",
    )
    empty_stop = AIMessage(id="supervisor-stop", name="supervisor", content="")

    command = multi_agent_langgraph.guard_supervisor_handoff(
        {"messages": [researcher, empty_stop]}
    )

    assert isinstance(command, Command)
    assert command.goto == "writer"
    assert command.graph == Command.PARENT
    handoff, confirmation = command.update["messages"][-2:]
    assert handoff.tool_calls[0]["name"] == "transfer_to_writer"
    assert handoff.response_metadata["handoff_fallback"] is True
    assert confirmation.tool_call_id == handoff.tool_calls[0]["id"]


def test_valid_writer_handoff_is_not_replaced_by_fallback() -> None:
    researcher = AIMessage(
        name="researcher",
        content="Факты [1].",
    )
    handoff = AIMessage(
        name="supervisor",
        content="",
        tool_calls=[
            {
                "name": "transfer_to_writer",
                "args": {},
                "id": "writer-handoff",
                "type": "tool_call",
            }
        ],
    )

    assert multi_agent_langgraph.guard_supervisor_handoff(
        {"messages": [researcher, handoff]}
    ) == {}


@pytest.mark.parametrize(
    "researcher_content",
    ["", "Контексты не найдены, передавать writer нечего."],
)
def test_writer_fallback_is_not_forced_without_successful_researcher_facts(
    researcher_content: str,
) -> None:
    messages = [
        AIMessage(name="researcher", content=researcher_content),
        AIMessage(name="supervisor", content=""),
    ]

    assert multi_agent_langgraph.guard_supervisor_handoff({"messages": messages}) == {}


def test_usage_handoffs_and_citations_are_collected_once() -> None:
    researcher_handoff = AIMessage(
        id="supervisor-1",
        content="",
        tool_calls=[
            {
                "name": "transfer_to_researcher",
                "args": {},
                "id": "handoff-1",
                "type": "tool_call",
            }
        ],
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    writer_handoff = AIMessage(
        id="supervisor-2",
        content="",
        tool_calls=[
            {
                "name": "transfer_to_writer",
                "args": {},
                "id": "handoff-2",
                "type": "tool_call",
            }
        ],
        usage_metadata={"input_tokens": 12, "output_tokens": 2, "total_tokens": 14},
    )
    final = AIMessage(
        id="writer-1",
        content="Проверьте подключение [1] и права доступа [2].",
        usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
    )
    events = [
        {"supervisor": {"messages": [researcher_handoff]}},
        {"supervisor": {"messages": [researcher_handoff, writer_handoff]}},
        {"writer": {"messages": [writer_handoff, final]}},
        {"supervisor_final": {}},
    ]

    assert common.collect_metrics(events) == {
        "prompt_tokens": 42,
        "completion_tokens": 12,
        "total_tokens": 54,
        "llm_calls": 3,
        "handoff_count": 2,
    }
    assert common.final_answer(events).endswith("[2].")


def test_result_contains_category_and_multi_agent_route() -> None:
    final = AIMessage(name="writer", content="Ответ [1].")
    events = [
        {"supervisor": {}},
        {"researcher": {}},
        {"supervisor": {}},
        {"writer": {"messages": [final]}},
        {"supervisor_final": None},
    ]

    result = common.build_result(
        implementation="multi_agent",
        question=common.TEST_QUESTIONS[0],
        events=events,
        latency_ms=1.0,
        trace=common.SearchTrace(),
        quality=None,
    )

    assert result.category == "corpus"
    assert result.route == [
        "supervisor",
        "researcher",
        "supervisor",
        "writer",
        "supervisor_final",
    ]


def test_results_serialization_merges_by_implementation_and_question(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    base = common.ExperimentResult(
        implementation="single_agent",
        question_id="1",
        question="q",
        total_tokens=1,
        llm_calls=1,
        latency_ms=1.5,
        handoff_count=0,
        quality=None,
        answer="first",
    )
    common.save_results([base], path)
    base.answer = "updated"
    common.save_results([base], path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["answer"] == "updated"


def test_result_aggregation_uses_means_and_latency_median() -> None:
    rows = [
        {
            "implementation": "single_agent",
            "total_tokens": 10,
            "llm_calls": 1,
            "latency_ms": 30,
            "handoff_count": 0,
            "quality": 0.8,
        },
        {
            "implementation": "single_agent",
            "total_tokens": 20,
            "llm_calls": 2,
            "latency_ms": 10,
            "handoff_count": 0,
            "quality": 1.0,
        },
    ]

    summary = common.aggregate_results(rows)["single_agent"]

    assert summary == {
        "runs": 2.0,
        "avg_total_tokens": 15.0,
        "avg_llm_calls": 1.5,
        "latency_p50_ms": 20.0,
        "avg_handoff_count": 0.0,
        "avg_quality": 0.9,
    }


def test_mermaid_is_generated_and_saved(tmp_path: Path) -> None:
    graph = SimpleNamespace(draw_mermaid=Mock(return_value="graph TD\n A --> B"))
    app = SimpleNamespace(get_graph=Mock(return_value=graph))
    path = tmp_path / "architecture.md"

    mermaid = multi_agent_langgraph.save_mermaid(app, path)

    assert mermaid == "graph TD\n A --> B"
    assert "```mermaid" in path.read_text(encoding="utf-8")
    graph.draw_mermaid.assert_called_once_with()


@pytest.mark.asyncio
async def test_question_runs_use_independent_thread_ids() -> None:
    final = AIMessage(id="final", content="Ответ [1].")
    app = FakeStreamApp([{"writer": {"messages": [final]}}])
    rag = FakeRAG()

    await multi_agent_langgraph.run_question(
        app, common.TEST_QUESTIONS[0], rag, print_updates=False
    )
    await multi_agent_langgraph.run_question(
        app, common.TEST_QUESTIONS[1], rag, print_updates=False
    )

    ids = [item["configurable"]["thread_id"] for item in app.configs]
    assert ids == ["exp-langgraph-1", "exp-langgraph-2"]
    assert common._ACTIVE_RAG.get() is None


def test_external_model_endpoint_is_rejected_before_client_creation(monkeypatch) -> None:
    llm = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        api_key="secret",
        request_timeout=60,
        max_retries=0,
    )
    monkeypatch.setattr(common, "ChatOpenAI", Mock())

    with pytest.raises(RuntimeError, match="local OpenAI-compatible endpoint"):
        common.build_experiment_model(SimpleNamespace(llm=llm))

    common.ChatOpenAI.assert_not_called()


def test_experiment_model_disables_thinking_for_every_agent(monkeypatch) -> None:
    llm = SimpleNamespace(
        base_url="http://127.0.0.1:1235/v1",
        api_key="local-only",
        request_timeout=60,
        max_retries=0,
    )
    client = object()
    monkeypatch.setenv("EXPERIMENT_MODEL", "qwen/qwen3.6-35b-a3b")
    monkeypatch.setattr(common, "ChatOpenAI", Mock(return_value=client))

    assert common.build_experiment_model(SimpleNamespace(llm=llm)) is client
    common.ChatOpenAI.assert_called_once_with(
        model="qwen/qwen3.6-35b-a3b",
        temperature=0,
        api_key="local-only",
        base_url="http://127.0.0.1:1235/v1",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=60,
        max_retries=0,
    )
