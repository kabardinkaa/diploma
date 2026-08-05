from types import SimpleNamespace
from unittest.mock import Mock

from app.services import agent_naive
from app.tools import naive_agent_tools


def _tool_call(
    name: str,
    arguments: str,
    *,
    call_id: str = "call_1",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        ),
    )


def _mock_client(monkeypatch, *responses: SimpleNamespace) -> Mock:
    client = Mock()
    client.chat.completions.create = Mock(side_effect=responses)
    monkeypatch.setattr(agent_naive, "_build_client", lambda: client)
    return client


def test_run_agent_returns_final_answer_without_tool_call(monkeypatch) -> None:
    client = _mock_client(monkeypatch, _response(content="Готовый ответ"))

    result = agent_naive.run_agent("Ответь кратко")

    assert result == {
        "answer": "Готовый ответ",
        "steps": 1,
        "trace": [
            {
                "step": 1,
                "tool_name": None,
                "tool_args": {},
                "tool_result": None,
                "llm_input_tokens": 10,
                "llm_output_tokens": 5,
                "duration_ms": result["trace"][0]["duration_ms"],
            }
        ],
    }
    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-5.4-mini"


def test_run_agent_executes_tool_and_keeps_full_history(monkeypatch) -> None:
    tool_response = _response(
        tool_calls=[_tool_call("get_current_time", '{"timezone":"Europe/Moscow"}')]
    )
    client = _mock_client(
        monkeypatch,
        tool_response,
        _response(content="Сейчас 12:00"),
    )
    monkeypatch.setitem(
        agent_naive.DISPATCH,
        "get_current_time",
        lambda timezone="Europe/Moscow": f"2026-08-05T12:00:00+03:00 {timezone}",
    )

    result = agent_naive.run_agent("Сколько сейчас времени?")

    assert result["answer"] == "Сейчас 12:00"
    assert result["steps"] == 2
    assert result["trace"][0]["tool_name"] == "get_current_time"
    assert result["trace"][0]["llm_input_tokens"] == 10
    second_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert second_messages[2] is tool_response.choices[0].message
    assert second_messages[3]["role"] == "tool"
    assert second_messages[3]["tool_call_id"] == "call_1"
    assert "2026-08-05T12:00:00+03:00" in second_messages[3]["content"]


def test_system_prompt_requires_nonempty_final_answer_after_tool(monkeypatch) -> None:
    client = _mock_client(
        monkeypatch,
        _response(tool_calls=[_tool_call("get_current_time", "{}")]),
        _response(content="Текущее время: 12:00"),
    )
    monkeypatch.setitem(agent_naive.DISPATCH, "get_current_time", lambda: "12:00")

    result = agent_naive.run_agent("Скажи время")

    first_messages = client.chat.completions.create.call_args_list[0].kwargs["messages"]
    prompt = first_messages[0]["content"]
    assert first_messages[0]["role"] == "system"
    assert "только доступные" in prompt
    assert "непустой финальный" in prompt
    assert "не раскрывай внутренние рассуждения" in prompt
    assert "запроси подтверждение" in prompt
    assert result["answer"] == "Текущее время: 12:00"
    assert result["steps"] == 2


def test_run_agent_returns_unknown_tool_error_to_model(monkeypatch) -> None:
    client = _mock_client(
        monkeypatch,
        _response(tool_calls=[_tool_call("get_user_balance", "{}")]),
        _response(content="Такого инструмента нет"),
    )

    result = agent_naive.run_agent("Вызови get_user_balance")

    assert result["answer"] == "Такого инструмента нет"
    tool_result = result["trace"][0]["tool_result"]
    assert "Неизвестный инструмент 'get_user_balance'" in tool_result
    assert "search_knowledge_base" in tool_result
    assert client.chat.completions.create.call_count == 2


def test_run_agent_returns_invalid_json_error_to_model(monkeypatch) -> None:
    client = _mock_client(
        monkeypatch,
        _response(
            tool_calls=[_tool_call("get_current_time", '{"timezone":')]
        ),
        _response(content="Не удалось выполнить инструмент"),
    )

    result = agent_naive.run_agent("Узнай время")

    assert result["answer"] == "Не удалось выполнить инструмент"
    assert result["trace"][0]["tool_args"] == {}
    assert "Невалидные JSON-аргументы" in result["trace"][0]["tool_result"]
    tool_message = client.chat.completions.create.call_args_list[1].kwargs["messages"][3]
    assert "Невалидные JSON-аргументы" in tool_message["content"]


def test_run_agent_stops_after_max_steps(monkeypatch) -> None:
    _mock_client(
        monkeypatch,
        _response(tool_calls=[_tool_call("get_current_time", "{}", call_id="call_1")]),
        _response(tool_calls=[_tool_call("get_current_time", "{}", call_id="call_2")]),
    )
    monkeypatch.setitem(agent_naive.DISPATCH, "get_current_time", lambda: "12:00")

    result = agent_naive.run_agent("Продолжай узнавать время", max_steps=2)

    assert result["answer"] == ""
    assert result["steps"] == 2
    assert result["error"] == "Агент остановлен после достижения max_steps=2."
    assert [entry["step"] for entry in result["trace"]] == [1, 2]


def test_search_knowledge_base_uses_rag_fragment(monkeypatch) -> None:
    async def fake_search(query: str) -> str:
        assert query == "VPN 691"
        return "Проверьте учётные данные VPN."

    monkeypatch.setattr(naive_agent_tools, "_search_rag_fragment", fake_search)

    assert (
        naive_agent_tools.search_knowledge_base("VPN 691")
        == "Проверьте учётные данные VPN."
    )


def test_tool_schemas_have_two_sentence_descriptions_and_allowlist() -> None:
    assert set(naive_agent_tools.DISPATCH) == {
        "search_knowledge_base",
        "get_current_time",
        "send_telegram_message",
    }
    for tool in naive_agent_tools.TOOLS:
        assert tool["function"]["description"].count(".") >= 2


def test_trace_truncates_tool_result_but_history_keeps_full_value(monkeypatch) -> None:
    client = _mock_client(
        monkeypatch,
        _response(tool_calls=[_tool_call("get_current_time", "{}")]),
        _response(content="Готово"),
    )
    monkeypatch.setitem(agent_naive.DISPATCH, "get_current_time", lambda: "x" * 250)

    result = agent_naive.run_agent("Узнай время")

    assert len(result["trace"][0]["tool_result"]) == 200
    messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert messages[3]["content"] == "x" * 250


def test_openai_error_keeps_result_contract(monkeypatch) -> None:
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("provider unavailable")
    monkeypatch.setattr(agent_naive, "_build_client", lambda: client)

    result = agent_naive.run_agent("Ответь")

    assert set(result) == {"answer", "steps", "trace", "error"}
    assert result["steps"] == 1
    assert "Ошибка OpenAI API" in result["error"]
