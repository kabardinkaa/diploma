from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services import agent_react
from app.tools import react_agent_tools


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
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _mock_client(monkeypatch, *responses: SimpleNamespace) -> Mock:
    client = Mock()
    client.chat.completions.create = Mock(side_effect=responses)
    monkeypatch.setattr(agent_react, "_build_client", lambda: client)
    return client


def _tool_then_critic_then_final(monkeypatch, *, verdict: str = "OK") -> Mock:
    monkeypatch.setattr(
        agent_react,
        "dispatch_react_tool",
        lambda name, arguments: "2026-08-05T12:00:00+03:00",
    )
    return _mock_client(
        monkeypatch,
        _response(
            content="Проверю текущее время.",
            tool_calls=[
                _tool_call(
                    "get_current_time",
                    '{"timezone":"Europe/Moscow"}',
                )
            ],
        ),
        _response(content=verdict, prompt_tokens=4, completion_tokens=1),
        _response(content="В Москве сейчас 12:00."),
    )


def test_final_answer_without_tool_call(monkeypatch) -> None:
    client = _mock_client(monkeypatch, _response(content="Готовый ответ"))

    result = agent_react.run_react_with_reflection("Ответь кратко")

    assert result["answer"] == "Готовый ответ"
    assert result["steps"] == 1
    assert result["revisions_used"] == 0
    assert result["trace"][0]["tool_name"] is None
    assert client.chat.completions.create.call_args.kwargs["tool_choice"] == "auto"


def test_successful_single_tool_call(monkeypatch) -> None:
    client = _tool_then_critic_then_final(monkeypatch)

    result = agent_react.run_react_with_reflection("Сколько времени в Москве?")

    assert result["answer"] == "В Москве сейчас 12:00."
    assert result["steps"] == 2
    assert result["trace"][0]["tool_name"] == "get_current_time"
    messages = client.chat.completions.create.call_args_list[2].kwargs["messages"]
    assert messages[3]["role"] == "tool"
    assert messages[3]["content"] == "2026-08-05T12:00:00+03:00"


def test_only_first_of_multiple_tool_calls_is_executed(monkeypatch) -> None:
    dispatch = Mock(return_value="first result")
    monkeypatch.setattr(agent_react, "dispatch_react_tool", dispatch)
    client = _mock_client(
        monkeypatch,
        _response(
            tool_calls=[
                _tool_call("get_current_time", '{"timezone":"UTC"}', call_id="one"),
                _tool_call("search_knowledge_base", '{"query":"VPN"}', call_id="two"),
            ]
        ),
        _response(content="OK"),
        _response(content="Готово"),
    )

    result = agent_react.run_react_with_reflection("Составная задача")

    dispatch.assert_called_once_with("get_current_time", {"timezone": "UTC"})
    assert "только первый tool-call" in result["trace"][0]["observation"]
    next_messages = client.chat.completions.create.call_args_list[2].kwargs["messages"]
    assert len(next_messages[2]["tool_calls"]) == 1
    assert next_messages[2]["tool_calls"][0]["id"] == "one"


def test_unknown_tool_becomes_observation(monkeypatch) -> None:
    client = _mock_client(
        monkeypatch,
        _response(tool_calls=[_tool_call("get_user_balance", "{}")]),
        _response(content="OK"),
        _response(content="Такого инструмента нет"),
    )

    result = agent_react.run_react_with_reflection("Узнай баланс")

    assert "Неизвестный инструмент 'get_user_balance'" in result["trace"][0][
        "observation"
    ]
    assert result["answer"] == "Такого инструмента нет"


def test_invalid_json_becomes_observation(monkeypatch) -> None:
    client = _mock_client(
        monkeypatch,
        _response(tool_calls=[_tool_call("get_current_time", '{"timezone":')]),
        _response(content="OK"),
        _response(content="Не удалось разобрать аргументы"),
    )

    result = agent_react.run_react_with_reflection("Узнай время")

    assert "Невалидные JSON-аргументы" in result["trace"][0]["observation"]
    assert result["trace"][0]["tool_args"] == {}


def test_tool_exception_becomes_observation(monkeypatch) -> None:
    def broken_tool() -> str:
        raise RuntimeError("tool failed")

    monkeypatch.setitem(react_agent_tools.REACT_DISPATCH, "broken_tool", broken_tool)
    client = _mock_client(
        monkeypatch,
        _response(tool_calls=[_tool_call("broken_tool", "{}")]),
        _response(content="OK"),
        _response(content="Инструмент недоступен"),
    )

    result = agent_react.run_react_with_reflection("Выполни действие")

    assert "завершился с ошибкой: tool failed" in result["trace"][0]["observation"]
    assert "error" not in result


def test_critic_ok_is_recorded(monkeypatch) -> None:
    client = _tool_then_critic_then_final(monkeypatch, verdict="OK")

    result = agent_react.run_react_with_reflection("Сколько времени?")

    assert result["trace"][0]["critic_verdict"] == "OK"
    assert result["trace"][0]["revision_applied"] is False
    critic_kwargs = client.chat.completions.create.call_args_list[1].kwargs
    assert "tools" not in critic_kwargs
    assert "tool_choice" not in critic_kwargs


def test_empty_critic_response_is_safe(monkeypatch) -> None:
    _tool_then_critic_then_final(monkeypatch, verdict="")

    result = agent_react.run_react_with_reflection("Сколько времени?")

    assert result["answer"] == "В Москве сейчас 12:00."
    assert result["trace"][0]["critic_verdict"] == "INVALID"
    assert result["revisions_used"] == 0


def test_critic_error_does_not_stop_actor(monkeypatch) -> None:
    monkeypatch.setattr(agent_react, "dispatch_react_tool", lambda name, args: "now")
    client = Mock()
    client.chat.completions.create = Mock(
        side_effect=[
            _response(tool_calls=[_tool_call("get_current_time", "{}")]),
            RuntimeError("critic unavailable"),
            _response(content="Финальный ответ"),
        ]
    )
    monkeypatch.setattr(agent_react, "_build_client", lambda: client)

    result = agent_react.run_react_with_reflection("Сколько времени?")

    assert result["answer"] == "Финальный ответ"
    assert result["trace"][0]["critic_verdict"].startswith("ERROR:")
    assert "error" not in result


def test_critic_revision_is_added_to_next_iteration(monkeypatch) -> None:
    client = _tool_then_critic_then_final(
        monkeypatch,
        verdict="REVISE: уточни часовой пояс",
    )

    result = agent_react.run_react_with_reflection("Сколько времени?")

    next_messages = client.chat.completions.create.call_args_list[2].kwargs["messages"]
    assert result["revisions_used"] == 1
    assert result["trace"][0]["revision_applied"] is True
    assert any("уточни часовой пояс" in item.get("content", "") for item in next_messages)


def test_max_revisions_is_global_and_limited_to_two(monkeypatch) -> None:
    responses: list[SimpleNamespace] = []
    for number in range(3):
        responses.extend(
            [
                _response(
                    tool_calls=[
                        _tool_call(
                            "get_current_time",
                            '{"timezone":"UTC"}',
                            call_id=f"call_{number}",
                        )
                    ]
                ),
                _response(content=f"REVISE: причина {number}"),
            ]
        )
    responses.append(_response(content="Финал"))
    monkeypatch.setattr(agent_react, "dispatch_react_tool", lambda name, args: "now")
    _mock_client(monkeypatch, *responses)

    result = agent_react.run_react_with_reflection("Повтори проверку")

    assert result["revisions_used"] == 2
    assert [item["revision_applied"] for item in result["trace"][:3]] == [
        True,
        True,
        False,
    ]
    assert result["trace"][2]["critic_verdict"] == "REVISE: причина 2"


def test_stops_at_max_iterations(monkeypatch) -> None:
    responses: list[SimpleNamespace] = []
    for number in range(8):
        responses.extend(
            [
                _response(
                    tool_calls=[
                        _tool_call("get_current_time", "{}", call_id=f"call_{number}")
                    ]
                ),
                _response(content="OK"),
            ]
        )
    monkeypatch.setattr(agent_react, "dispatch_react_tool", lambda name, args: "now")
    _mock_client(monkeypatch, *responses)

    result = agent_react.run_react_with_reflection("Не завершай", max_iterations=8)

    assert result["answer"] == "Превышен лимит итераций"
    assert result["error"] == "Превышен лимит итераций"
    assert result["steps"] == 8
    assert len(result["trace"]) == 8


def test_timeout_returns_explicit_contract(monkeypatch) -> None:
    _mock_client(monkeypatch, _response(content="unused"))

    def timeout(operation, seconds):
        raise agent_react.IterationTimeout("deadline")

    monkeypatch.setattr(agent_react, "_run_with_timeout", timeout)

    result = agent_react.run_react_with_reflection("Медленная задача")

    assert result["answer"] == "Timeout"
    assert result["error"] == "Timeout"
    assert result["trace"][0]["timeout"] is True


def test_timeout_before_tool_does_not_dispatch(monkeypatch) -> None:
    _mock_client(
        monkeypatch,
        _response(tool_calls=[_tool_call("get_current_time", "{}")]),
    )
    dispatch = Mock(return_value="now")
    monkeypatch.setattr(agent_react, "dispatch_react_tool", dispatch)
    remaining = Mock(
        side_effect=[5.0, agent_react.IterationTimeout("deadline before tool")]
    )
    monkeypatch.setattr(agent_react, "_remaining", remaining)

    result = agent_react.run_react_with_reflection("Узнай время")

    assert result["answer"] == "Timeout"
    dispatch.assert_not_called()


def test_timeout_before_critic_does_not_call_critic(monkeypatch) -> None:
    client = _mock_client(
        monkeypatch,
        _response(tool_calls=[_tool_call("get_current_time", "{}")]),
    )
    dispatch = Mock(return_value="now")
    monkeypatch.setattr(agent_react, "dispatch_react_tool", dispatch)
    remaining = Mock(
        side_effect=[5.0, 2.0, agent_react.IterationTimeout("deadline before critic")]
    )
    monkeypatch.setattr(agent_react, "_remaining", remaining)

    result = agent_react.run_react_with_reflection("Узнай время")

    assert result["answer"] == "Timeout"
    dispatch.assert_called_once()
    assert client.chat.completions.create.call_count == 1


def test_expired_deadline_never_starts_write_tool(monkeypatch) -> None:
    _mock_client(
        monkeypatch,
        _response(
            tool_calls=[
                _tool_call(
                    "send_telegram_message",
                    '{"chat_id":"12345","text":"Сообщение"}',
                )
            ]
        ),
    )
    dispatch = Mock(return_value="sent")
    monkeypatch.setattr(agent_react, "dispatch_react_tool", dispatch)
    monkeypatch.setattr(agent_react, "_remaining", Mock(side_effect=[5.0, 0.05]))

    result = agent_react.run_react_with_reflection("Отправь сообщение")

    assert result["answer"] == "Timeout"
    dispatch.assert_not_called()


def test_actor_tool_critic_share_one_remaining_budget(monkeypatch) -> None:
    client = _mock_client(
        monkeypatch,
        _response(tool_calls=[_tool_call("get_current_time", "{}")]),
        _response(content="OK"),
        _response(content="Финал"),
    )
    monkeypatch.setattr(agent_react, "dispatch_react_tool", lambda name, args: "now")
    monkeypatch.setattr(
        agent_react,
        "_remaining",
        Mock(side_effect=[9.0, 6.0, 4.0, 3.0]),
    )
    timeout_calls: list[float] = []

    def run_now(operation, timeout):
        timeout_calls.append(timeout)
        return operation()

    monkeypatch.setattr(agent_react, "_run_with_timeout", run_now)

    result = agent_react.run_react_with_reflection("Узнай время")

    assert result["answer"] == "Финал"
    assert timeout_calls == [9.0, 4.0, 3.0]
    assert client.chat.completions.create.call_args_list[0].kwargs["timeout"] == 9.0
    assert client.chat.completions.create.call_args_list[1].kwargs["timeout"] == 4.0


def test_usage_sums_actor_and_critic_calls(monkeypatch) -> None:
    monkeypatch.setattr(agent_react, "dispatch_react_tool", lambda name, args: "now")
    _mock_client(
        monkeypatch,
        _response(
            tool_calls=[_tool_call("get_current_time", "{}")],
            prompt_tokens=10,
            completion_tokens=2,
        ),
        _response(content="OK", prompt_tokens=3, completion_tokens=1),
        _response(content="Финал", prompt_tokens=7, completion_tokens=4),
    )

    result = agent_react.run_react_with_reflection("Узнай время")

    assert result["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 7,
        "total_tokens": 27,
    }
    assert result["trace"][0]["total_tokens"] == 16
    assert result["trace"][1]["total_tokens"] == 11


def test_usage_normalizes_input_output_tokens() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=8, output_tokens=3)
    )

    assert agent_react._usage(response) == {
        "prompt_tokens": 8,
        "completion_tokens": 3,
        "total_tokens": 11,
    }


def test_strict_tool_schemas_and_allowlist() -> None:
    assert set(react_agent_tools.REACT_DISPATCH) >= {
        "search_knowledge_base",
        "get_current_time",
        "send_telegram_message",
    }
    for tool in react_agent_tools.REACT_TOOLS:
        function = tool["function"]
        assert function["strict"] is True
        assert function["parameters"]["additionalProperties"] is False
        assert set(function["parameters"]["required"]) == set(
            function["parameters"]["properties"]
        )
        assert function["description"].count(".") >= 4


def test_provocative_scenario_finishes_without_tools(monkeypatch) -> None:
    client = _mock_client(
        monkeypatch,
        _response(content="HTTP передаёт данные без шифрования, HTTPS — с TLS."),
    )

    result = agent_react.run_react_with_reflection("Чем HTTP отличается от HTTPS?")

    assert result["answer"].startswith("HTTP")
    assert result["trace"][0]["tool_name"] is None
    assert client.chat.completions.create.call_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iterations": 7}, "max_iterations"),
        ({"max_iterations": 30}, "max_iterations"),
        ({"timeout_per_iteration_sec": 4}, "timeout_per_iteration_sec"),
        ({"max_revisions": 3}, "max_revisions"),
    ],
)
def test_invalid_limits_are_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        agent_react.run_react_with_reflection("Задача", **kwargs)


def test_agent_naive_baseline_is_byte_identical() -> None:
    baseline = Path(agent_react.__file__).with_name("agent_naive.py").read_bytes()

    assert hashlib.sha256(baseline).hexdigest() == (
        "db07cf8de5d2937eb41194fd4d50873c7db2c2e2659dd071d79265ad49910901"
    )
