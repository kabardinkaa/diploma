from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.llm.testing_utils import (
    build_chat_messages,
    calculate_llm_cost,
    extract_json_object,
    extract_tool_calls,
)
from app.schemas.chat import ChatRequest, ChatResponse, Message
from app.services.llm import LLMService


def test_build_chat_messages_keeps_stable_role_order() -> None:
    messages = build_chat_messages(
        system="Ты ассистент техподдержки.",
        history=[
            {"role": "user", "content": "Не работает VPN"},
            {"role": "assistant", "content": "Уточните ошибку"},
        ],
        user="Ошибка 691",
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_build_chat_messages_escapes_braces_from_user_input() -> None:
    messages = build_chat_messages(
        system="Ты ассистент.",
        user="Подставь {secret} и {token}",
    )

    assert messages[-1]["content"] == "Подставь {{secret}} и {{token}}"


def test_extract_json_object_from_raw_json() -> None:
    parsed = extract_json_object('{"category": "vpn", "priority": "high"}')

    assert parsed == {"category": "vpn", "priority": "high"}


def test_extract_json_object_from_markdown_fence() -> None:
    parsed = extract_json_object(
        """
        ```json
        {"status": "ok", "ticket_required": true}
        ```
        """
    )

    assert parsed["status"] == "ok"
    assert parsed["ticket_required"] is True


def test_extract_json_object_raises_on_malformed_json() -> None:
    with pytest.raises(ValueError, match="Malformed JSON response"):
        extract_json_object('{"status": "ok"')


def test_extract_tool_calls_from_openai_like_dict() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "create_ticket",
                                "arguments": '{"category": "access"}',
                            },
                        }
                    ]
                }
            }
        ]
    }

    tool_calls = extract_tool_calls(response)

    assert tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "name": "create_ticket",
            "arguments": '{"category": "access"}',
        }
    ]


def test_chat_request_rejects_empty_message_content() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=[Message(role="user", content="")])


def test_chat_request_rejects_too_long_message_content() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=[Message(role="user", content="x" * 4001)])


def test_message_repr_does_not_expose_raw_pii() -> None:
    message = Message(role="user", content="Мой email user@example.com")

    assert "user@example.com" not in repr(message)


def test_calculate_llm_cost_from_usage_dict() -> None:
    cost = calculate_llm_cost(
        {"prompt_tokens": 1000, "completion_tokens": 500},
        input_price_per_1k=0.01,
        output_price_per_1k=0.03,
    )

    assert cost == 0.025


@pytest.mark.asyncio
async def test_llm_service_cache_hit_uses_cache_without_openai_call(mocker) -> None:
    openai_client = mocker.Mock()
    openai_client.chat.completions.create = mocker.AsyncMock()

    cache: dict[str, str] = {}
    settings = SimpleNamespace(
        llm=SimpleNamespace(default_model="mock-model"),
        cache_ttl_seconds=60,
    )
    service = LLMService(openai_client, cache, settings)

    request = ChatRequest(messages=[Message(role="user", content="Привет")])
    key = service._cache_key(request)

    cache[key] = ChatResponse(
        content="Ответ из кеша",
        model="mock-model",
        finish_reason="stop",
        cached=False,
    ).model_dump_json()

    response = await service.complete(request)

    assert response.cached is True
    assert response.content == "Ответ из кеша"
    openai_client.chat.completions.create.assert_not_called()