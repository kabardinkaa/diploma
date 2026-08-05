from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.tools.naive_agent_tools import (
    get_current_time,
    search_knowledge_base,
    send_telegram_message,
)


REACT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Ищет корпоративную инструкцию по узкому запросу. "
                "Вызывай при вопросах о внутренних процедурах техподдержки. "
                "Возвращает top-1 текстовый фрагмент для следующего действия. "
                "Не ищет в интернете и не выполняет пишущих действий."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Получает текущее время в заданной IANA timezone. "
                "Вызывай, когда следующий шаг зависит от текущей даты или времени. "
                "Возвращает ISO datetime, пригодный для подготовки сообщения. "
                "Не обращается к сети и не планирует действия пользователя."
            ),
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram_message",
            "description": (
                "Имитирует отправку готового текста в указанный Telegram-чат. "
                "Вызывай только после явного подтверждения отправки пользователем. "
                "Возвращает строковый статус, который завершает пишущее действие. "
                "Не использует Telegram API и не редактирует содержание сообщения."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["chat_id", "text"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


REACT_DISPATCH: dict[str, Callable[..., str]] = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}


def dispatch_react_tool(name: str, arguments: dict[str, Any]) -> str:
    handler = REACT_DISPATCH.get(name)
    if handler is None:
        available = ", ".join(sorted(REACT_DISPATCH))
        return f"Неизвестный инструмент '{name}'. Доступны: {available}."
    try:
        return str(handler(**arguments))
    except Exception as exc:
        return f"Инструмент '{name}' завершился с ошибкой: {exc}"
