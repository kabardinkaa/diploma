from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.core.config import get_settings
from app.tools.handlers import search_knowledge_base as search_local_knowledge_base


NOT_FOUND = "По запросу ничего не найдено в базе знаний."


async def _search_rag_fragment(query: str) -> str | None:
    from app.services.rag import RAGService

    service = RAGService(get_settings())
    try:
        await service.build()
        contexts = await service.retrieve_contexts(query, top_k=1)
        return contexts[0]["text"] if contexts else None
    finally:
        await service.close()


def search_knowledge_base(query: str) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        return NOT_FOUND
    try:
        fragment = asyncio.run(_search_rag_fragment(normalized_query))
        if fragment:
            return fragment
    except Exception as exc:
        logging.warning("RAG search unavailable, using local fallback: %s", exc)

    results = search_local_knowledge_base(normalized_query).get("results", [])
    return str(results[0]["content"]) if results else NOT_FOUND


def get_current_time(timezone: str = "Europe/Moscow") -> str:
    return datetime.now(ZoneInfo(timezone)).isoformat()


def send_telegram_message(chat_id: str, text: str) -> str:
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Ищет top-1 фрагмент во внутренней базе знаний техподдержки. "
                "Вызывай инструмент, когда нужны корпоративные инструкции."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Возвращает текущее время в заданной IANA timezone без сети. "
                "Вызывай инструмент, когда задача зависит от даты или времени."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "default": "Europe/Moscow",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram_message",
            "description": (
                "Имитирует отправку текста в Telegram-чат через локальный print. "
                "Вызывай инструмент при явной просьбе отправить сообщение."
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
        },
    },
]


DISPATCH: dict[str, Callable[..., str]] = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}


def dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    handler = DISPATCH.get(name)
    if handler is None:
        available = ", ".join(sorted(DISPATCH))
        return f"Неизвестный инструмент '{name}'. Доступны: {available}."
    try:
        return str(handler(**arguments))
    except Exception as exc:
        return f"Инструмент '{name}' завершился с ошибкой: {exc}"
