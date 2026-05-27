import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from openai import OpenAI

from app.prompts.loader import render_system_prompt
from app.tools.handlers import TOOL_HANDLERS
from app.tools.schemas import TOOLS


load_dotenv()

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "tool_calls.jsonl"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8",
)


def _log_event(event: str, payload: dict[str, Any]) -> None:
    log_record = {
        "event": event,
        **payload,
    }
    logging.info(json.dumps(log_record, ensure_ascii=False))


def _build_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Не найден OPENAI_API_KEY. Создайте файл .env на основе .env.example "
            "и добавьте API-ключ."
        )

    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def validate_tool_schemas() -> None:
    """Validate JSON Schema blocks before sending tools to the model."""
    for tool in TOOLS:
        parameters_schema = tool["function"]["parameters"]
        Draft202012Validator.check_schema(parameters_schema)


def _run_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name not in TOOL_HANDLERS:
        return {
            "ok": False,
            "error": f"Неизвестный инструмент: {tool_name}",
        }

    handler = TOOL_HANDLERS[tool_name]
    result = handler(**arguments)

    return {
        "ok": True,
        "data": result,
    }


def ask_support_assistant(user_input: str) -> str:
    """
    Full Function Calling cycle:
    user request -> model tool_call decision -> function execution ->
    tool result -> final model answer.
    """
    validate_tool_schemas()
    client = _build_client()

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": render_system_prompt(
                product_name="ИИ-ассистент техподдержки"
            ),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    _log_event(
        "input",
        {
            "user_input": user_input,
            "model": MODEL,
        },
    )

    first_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    first_message = first_response.choices[0].message
    tool_calls = first_message.tool_calls or []

    if not tool_calls:
        final_text = first_message.content or ""

        _log_event(
            "final_without_tool",
            {
                "final_answer": final_text,
                "total_tokens": (
                    first_response.usage.total_tokens
                    if first_response.usage
                    else None
                ),
            },
        )

        return final_text

    messages.append(first_message.model_dump(exclude_none=True))

    for tool_call in tool_calls:
        tool_name = tool_call.function.name

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as error:
            arguments = {}
            tool_result = {
                "ok": False,
                "error": f"Не удалось распарсить аргументы tool_call: {error}",
            }
        else:
            tool_result = _run_tool(tool_name, arguments)

        _log_event(
            "tool_call",
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_result": tool_result,
            },
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )

    second_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )

    final_message = second_response.choices[0].message
    final_text = final_message.content or ""

    _log_event(
        "final_with_tool",
        {
            "final_answer": final_text,
            "total_tokens": (
                second_response.usage.total_tokens
                if second_response.usage
                else None
            ),
        },
    )

    return final_text
