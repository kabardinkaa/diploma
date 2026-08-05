import argparse
import json
import logging
import time
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from app.core.config import get_settings
from app.tools.naive_agent_tools import DISPATCH, TOOLS, dispatch_tool
MODEL = "gpt-5.4-mini"
SYSTEM_PROMPT = """Ты агент внутренней техподдержки. Используй только доступные
tools. После результатов tools обязательно дай пользователю непустой финальный
ответ; не завершай работу пустым сообщением и не раскрывай внутренние рассуждения.
Перед пишущим действием запроси подтверждение, если отправка явно не подтверждена."""

def _build_client() -> OpenAI:
    settings = get_settings().llm
    kwargs: dict[str, Any] = {
        "api_key": settings.api_key.get_secret_value(),
        "timeout": settings.request_timeout,
        "max_retries": settings.max_retries,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return OpenAI(**kwargs)

def _trace(
    step: int, started: float, response: Any = None,
    name: str | None = None, args: dict | None = None,
    result: str | None = None, error: str | None = None,
) -> dict:
    usage = response.usage if response else None
    entry = {
        "step": step,
        "tool_name": name,
        "tool_args": args or {},
        "tool_result": result[:200] if result else None,
        "llm_input_tokens": usage.prompt_tokens if usage else None,
        "llm_output_tokens": usage.completion_tokens if usage else None,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if error:
        entry["error"] = error
    logging.info("agent_step %s", json.dumps(entry, ensure_ascii=False))
    return entry

def _result(answer: str, steps: int, trace: list[dict], error: str | None = None) -> dict:
    payload = {"answer": answer, "steps": steps, "trace": trace}
    if error:
        payload["error"] = error
    return payload

def _run_tool(name: str, raw_args: str) -> tuple[dict, str]:
    try:
        arguments = json.loads(raw_args or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("arguments должны быть JSON-объектом")
        return arguments, dispatch_tool(name, arguments)
    except (json.JSONDecodeError, ValueError) as exc:
        return {}, f"Невалидные JSON-аргументы для '{name}': {exc}"

def run_agent(task: str, max_steps: int = 6) -> dict:
    trace: list[dict] = []
    if not task.strip() or max_steps < 1:
        error = "Задача не должна быть пустой." if not task.strip() else "max_steps < 1."
        return _result("", 0, trace, error)
    try:
        client = _build_client()
    except Exception as exc:
        return _result("", 0, trace, f"Не удалось создать OpenAI-клиент: {exc}")

    messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": task}]
    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
            )
        except Exception as exc:
            error = f"Ошибка OpenAI API: {exc}"
            trace.append(_trace(step, started, error=error))
            return _result("", step, trace, error)

        message = response.choices[0].message if response.choices else None
        if message is None:
            error = "Модель вернула ответ без сообщения."
            trace.append(_trace(step, started, response, error=error))
            return _result("", step, trace, error)
        messages.append(message)

        if not message.tool_calls:
            answer = (message.content or "").strip()
            error = None if answer else "Модель вернула пустой финальный ответ."
            trace.append(_trace(step, started, response, error=error))
            return _result(answer, step, trace, error)

        for call in message.tool_calls:
            name = call.function.name
            arguments, tool_result = _run_tool(name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": tool_result})
            trace.append(_trace(step, started, response, name, arguments, tool_result))

    error = f"Агент остановлен после достижения max_steps={max_steps}."
    return _result("", max_steps, trace, error)

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Наивный агент блока 6.1")
    parser.add_argument("task", help="Задача для агента")
    parser.add_argument("--trace", action="store_true", help="Вывести JSON-трассу")
    args = parser.parse_args(argv)
    result = run_agent(args.task)
    print(result["answer"] or result.get("error") or "Агент не вернул результат.")
    if args.trace:
        print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
    return 1 if result.get("error") else 0

if __name__ == "__main__":
    raise SystemExit(main())
