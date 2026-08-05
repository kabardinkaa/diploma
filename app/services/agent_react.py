from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import structlog
from openai import OpenAI

from app.core.config import get_settings
from app.tools.react_agent_tools import REACT_TOOLS, dispatch_react_tool


log = structlog.get_logger("react-agent")
WRITE_TOOL_MIN_REMAINING_SEC = 0.1
SYSTEM_PROMPT = """Ты ReAct-агент внутренней техподдержки.
Самостоятельно выбирай доступные инструменты и их порядок; при необходимости
декомпозируй задачу. Перед действием дай одно короткое предложение о том, что и
зачем делаешь, затем вызови ровно один инструмент. После observation пересмотри
следующий шаг. Как только данных достаточно, дай финальный ответ без tool-call.
Не выдумывай данные и честно сообщай, если инструментов недостаточно или поиск
ничего не нашёл. Пишущие действия разрешены только после явного подтверждения
пользователя. Не раскрывай скрытые рассуждения или chain of thought."""
CRITIC_PROMPT = """Ты краткий critic агентной траектории. Проверь, помогает ли
observation решить исходную задачу и нужен ли пересмотр следующего шага.
Верни только OK или REVISE: <краткая причина>. Не вызывай инструменты и не
раскрывай внутренние рассуждения."""


class IterationTimeout(TimeoutError):
    pass


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


def _run_with_timeout(operation: Callable[[], Any], timeout: float) -> Any:
    if timeout <= 0:
        raise IterationTimeout("iteration deadline exceeded")
    output: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            output.put((True, operation()))
        except Exception as exc:
            output.put((False, exc))

    threading.Thread(target=worker, daemon=True).start()
    try:
        ok, value = output.get(timeout=timeout)
    except queue.Empty as exc:
        raise IterationTimeout("iteration deadline exceeded") from exc
    if not ok:
        raise value
    return value


def _remaining(deadline: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise IterationTimeout("iteration deadline exceeded")
    return remaining


def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    deadline: float,
) -> str:
    remaining = _remaining(deadline)
    if name == "send_telegram_message" and remaining < WRITE_TOOL_MIN_REMAINING_SEC:
        raise IterationTimeout("insufficient time for write tool")
    return dispatch_react_tool(name, arguments)


def _usage(response: Any) -> dict[str, int]:
    raw = response.usage if response and response.usage else None
    if raw is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(raw, dict):
        data = raw
    elif hasattr(raw, "model_dump"):
        data = raw.model_dump()
    else:
        data = vars(raw)
    prompt = int(data.get("prompt_tokens") or data.get("input_tokens") or 0)
    completion = int(
        data.get("completion_tokens") or data.get("output_tokens") or 0
    )
    total = int(data.get("total_tokens") or prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _add_usage(total: dict[str, int], current: dict[str, int]) -> None:
    for key in total:
        total[key] += current[key]


def _finish(
    answer: str,
    steps: int,
    trace: list[dict[str, Any]],
    usage: dict[str, int],
    revisions_used: int,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "answer": answer,
        "steps": steps,
        "trace": trace,
        "usage": usage,
        "revisions_used": revisions_used,
    }
    if error:
        result["error"] = error
    log.info("react.usage", **usage, steps=steps, revisions_used=revisions_used)
    return result


def _assistant_tool_message(message: Any, call: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        ],
    }


def _parse_arguments(name: str, raw_arguments: str) -> tuple[dict[str, Any], str | None]:
    try:
        arguments = json.loads(raw_arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("arguments должны быть JSON-объектом")
        return arguments, None
    except (json.JSONDecodeError, ValueError) as exc:
        return {}, f"Невалидные JSON-аргументы для '{name}': {exc}"


def _critic_verdict(content: str | None) -> str:
    verdict = (content or "").strip()
    if verdict == "OK":
        return verdict
    if verdict.startswith("REVISE:") and verdict.removeprefix("REVISE:").strip():
        return verdict
    return "INVALID"


def _validate_limits(
    max_iterations: int,
    timeout_per_iteration_sec: float,
    max_revisions: int,
) -> None:
    if not 8 <= max_iterations <= 20:
        raise ValueError("max_iterations должен быть в диапазоне 8..20")
    if not 5 <= timeout_per_iteration_sec <= 15:
        raise ValueError("timeout_per_iteration_sec должен быть в диапазоне 5..15")
    if not 0 <= max_revisions <= 2:
        raise ValueError("max_revisions должен быть в диапазоне 0..2")


def run_react_with_reflection(
    question: str,
    max_iterations: int = 10,
    timeout_per_iteration_sec: float = 10.0,
    max_revisions: int = 2,
    model_main: str = "gpt-5.4-mini",
    model_critic: str = "gpt-5.4-mini",
) -> dict:
    _validate_limits(max_iterations, timeout_per_iteration_sec, max_revisions)
    trace: list[dict[str, Any]] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    revisions_used = 0
    revision_history: list[str] = []
    try:
        client = _build_client()
    except Exception as exc:
        return _finish("", 0, trace, usage_total, 0, f"OpenAI client: {exc}")

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for iteration in range(1, max_iterations + 1):
        started = time.perf_counter()
        deadline = started + timeout_per_iteration_sec
        iteration_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        entry: dict[str, Any] = {
            "iteration": iteration,
            "tool_name": None,
            "tool_args": {},
            "observation": None,
            "latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "critic_verdict": None,
            "revision_applied": False,
            "model": model_main,
        }
        try:
            request_timeout = _remaining(deadline)
            response = _run_with_timeout(
                lambda: client.chat.completions.create(
                    model=model_main,
                    messages=messages,
                    tools=REACT_TOOLS,
                    tool_choice="auto",
                    timeout=request_timeout,
                ),
                request_timeout,
            )
            main_usage = _usage(response)
            _add_usage(iteration_usage, main_usage)
            _add_usage(usage_total, main_usage)
            message = response.choices[0].message if response.choices else None
            if message is None:
                raise RuntimeError("Основная модель вернула ответ без message")

            if not message.tool_calls:
                messages.append(message)
                answer = (message.content or "").strip()
                if not answer:
                    raise RuntimeError("Основная модель вернула пустой ответ")
                entry.update(iteration_usage)
                entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                trace.append(entry)
                log.info("react.iteration", **entry)
                return _finish(
                    answer,
                    iteration,
                    trace,
                    usage_total,
                    revisions_used,
                )

            call = message.tool_calls[0]
            messages.append(_assistant_tool_message(message, call))
            name = call.function.name
            arguments, parse_error = _parse_arguments(
                name,
                call.function.arguments,
            )
            if parse_error:
                observation = parse_error
            else:
                observation = _execute_tool(name, arguments, deadline)
            if len(message.tool_calls) > 1:
                observation = (
                    f"{observation}\nОбработан только первый tool-call; "
                    "следующий инструмент выбери на новой итерации."
                )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": observation}
            )

            critic_timeout = _remaining(deadline)
            critic_context = (
                f"Исходная задача: {question}\n"
                f"Краткий план actor: {message.content or 'не указан'}\n"
                f"Инструмент: {name}\nАргументы: "
                f"{json.dumps(arguments, ensure_ascii=False)}\n"
                f"Observation: {observation}\n"
                f"Предыдущие ревизии: {revision_history or 'нет'}"
            )
            try:
                critic_response = _run_with_timeout(
                    lambda: client.chat.completions.create(
                        model=model_critic,
                        messages=[
                            {"role": "system", "content": CRITIC_PROMPT},
                            {"role": "user", "content": critic_context},
                        ],
                        timeout=critic_timeout,
                    ),
                    critic_timeout,
                )
                critic_usage = _usage(critic_response)
                _add_usage(iteration_usage, critic_usage)
                _add_usage(usage_total, critic_usage)
                critic_message = (
                    critic_response.choices[0].message
                    if critic_response.choices
                    else None
                )
                verdict = _critic_verdict(
                    critic_message.content if critic_message else None
                )
                log.info("react.critic", iteration=iteration, verdict=verdict, **critic_usage)
            except IterationTimeout:
                log.warning("react.critic_timeout", iteration=iteration)
                raise
            except Exception as exc:
                verdict = f"ERROR: {exc}"
                log.warning("react.critic_failed", iteration=iteration, error=str(exc))

            revision_applied = False
            if verdict.startswith("REVISE:") and revisions_used < max_revisions:
                revisions_used += 1
                revision_applied = True
                feedback = verdict.removeprefix("REVISE:").strip()
                revision_history.append(feedback)
                messages.append(
                    {
                        "role": "system",
                        "content": f"Critic feedback для следующего шага: {feedback}",
                    }
                )

            entry.update(
                {
                    "tool_name": name,
                    "tool_args": arguments,
                    "observation": observation[:500],
                    "critic_verdict": verdict,
                    "revision_applied": revision_applied,
                    **iteration_usage,
                }
            )
        except IterationTimeout:
            entry.update(iteration_usage)
            entry.update({"observation": "Timeout", "timeout": True, "error": "Timeout"})
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            trace.append(entry)
            log.info("react.iteration", **entry)
            return _finish(
                "Timeout",
                iteration,
                trace,
                usage_total,
                revisions_used,
                "Timeout",
            )
        except Exception as exc:
            entry.update(iteration_usage)
            entry["error"] = str(exc)
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            trace.append(entry)
            log.info("react.iteration", **entry)
            return _finish(
                "",
                iteration,
                trace,
                usage_total,
                revisions_used,
                str(exc),
            )

        entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        trace.append(entry)
        log.info("react.iteration", **entry)

    answer = "Превышен лимит итераций"
    return _finish(
        answer,
        max_iterations,
        trace,
        usage_total,
        revisions_used,
        answer,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ReAct-агент блока 6.2")
    parser.add_argument("question", help="Задача для агента")
    parser.add_argument("--trace", action="store_true", help="Вывести usage и trace")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--timeout-per-iteration", type=float, default=10.0)
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--model-main", default="gpt-5.4-mini")
    parser.add_argument("--model-critic", default="gpt-5.4-mini")
    args = parser.parse_args(argv)
    try:
        result = run_react_with_reflection(
            args.question,
            max_iterations=args.max_iterations,
            timeout_per_iteration_sec=args.timeout_per_iteration,
            max_revisions=args.max_revisions,
            model_main=args.model_main,
            model_critic=args.model_critic,
        )
    except ValueError as exc:
        print(str(exc))
        return 2
    print(result["answer"] or result.get("error") or "Агент не вернул результат.")
    if args.trace:
        print(
            json.dumps(
                {
                    "usage": result["usage"],
                    "revisions_used": result["revisions_used"],
                    "trace": result["trace"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
