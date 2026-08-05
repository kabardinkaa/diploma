from __future__ import annotations

import argparse
import asyncio
import gc
import json
import re
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import BaseMessage

from app.core.config import get_settings
from app.eval.agent_scenarios import AGENT_SCENARIOS, AgentScenario
from app.services.agent_graph import run_custom_graph, run_prebuilt_graph
from app.services.agent_react import run_react_with_reflection


IMPLEMENTATIONS = ("react_6_2", "custom", "prebuilt")
REPEATS = 3
REACT_TIMEOUT_PER_ITERATION_SEC = 15.0
RUN_TIMEOUT_SEC = 60.0
RAW_DIR = Path("docs/agent-graph-results")
REPORT_PATH = Path("docs/agent-graph-report.md")
START_MARKER = "<!-- BENCHMARK_RESULTS_START -->"
END_MARKER = "<!-- BENCHMARK_RESULTS_END -->"
TASKS = [scenario.task for scenario in AGENT_SCENARIOS]
RAW_RUN_PATTERN = re.compile(
    r"^(bench-(\d+)-(react_6_2|custom|prebuilt)-([123]))"
    r"(?:\.attempt-(\d+))?\.json$"
)
SUMMARY_FIELDS = (
    "run_id",
    "task_id",
    "task",
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
)


async def _run_react(
    task: str,
    thread_id: str,
    *,
    timeout_per_iteration_sec: float = REACT_TIMEOUT_PER_ITERATION_SEC,
) -> dict[str, Any]:
    del thread_id
    operation = partial(
        run_react_with_reflection,
        task,
        timeout_per_iteration_sec=timeout_per_iteration_sec,
    )
    return await asyncio.to_thread(operation)


async def _run_custom(task: str, thread_id: str) -> dict[str, Any]:
    return await run_custom_graph(task, thread_id=thread_id)


async def _run_prebuilt(task: str, thread_id: str) -> dict[str, Any]:
    return await run_prebuilt_graph(task, thread_id=thread_id)


RUNNERS: dict[str, Callable[[str, str], Awaitable[dict[str, Any]]]] = {
    "react_6_2": _run_react,
    "custom": _run_custom,
    "prebuilt": _run_prebuilt,
}


def dry_run_matrix(repeats: int = REPEATS) -> list[dict[str, Any]]:
    if repeats != REPEATS:
        raise ValueError("Домашнее задание требует ровно 3 повтора")
    return [
        {
            "run_id": f"bench-{scenario.task_id}-{implementation}-{repeat}",
            "task_id": scenario.task_id,
            "implementation": implementation,
            "repeat": repeat,
            "task": scenario.task,
        }
        for scenario in AGENT_SCENARIOS
        for implementation in IMPLEMENTATIONS
        for repeat in range(1, repeats + 1)
    ]


def select_matrix(
    *,
    task_id: int | None = None,
    implementation: str | None = None,
    repeat: int | None = None,
) -> list[dict[str, Any]]:
    return [
        item
        for item in dry_run_matrix()
        if (task_id is None or item["task_id"] == task_id)
        and (implementation is None or item["implementation"] == implementation)
        and (repeat is None or item["repeat"] == repeat)
    ]


def _raw_run_paths() -> list[Path]:
    if not RAW_DIR.exists():
        return []
    return sorted(
        path
        for path in RAW_DIR.iterdir()
        if path.is_file() and RAW_RUN_PATTERN.fullmatch(path.name)
    )


def _is_technical_success(row: dict[str, Any]) -> bool:
    return (
        not row.get("timed_out")
        and not row.get("error")
        and bool(str(row.get("answer") or "").strip())
    )


def _technical_error_type(row: dict[str, Any]) -> str | None:
    error = str(row.get("error") or "")
    if row.get("timed_out"):
        return "timeout"
    if "APIConnectionError" in error or "Connection error" in error:
        return "endpoint_connection"
    if "BadRequestError" in error or "Error code: 400" in error:
        return "endpoint_protocol_400"
    if error:
        return "technical_error"
    if not str(row.get("answer") or "").strip():
        return "empty_answer"
    return None


def _read_attempts() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    attempts: dict[str, list[dict[str, Any]]] = {}
    problems: list[dict[str, str]] = []
    for path in _raw_run_paths():
        try:
            if path.stat().st_size == 0:
                raise ValueError("пустой файл")
            payload = json.loads(path.read_text(encoding="utf-8"))
            match = RAW_RUN_PATTERN.fullmatch(path.name)
            if match is None or not isinstance(payload, dict):
                raise ValueError("неверное имя attempt-файла")
            run_id = match.group(1)
            attempt = int(match.group(5) or 1)
            if payload.get("run_id") != run_id:
                raise ValueError("run_id не совпадает с именем файла")
            if any(field not in payload for field in SUMMARY_FIELDS):
                raise ValueError("неполный benchmark-контракт")
            row = {field: payload[field] for field in SUMMARY_FIELDS}
            row["_attempt"] = attempt
            row["_path"] = path.as_posix()
            attempts.setdefault(run_id, []).append(row)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            problems.append({"file": path.as_posix(), "error": str(exc)})
    for rows in attempts.values():
        rows.sort(key=lambda item: item["_attempt"])
    return attempts, problems


def _select_canonical(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in attempts if _is_technical_success(row)]
    return (successful or attempts)[-1]


def _read_raw_results() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    attempts_by_run, problems = _read_attempts()
    rows: list[dict[str, Any]] = []
    for attempts in attempts_by_run.values():
        selected = _select_canonical(attempts)
        row = {field: selected[field] for field in SUMMARY_FIELDS}
        row["selected_attempt"] = selected["_attempt"]
        row["attempts_count"] = len(attempts)
        row["previous_errors"] = [
            {
                "attempt": attempt["_attempt"],
                "type": _technical_error_type(attempt),
                "error": str(attempt.get("error") or "empty answer"),
            }
            for attempt in attempts
            if attempt is not selected and not _is_technical_success(attempt)
        ]
        rows.append(row)
    rows.sort(key=lambda item: item["run_id"])
    return rows, problems


def _attempt_path(run_id: str, attempt: int) -> Path:
    return RAW_DIR / f"{run_id}.attempt-{attempt}.json"


def _archive_first_attempt(run_id: str) -> Path:
    source = RAW_DIR / f"{run_id}.json"
    target = _attempt_path(run_id, 1)
    if source.exists():
        if target.exists():
            raise FileExistsError(f"Обе версии attempt 1 существуют для {run_id}")
        source.rename(target)
    return target


def _preserve_corrupt_raw(path: Path) -> Path:
    target = path.with_suffix(path.suffix + ".corrupt")
    index = 1
    while target.exists():
        target = path.with_suffix(path.suffix + f".corrupt-{index}")
        index += 1
    path.rename(target)
    return target


def _require_local_endpoint() -> None:
    base_url = get_settings().llm.base_url
    host = urlparse(base_url or "").hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            "Полный agent benchmark разрешён только через локальный "
            "OpenAI-compatible endpoint"
        )


def _tool_names(result: dict[str, Any]) -> list[str]:
    names = [
        str(item.get("name") or "")
        for item in result.get("tool_results", [])
        if item.get("name")
    ]
    if names:
        return names
    return [
        str(item.get("tool_name") or "")
        for item in result.get("trace", [])
        if item.get("tool_name")
    ]


def _tool_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    records = result.get("tool_results") or result.get("trace") or []
    return [item for item in records if isinstance(item, dict)]


def _correctness(
    scenario: AgentScenario,
    answer: str,
    result: dict[str, Any],
    error: str | None,
    timed_out: bool,
) -> tuple[bool | None, str]:
    tool_names = _tool_names(result)
    if timed_out:
        return False, "Запуск завершился по timeout"
    if error:
        return False, f"Ошибка запуска: {error}"
    missing = [name for name in scenario.required_tools if name not in tool_names]
    if missing:
        return False, f"Не вызваны обязательные tools: {', '.join(missing)}"
    forbidden = [name for name in scenario.forbidden_tools if name in tool_names]
    if forbidden:
        return False, f"Вызваны запрещённые tools: {', '.join(forbidden)}"
    if not answer.strip():
        return False, "Финальный ответ пуст"
    if scenario.task_id == 2:
        matching = [
            item
            for item in _tool_records(result)
            if (item.get("name") or item.get("tool_name")) == "get_current_time"
            and (item.get("args") or item.get("tool_args") or {}).get("timezone")
            == "Europe/Moscow"
        ]
        if not matching:
            return False, "get_current_time вызван без timezone Europe/Moscow"
    if scenario.task_id == 5:
        lowered = answer.lower()
        correct = "http" in lowered and "https" in lowered
        return correct, (
            "Ответ содержит HTTP и HTTPS, tools не вызваны"
            if correct
            else "Нужна ручная проверка содержания ответа"
        )
    if scenario.manual_answer_review:
        return None, "Tool-маршрут корректен; смысл ответа требует ручной оценки"
    return True, "Детерминированные критерии выполнены"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


async def _run_one(
    scenario: AgentScenario,
    implementation: str,
    repeat: int,
    run_timeout_sec: float,
    *,
    react_timeout_per_iteration_sec: float = REACT_TIMEOUT_PER_ITERATION_SEC,
    run_id: str | None = None,
    raw_path: Path | None = None,
) -> dict[str, Any]:
    run_id = run_id or f"bench-{scenario.task_id}-{implementation}-{repeat}"
    runner = RUNNERS[implementation]
    started = time.perf_counter()
    result: dict[str, Any] = {}
    error: str | None = None
    timed_out = False
    try:
        operation = (
            runner(
                scenario.task,
                run_id,
                timeout_per_iteration_sec=react_timeout_per_iteration_sec,
            )
            if implementation == "react_6_2"
            else runner(scenario.task, run_id)
        )
        result = await asyncio.wait_for(operation, timeout=run_timeout_sec)
        if result.get("error"):
            error = str(result["error"])
        timed_out = result.get("answer") == "Timeout" or error == "Timeout"
    except TimeoutError:
        timed_out = True
        error = f"Per-run timeout after {run_timeout_sec:g} sec"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    answer = str(result.get("answer") or "")
    usage = result.get("usage") or {}
    tool_names = _tool_names(result)
    correct, reason = _correctness(
        scenario,
        answer,
        result,
        error,
        timed_out,
    )
    raw_path = raw_path or RAW_DIR / f"{run_id}.json"
    row = {
        "run_id": run_id,
        "task_id": scenario.task_id,
        "implementation": implementation,
        "repeat": repeat,
        "answer": answer,
        "correct": correct,
        "correctness_reason": reason,
        "latency_ms": latency_ms,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "total_steps": int(result.get("steps") or 0),
        "tool_calls_count": len(tool_names),
        "tools_used": tool_names,
        "error": error,
        "timed_out": timed_out,
        "raw_file": raw_path.as_posix(),
    }
    raw_payload = {
        **row,
        "task": scenario.task,
        "result": _jsonable(result),
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return row


async def run_benchmark(
    repeats: int = REPEATS,
    run_timeout_sec: float = RUN_TIMEOUT_SEC,
    react_timeout_per_iteration_sec: float = REACT_TIMEOUT_PER_ITERATION_SEC,
    *,
    task_id: int | None = None,
    implementation: str | None = None,
    repeat: int | None = None,
    resume: bool = False,
    rerun_failed: bool = False,
    max_attempts: int = 2,
) -> list[dict[str, Any]]:
    if repeats != REPEATS:
        raise ValueError("Домашнее задание требует ровно 3 повтора")
    matrix = select_matrix(
        task_id=task_id,
        implementation=implementation,
        repeat=repeat,
    )
    scenarios = {scenario.task_id: scenario for scenario in AGENT_SCENARIOS}
    rows: list[dict[str, Any]] = []
    if rerun_failed and not resume:
        raise ValueError("--rerun-failed требует --resume")
    attempts_by_run, problems = _read_attempts()
    existing_rows, _ = _read_raw_results()
    completed = {row["run_id"] for row in existing_rows}
    problem_paths = {item["file"] for item in problems}
    for position, item in enumerate(matrix, start=1):
        run_id = item["run_id"]
        print(f"[{position}/{len(matrix)}] {run_id}")
        attempts = attempts_by_run.get(run_id, [])
        if rerun_failed:
            if not attempts:
                print("  пропущен: нет валидной исходной попытки")
                continue
            if _is_technical_success(_select_canonical(attempts)):
                print("  пропущен: технически успешный результат")
                continue
            if len(attempts) >= max_attempts:
                print(f"  пропущен: лимит {max_attempts} попыток исчерпан")
                continue
            _archive_first_attempt(run_id)
            attempt_number = max(item["_attempt"] for item in attempts) + 1
            raw_path = _attempt_path(run_id, attempt_number)
        else:
            if resume and run_id in completed:
                print("  пропущен: валидный raw уже существует")
                continue
            raw_path = RAW_DIR / f"{run_id}.json"
        if raw_path.as_posix() in problem_paths:
            preserved = _preserve_corrupt_raw(raw_path)
            print(f"  повреждённый raw сохранён как {preserved.as_posix()}")
        elif raw_path.exists():
            raise FileExistsError(
                f"{raw_path} уже существует; используйте --resume"
            )
        row = await _run_one(
                scenarios[item["task_id"]],
                item["implementation"],
                item["repeat"],
                run_timeout_sec,
                react_timeout_per_iteration_sec=react_timeout_per_iteration_sec,
                raw_path=raw_path,
            )
        rows.append(row)
        aggregate_results(update_report=False)
        status = "success" if _is_technical_success(row) else row["error"]
        print(
            f"  результат сохранён; latency_ms={row['latency_ms']}; "
            f"status={status}; raw_total={len(_raw_run_paths())}"
        )
        del row
        gc.collect()
    aggregate_results(update_report=True)
    return rows


def _benchmark_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Задача | Реализация | Avg latency ms | Min | Max | Avg prompt | "
        "Avg completion | Avg total tokens | Avg steps | Avg tool calls | "
        "Успешно | Корректно | Timeout | Другие ошибки |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["task_id"], row["implementation"]), []).append(row)
    for (task_id, implementation), group in sorted(groups.items()):
        successful_rows = [item for item in group if _is_technical_success(item)]
        successful = len(successful_rows)
        correct = sum(item["correct"] is True for item in group)
        timed_out = sum(bool(item["timed_out"]) for item in group)
        other_errors = sum(bool(item["error"]) and not item["timed_out"] for item in group)
        if successful_rows:
            averages = [
                round(statistics.fmean(item[key] for item in successful_rows), 2)
                for key in (
                    "latency_ms",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "total_steps",
                    "tool_calls_count",
                )
            ]
            avg_latency, *other_averages = averages
            metrics = (
                avg_latency,
                round(min(item["latency_ms"] for item in successful_rows), 2),
                round(max(item["latency_ms"] for item in successful_rows), 2),
                *other_averages,
            )
        else:
            metrics = ("—",) * 8
        values = " | ".join(
            str(value)
            for value in (*metrics, successful, correct, timed_out, other_errors)
        )
        lines.append(
            f"| {task_id} | {implementation} | {values} |"
        )
    return "\n".join(lines)


def _overall_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Реализация | Avg latency ms | Avg prompt | Avg completion | "
        "Avg total tokens | Avg steps | Avg tool calls | Success rate | "
        "Correctness rate | Timeout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for implementation in IMPLEMENTATIONS:
        group = [row for row in rows if row["implementation"] == implementation]
        successful = [row for row in group if _is_technical_success(row)]
        averages: list[float | str] = (
            [
                round(statistics.fmean(row[key] for row in successful), 2)
                for key in (
                    "latency_ms",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "total_steps",
                    "tool_calls_count",
                )
            ]
            if successful
            else ["—"] * 6
        )
        success_rate = round(100 * len(successful) / len(group), 2) if group else 0.0
        correctness_rate = (
            round(100 * sum(row["correct"] is True for row in group) / len(group), 2)
            if group
            else 0.0
        )
        timeout_count = sum(bool(row["timed_out"]) for row in group)
        values = " | ".join(str(value) for value in averages)
        lines.append(
            f"| {implementation} | {values} | {success_rate}% | "
            f"{correctness_rate}% | {timeout_count} |"
        )
    return "\n".join(lines)


def _detailed_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Задача | Реализация | Repeat | Latency ms | Prompt tokens | "
        "Completion tokens | Total tokens | Steps | Tool calls | Корректно | Ошибка |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: item["run_id"]):
        error = str(row["error"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['task_id']} | {row['implementation']} | {row['repeat']} | "
            f"{row['latency_ms']} | {row['prompt_tokens']} | "
            f"{row['completion_tokens']} | {row['total_tokens']} | "
            f"{row['total_steps']} | {row['tool_calls_count']} | "
            f"{row['correct']} | {error or '—'} |"
        )
    return "\n".join(lines)


def aggregate_results(*, update_report: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows, problems = _read_raw_results()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "benchmark-results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not update_report:
        return rows, problems
    report = REPORT_PATH.read_text(encoding="utf-8")
    before, remainder = report.split(START_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    attempts_by_run, _ = _read_attempts()
    initially_failed = sum(
        not _is_technical_success(attempts[0])
        for attempts in attempts_by_run.values()
    )
    recovered = sum(
        not _is_technical_success(attempts[0])
        and _is_technical_success(_select_canonical(attempts))
        for attempts in attempts_by_run.values()
    )
    remaining_failed = sum(not _is_technical_success(row) for row in rows)
    status = (
        f"Сохранено основных raw: {len(rows)}/45. "
        f"Первично неуспешных: {initially_failed}; восстановлено повторными "
        f"попытками: {recovered}; осталось неуспешных: {remaining_failed}. "
        + (
            "Итоговая агрегация завершена."
            if len(rows) == len(dry_run_matrix())
            else "Benchmark выполнен частично."
        )
    )
    tables = (
        f"{status}\n\n"
        f"### Основные запуски\n\n{_detailed_table(rows)}\n\n"
        f"### Агрегация\n\n{_benchmark_table(rows)}\n\n"
        f"### Общие средние по реализациям\n\n{_overall_table(rows)}"
    )
    REPORT_PATH.write_text(
        f"{before}{START_MARKER}\n{tables}\n{END_MARKER}{after}",
        encoding="utf-8",
    )
    return rows, problems


def plan_rerun_failed(
    *,
    task_id: int | None = None,
    implementation: str | None = None,
    repeat: int | None = None,
    max_attempts: int = 2,
) -> tuple[list[dict[str, Any]], list[str]]:
    attempts_by_run, _ = _read_attempts()
    selected_ids = {
        item["run_id"]
        for item in select_matrix(
            task_id=task_id,
            implementation=implementation,
            repeat=repeat,
        )
    }
    failed: list[dict[str, Any]] = []
    for run_id, attempts in attempts_by_run.items():
        selected = _select_canonical(attempts)
        if (
            run_id in selected_ids
            and not _is_technical_success(selected)
            and len(attempts) < max_attempts
        ):
            failed.append(
                {
                    "run_id": run_id,
                    "task_id": selected["task_id"],
                    "implementation": selected["implementation"],
                    "repeat": selected["repeat"],
                    "attempts_count": len(attempts),
                    "error_type": _technical_error_type(selected),
                    "error": str(selected.get("error") or "empty answer"),
                }
            )
    failed.sort(key=lambda item: item["run_id"])
    groups = sorted({(item["task_id"], item["implementation"]) for item in failed})
    commands = []
    for group_task, group_implementation in groups:
        command = (
            "python scripts/bench_agents.py --resume --rerun-failed "
            f"--max-attempts {max_attempts} --task-id {group_task} "
            f"--implementation {group_implementation} --repeats 3 "
            "--react-timeout-per-iteration 15 --run-timeout 60"
        )
        if repeat is not None:
            command += f" --repeat {repeat}"
        commands.append(command)
    return failed, commands


def _print_dry_run(
    task_id: int | None,
    implementation: str | None,
    repeat: int | None,
) -> None:
    matrix = select_matrix(
        task_id=task_id,
        implementation=implementation,
        repeat=repeat,
    )
    for item in matrix:
        print(
            f"{item['run_id']} | task={item['task_id']} | "
            f"implementation={item['implementation']} | repeat={item['repeat']}"
        )
    print(f"TOTAL={len(matrix)}")


def _react_timeout(value: str) -> float:
    timeout = float(value)
    if not 5 <= timeout <= 15:
        raise argparse.ArgumentTypeError("значение должно быть в диапазоне 5..15")
    return timeout


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark агентов блоков 6.2/6.3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--plan-rerun-failed", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--task-id", type=int, choices=range(1, 6))
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS)
    parser.add_argument("--repeat", type=int, choices=range(1, REPEATS + 1))
    parser.add_argument("--repeats", type=int, choices=(REPEATS,), default=REPEATS)
    parser.add_argument(
        "--react-timeout-per-iteration",
        type=_react_timeout,
        default=REACT_TIMEOUT_PER_ITERATION_SEC,
    )
    parser.add_argument("--run-timeout", type=float, default=RUN_TIMEOUT_SEC)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser


async def _async_main(
    run_timeout_sec: float,
    react_timeout_per_iteration_sec: float,
    *,
    task_id: int | None,
    implementation: str | None,
    repeat: int | None,
    resume: bool,
    rerun_failed: bool,
    max_attempts: int,
) -> None:
    _require_local_endpoint()
    await run_benchmark(
        run_timeout_sec=run_timeout_sec,
        react_timeout_per_iteration_sec=react_timeout_per_iteration_sec,
        task_id=task_id,
        implementation=implementation,
        repeat=repeat,
        resume=resume,
        rerun_failed=rerun_failed,
        max_attempts=max_attempts,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.rerun_failed and not args.resume:
        parser.error("--rerun-failed можно использовать только вместе с --resume")
    if args.max_attempts < 1:
        parser.error("--max-attempts должен быть положительным")
    if args.dry_run:
        _print_dry_run(args.task_id, args.implementation, args.repeat)
        return
    if args.aggregate_only:
        rows, problems = aggregate_results()
        completed = {row["run_id"] for row in rows}
        missing = [
            item["run_id"] for item in dry_run_matrix() if item["run_id"] not in completed
        ]
        print(f"Готово: {len(rows)}/45")
        for problem in problems:
            print(f"Повреждён: {problem['file']}: {problem['error']}")
        print("Недостающие run_id: " + (", ".join(missing) or "нет"))
        return
    if args.plan_rerun_failed:
        failed, commands = plan_rerun_failed(
            task_id=args.task_id,
            implementation=args.implementation,
            repeat=args.repeat,
            max_attempts=args.max_attempts,
        )
        for item in failed:
            print(
                f"{item['run_id']} | {item['error_type']} | "
                f"attempts={item['attempts_count']}/{args.max_attempts}"
            )
        print(f"Технически неуспешных для rerun: {len(failed)}")
        for command in commands:
            print(command)
        return
    asyncio.run(
        _async_main(
            args.run_timeout,
            args.react_timeout_per_iteration,
            task_id=args.task_id,
            implementation=args.implementation,
            repeat=args.repeat,
            resume=args.resume,
            rerun_failed=args.rerun_failed,
            max_attempts=args.max_attempts,
        )
    )


if __name__ == "__main__":
    main()
