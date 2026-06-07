import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from openai import OpenAI

# Чтобы скрипт запускался из корня проекта и видел app.llm.async_client
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.llm.async_client import AsyncLLMClient  # noqa: E402


load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
PROMPT_COUNT = int(os.getenv("BENCHMARK_PROMPT_COUNT", "20"))

SYSTEM_PROMPT = (
    "Ты — ИИ-ассистент внутренней техподдержки. "
    "Отвечай кратко, понятно и на русском языке."
)


def build_prompts() -> list[str]:
    return [
        f"Объясни одним коротким абзацем концепцию async в Python. Пример №{i}."
        for i in range(1, PROMPT_COUNT + 1)
    ]


def build_sync_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")

    if not api_key:
        raise ValueError(
            "Не найден API-ключ. Добавь OPENAI_API_KEY или OPENROUTER_API_KEY в .env"
        )

    client_kwargs = {
        "api_key": api_key,
        "timeout": 60,
        "max_retries": 3,
    }

    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


def sync_complete(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
    temperature=0.2,
    max_tokens=120,
)

    return response.choices[0].message.content or ""


def run_sync_benchmark(prompts: list[str]) -> dict[str, Union[str, int, float]]:
    client = build_sync_client()

    started_at = time.perf_counter()

    answers = []
    for prompt in prompts:
        answers.append(sync_complete(client, prompt))

    duration = round(time.perf_counter() - started_at, 2)

    return {
        "mode": "sync sequential",
        "concurrency": 1,
        "requests": len(prompts),
        "success": len([answer for answer in answers if answer]),
        "errors": 0,
        "duration_sec": duration,
    }


async def run_async_benchmark(
    prompts: list[str],
    concurrency: int,
) -> dict[str, Union[str, int, float]]:
    client = AsyncLLMClient(
        concurrency=concurrency,
        sdk_timeout=180,
        logic_timeout=180,
    )

    started_at = time.perf_counter()
    results = await client.batch_chat(prompts)
    duration = round(time.perf_counter() - started_at, 2)

    errors = [result for result in results if isinstance(result, Exception)]
    success = len(results) - len(errors)

    return {
        "mode": "async batch_chat",
        "concurrency": concurrency,
        "requests": len(prompts),
        "success": success,
        "errors": len(errors),
        "duration_sec": duration,
    }


def print_results(results: list[dict[str, Union[str, int, float]]]) -> None:
    print("\nBenchmark results")
    print("-" * 86)
    print(f"{'Mode':<20} {'Concurrency':<12} {'Requests':<10} {'Success':<8} {'Errors':<8} {'Time, sec':<10}")
    print("-" * 86)

    for row in results:
        print(
            f"{row['mode']:<20} "
            f"{row['concurrency']:<12} "
            f"{row['requests']:<10} "
            f"{row['success']:<8} "
            f"{row['errors']:<8} "
            f"{row['duration_sec']:<10}"
        )

    print("-" * 86)


def save_results(results: list[dict[str, Union[str, int, float]]]) -> None:
    output_path = PROJECT_ROOT / "scripts" / "benchmark_results.md"

    sync_time = float(results[0]["duration_sec"])

    lines = [
        "# Benchmark results",
        "",
        f"Модель: `{MODEL}`",
        f"Количество промптов: `{PROMPT_COUNT}`",
        "",
        "| Режим | Concurrency / Semaphore | Кол-во запросов | Успешно | Ошибки | Время, сек | Ускорение относительно sync |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        duration = float(row["duration_sec"])
        speedup = round(sync_time / duration, 2) if duration > 0 else 0

        lines.append(
            f"| {row['mode']} "
            f"| {row['concurrency']} "
            f"| {row['requests']} "
            f"| {row['success']} "
            f"| {row['errors']} "
            f"| {row['duration_sec']} "
            f"| {speedup}x |"
        )

    lines.extend(
        [
            "",
            "## Вывод",
            "",
            "Асинхронная версия быстрее синхронной последовательной обработки, потому что LLM-запросы являются I/O-bound: "
            "основное время программа ждёт сетевой ответ от API. `asyncio.gather` запускает несколько запросов конкурентно, "
            "а `Semaphore` ограничивает количество одновременных запросов и снижает риск rate limit.",
            "",
            "Оптимальный лимит конкурентности выбирается экспериментально. Для стабильной работы в учебном проекте выбран "
            "`concurrency=5`: он даёт ускорение относительно sync-режима и не создаёт слишком агрессивную нагрузку на API.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {output_path}")


async def main() -> None:
    prompts = build_prompts()

    results = []

    print(f"Model: {MODEL}")
    print(f"Prompts: {len(prompts)}")

    print("\nRunning sync benchmark...")
    results.append(run_sync_benchmark(prompts))

    for concurrency in (1, 5, 10):
        print(f"\nRunning async benchmark, concurrency={concurrency}...")
        results.append(await run_async_benchmark(prompts, concurrency))

    print_results(results)
    save_results(results)


if __name__ == "__main__":
    asyncio.run(main())