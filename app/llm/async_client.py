import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Union

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """
    Асинхронный LLM-клиент для домашнего задания 3.3.

    Что реализовано:
    - AsyncOpenAI вместо синхронного OpenAI
    - complete(prompt) для одного запроса
    - batch_chat(prompts) через asyncio.gather + return_exceptions=True
    - Semaphore как атрибут экземпляра
    - stream_chat(prompt) как async-генератор
    - таймаут SDK + asyncio.timeout вокруг бизнес-логики
    - логирование длительности каждого запроса
    """

    def __init__(
        self,
        model: str | None = None,
        concurrency: int = 5,
        sdk_timeout: float = 30,
        logic_timeout: float = 15,
        max_retries: int = 3,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.concurrency = concurrency
        self.logic_timeout = logic_timeout

        # Важно для ДЗ: Semaphore — атрибут экземпляра, а не локальная переменная метода.
        self._sem = asyncio.Semaphore(concurrency)

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")

        if not api_key:
            raise ValueError(
                "Не найден API-ключ. Добавь OPENAI_API_KEY или OPENROUTER_API_KEY в .env"
            )

        client_kwargs = {
            "api_key": api_key,
            "timeout": sdk_timeout,
            "max_retries": max_retries,
        }

        # base_url нужен, если используем OpenRouter или другой OpenAI-compatible API.
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**client_kwargs)

    async def complete(self, prompt: str) -> str:
        """
        Выполняет один асинхронный запрос к LLM.
        """
        started_at = time.perf_counter()

        async with self._sem:
            try:
                async with asyncio.timeout(self.logic_timeout):
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Ты — ИИ-ассистент внутренней техподдержки. "
                                    "Отвечай кратко, понятно и на русском языке."
                                ),
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        temperature=0.2,
                        max_tokens=120,
                    )

                    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

                    if not response.choices or not response.choices[0].message:
                        raise RuntimeError("LLM вернула ответ без choices/message")

                    answer = response.choices[0].message.content or ""

                    if not answer.strip():
                        raise RuntimeError("LLM вернула пустой ответ")

                logger.info(
                    "llm.call",
                    extra={
                        "duration_ms": duration_ms,
                        "model": self.model,
                        "prompt_chars": len(prompt),
                        "status": "success",
                    },
                )

                return answer

            except Exception:
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

                logger.exception(
                    "llm.call",
                    extra={
                        "duration_ms": duration_ms,
                        "model": self.model,
                        "prompt_chars": len(prompt),
                        "status": "error",
                    },
                )

                raise

    async def batch_chat(
        self,
        prompts: list[str],
        concurrency: int | None = None,
    ) -> list[Union[str, Exception]]:
        """
        Параллельно обрабатывает список промптов.

        concurrency оставлен в сигнатуре по заданию, но фактический лимит задаётся
        при создании клиента: AsyncLLMClient(concurrency=N).

        Это сделано специально, чтобы Semaphore создавался в __init__ один раз
        и был атрибутом экземпляра, как требует самопроверка.
        """
        if concurrency is not None and concurrency != self.concurrency:
            raise ValueError(
                "В этой реализации concurrency задаётся при создании клиента: "
                f"AsyncLLMClient(concurrency={concurrency}). "
                "Так Semaphore остаётся атрибутом экземпляра."
            )

        coroutines = [self.complete(prompt) for prompt in prompts]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        return list(results)

    async def stream_chat(self, prompt: str) -> AsyncIterator[str]:
        """
        Стримит ответ LLM токенами/дельтами.

        Проверка:
        curl -N -X POST http://localhost:8000/chat/stream ^
          -H "Content-Type: application/json" ^
          -d "{\"prompt\": \"Что такое event loop?\"}"
        """
        started_at = time.perf_counter()
        first_token_at: float | None = None
        total_tokens: int | None = None

        async with self._sem:
            try:
                async with asyncio.timeout(self.logic_timeout):
                    stream = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Ты — ИИ-ассистент внутренней техподдержки. "
                                    "Отвечай кратко, понятно и на русском языке."
                                ),
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        temperature=0.2,
                        stream=True,
                        stream_options={"include_usage": True},
                    )

                    async for chunk in stream:
                        if chunk.usage is not None:
                            total_tokens = chunk.usage.total_tokens

                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta.content

                        if delta:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()

                            yield delta

                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                ttft_ms = (
                    round((first_token_at - started_at) * 1000, 2)
                    if first_token_at is not None
                    else None
                )

                logger.info(
                    "llm.stream",
                    extra={
                        "duration_ms": duration_ms,
                        "ttft_ms": ttft_ms,
                        "model": self.model,
                        "prompt_chars": len(prompt),
                        "total_tokens": total_tokens,
                        "status": "success",
                    },
                )

            except Exception:
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

                logger.exception(
                    "llm.stream",
                    extra={
                        "duration_ms": duration_ms,
                        "model": self.model,
                        "prompt_chars": len(prompt),
                        "status": "error",
                    },
                )

                raise