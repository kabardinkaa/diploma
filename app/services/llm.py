import asyncio
import hashlib
import json
import time

import structlog

from app.observability.pii import prompt_hash, redact_pii

from collections.abc import AsyncIterator
from typing import Any

import openai
from redis.exceptions import RedisError

from app.core.config import Settings
from app.core.exceptions import LLMAuthError, LLMError, LLMRateLimitError, LLMTimeoutError
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Usage
from app.security.llm_guard import BLOCKED_RESPONSE, filter_output, is_prompt_attack

logger = structlog.get_logger("llm-service")

class LLMService:
    """
    Сервисный слой для LLM.

    Отвечает за:
    - вызов AsyncOpenAI;
    - кеширование обычных /chat ответов;
    - streaming без кеша;
    - адаптацию raw-ответа SDK в Pydantic-схемы;
    - преобразование ошибок SDK в доменные ошибки.
    """

    def __init__(self, openai_client, cache, settings: Settings) -> None:
        self.openai = openai_client
        self.cache = cache
        self.settings = settings

    def _cache_key(self, req: ChatRequest) -> str:
        payload = req.model_dump(
            mode="json",
            exclude={"user_id", "session_id"},
            exclude_none=True,
        )
        normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"chat:{digest}"

    async def _cache_get(self, key: str) -> str | None:
        try:
            if self.cache is None:
                return None

            if isinstance(self.cache, dict):
                return self.cache.get(key)

            value = await self.cache.get(key)

            if isinstance(value, bytes):
                return value.decode("utf-8")

            return value

        except RedisError:
            return None

    async def _cache_set(self, key: str, value: str) -> None:
        try:
            if self.cache is None:
                return

            if isinstance(self.cache, dict):
                self.cache[key] = value
                return

            await self.cache.setex(
                key,
                self.settings.cache_ttl_seconds,
                value,
            )

        except RedisError:
            return

    async def complete(self, req: ChatRequest) -> ChatResponse:
        started_at = time.perf_counter()
        raw_prompt = json.dumps(
            [message.model_dump() for message in req.messages],
            ensure_ascii=False,
        )

        guard_result = is_prompt_attack(raw_prompt)
        if not guard_result.allowed:
            logger.warning(
                "llm_request_blocked",
                reason=guard_result.reason,
                model=req.model or self.settings.llm.default_model,
            )
            return ChatResponse(
                content=BLOCKED_RESPONSE,
                model=req.model or self.settings.llm.default_model,
                finish_reason="security_blocked",
                cached=False,
            )

        key = self._cache_key(req)

        cached_value = await self._cache_get(key)
        if cached_value is not None:
            cached_response = ChatResponse.model_validate_json(cached_value)
            cached_response.cached = True

            logger.info(
                "llm_request_completed",
                model=req.model or self.settings.llm.default_model,
                input_tokens=None,
                output_tokens=None,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
                finish_reason="cache_hit",
                prompt_hash=prompt_hash(raw_prompt),
                prompt_preview=redact_pii(raw_prompt)[:120],
                cached=True,
            )

            return cached_response

        try:
            response = await self.openai.chat.completions.create(
                model=req.model or self.settings.llm.default_model,
                messages=[message.model_dump() for message in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )

            chat_response = ChatResponse.from_openai(response, cached=False)
            await self._cache_set(key, chat_response.model_dump_json())

            usage = getattr(response, "usage", None)
            choice = response.choices[0] if response.choices else None

            logger.info(
                "llm_request_completed",
                model=getattr(response, "model", req.model or self.settings.llm.default_model),
                input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
                finish_reason=getattr(choice, "finish_reason", None) if choice else None,
                prompt_hash=prompt_hash(raw_prompt),
                prompt_preview=redact_pii(raw_prompt)[:120],
                cached=False,
            )

            return chat_response

        except openai.RateLimitError as exc:
            raise LLMRateLimitError("Превышен лимит запросов к LLM-провайдеру") from exc

        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("LLM-провайдер не ответил за отведённое время") from exc

        except openai.AuthenticationError as exc:
            raise LLMAuthError("Ошибка авторизации у LLM-провайдера") from exc

        except Exception as exc:
            raise LLMError("Ошибка при обращении к LLM-провайдеру") from exc

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        try:
            stream = await self.openai.chat.completions.create(
                model=req.model or self.settings.llm.default_model,
                messages=[message.model_dump() for message in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if chunk.usage is not None:
                    yield ChatDelta(
                        usage=Usage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        )
                    )
                    continue

                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content
                if content:
                    yield ChatDelta(content=content)

        except openai.RateLimitError as exc:
            raise LLMRateLimitError("Превышен лимит запросов к LLM-провайдеру") from exc

        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("LLM-провайдер не ответил за отведённое время") from exc

        except openai.AuthenticationError as exc:
            raise LLMAuthError("Ошибка авторизации у LLM-провайдера") from exc

        except Exception as exc:
            raise LLMError("Ошибка при streaming-обращении к LLM-провайдеру") from exc

    async def batch(self, requests: list[ChatRequest]) -> list[ChatResponse | dict[str, Any]]:
        async def safe_complete(req: ChatRequest) -> ChatResponse | dict[str, Any]:
            try:
                return await self.complete(req)
            except LLMError as exc:
                return {
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                    }
                }

        return await asyncio.gather(
            *(safe_complete(req) for req in requests),
            return_exceptions=False,
        )

        return results