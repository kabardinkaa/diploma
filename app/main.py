import json
from typing import Any
import time
import uuid
import os

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.observability.logging import setup_logging
from app.observability.tracing import setup_tracing

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.exceptions import LLMAuthError, LLMError, LLMRateLimitError, LLMTimeoutError
from app.routers import chat, health, models
from app.chat.routes import router as chat_history_router

setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = structlog.get_logger("llm-service")

class SafeJSONResponse(JSONResponse):
    """
    JSONResponse с ASCII-safe сериализацией.

    Это помогает Windows PowerShell корректно отображать кириллицу
    в ответах API при локальной проверке ДЗ.
    """
    media_type = "application/json; charset=utf-8"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_tracing()

    client_kwargs = {
        "api_key": settings.llm.api_key.get_secret_value(),
        "timeout": settings.llm.request_timeout,
        "max_retries": settings.llm.max_retries,
    }

    if settings.llm.base_url:
        client_kwargs["base_url"] = settings.llm.base_url

    app.state.openai = AsyncOpenAI(**client_kwargs)
    app.state.redis = Redis.from_url(
    settings.redis_url,
    decode_responses=True,
)

    # In-memory cache для локального MVP.
    # Он нужен, чтобы ДЗ работало даже без запущенного Redis.
    app.state.cache = {}

    logger.info("Application startup complete")

    try:
        yield
    finally:
        await app.state.openai.close()

        if getattr(app.state, "redis", None) is not None:
            await app.state.redis.aclose()

        logger.info("Application shutdown complete")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="FastAPI-сервис для LLM в дипломном проекте",
    version=settings.app_version,
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    clear_contextvars()

    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    started_at = time.perf_counter()

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception:
        status_code = 500
        logger.exception(
    "request.failed",
    status=status_code,
    )
        raise

    finally:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

        logger.info(
        "request.completed",
        status=status_code,
        duration_ms=duration_ms,
        )

        if "response" in locals():
            response.headers["X-Request-ID"] = request_id
        clear_contextvars()


@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
    status_code = 502

    if isinstance(exc, LLMRateLimitError):
        status_code = 429
    elif isinstance(exc, LLMTimeoutError):
        status_code = 504
    elif isinstance(exc, LLMAuthError):
        status_code = 502

    return SafeJSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return SafeJSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Ошибка валидации запроса",
                "details": [
                    {
                        "field": ".".join(str(part) for part in error["loc"]),
                        "message": error["msg"],
                    }
                    for error in exc.errors()
                ],
            }
        },
    )


app.include_router(health.router)
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(chat_history_router)