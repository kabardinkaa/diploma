import asyncio
import json
from typing import Any
import time
import uuid
import os

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.observability.logging import setup_logging
from app.observability.tracing import (
    force_flush_tracing,
    setup_tracing,
    shutdown_tracing,
)

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.cache import BoundedTTLCache
from app.core.exceptions import (
    InfrastructureError,
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    SafeInputError,
)
from app.admin.routes import router as admin_router
from app.routers import agent, chat, documents, health, models, rag
from app.chat.routes import router as chat_history_router
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repositories.postgres_repo import (
    PostgresChatRepository,
    create_postgres_pool,
)
from app.services.rag import RAGService
from app.services.agent_persistent import agent_lifespan
from app.services.readiness import ReadinessService
from app.security.rate_limit import PublicRateLimitMiddleware

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
    setup_tracing(enabled=settings.rag_tracing_enabled)

    client_kwargs = {
        "api_key": settings.llm.api_key.get_secret_value(),
        "timeout": settings.llm.request_timeout,
        "max_retries": settings.llm.max_retries,
    }

    if settings.llm.base_url:
        client_kwargs["base_url"] = settings.llm.base_url

    openai_client = AsyncOpenAI(**client_kwargs)
    postgres_pool = None
    rag_service = None
    try:
        app.state.openai = openai_client
        app.state.cache = BoundedTTLCache(
            max_entries=settings.llm_cache_max_entries,
            ttl_seconds=settings.llm_cache_ttl_seconds,
            enabled=settings.llm_cache_enabled,
        )

        if settings.chat_repository == "postgres":
            postgres_pool = await create_postgres_pool(settings)
            chat_repository = PostgresChatRepository(postgres_pool)
            await chat_repository.initialize()
        else:
            chat_repository = JsonChatRepository(
                base_dir=settings.chat_storage_dir
            )
        app.state.db_pool = postgres_pool
        app.state.chat_repository = chat_repository

        rag_service = RAGService(settings, openai_client=openai_client)
        app.state.rag_service = rag_service
        app.state.readiness_probe = ReadinessService(
            settings,
            rag_service.async_client,
            postgres_pool=postgres_pool,
        )

        async with agent_lifespan(
            settings,
            rag_service=rag_service,
        ) as persistent_agent:
            app.state.persistent_agent = persistent_agent
            await app.state.rag_service.build()
            logger.info("Application startup complete")
            yield
    finally:
        if rag_service is not None:
            try:
                async with asyncio.timeout(10):
                    await rag_service.close()
            except Exception as exc:
                logger.warning("rag.shutdown_failed", error_type=type(exc).__name__)

        try:
            async with asyncio.timeout(10):
                await openai_client.close()
        except Exception as exc:
            logger.warning("openai.shutdown_failed", error_type=type(exc).__name__)

        if postgres_pool is not None:
            try:
                async with asyncio.timeout(10):
                    await postgres_pool.close()
            except Exception as exc:
                logger.warning("postgres.shutdown_failed", error_type=type(exc).__name__)
                postgres_pool.terminate()

        try:
            await asyncio.wait_for(
                asyncio.to_thread(force_flush_tracing, 5_000),
                timeout=6.0,
            )
        except Exception as exc:
            logger.warning("tracing.flush_failed", error_type=type(exc).__name__)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(shutdown_tracing),
                timeout=6.0,
            )
        except Exception as exc:
            logger.warning("tracing.shutdown_failed", error_type=type(exc).__name__)

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
    PublicRateLimitMiddleware,
    requests=settings.public_rate_limit_requests,
    window_seconds=settings.public_rate_limit_window_seconds,
    max_concurrent=settings.public_max_concurrent_requests,
    enabled=settings.public_rate_limit_enabled,
    trusted_proxy_cidrs=settings.trusted_proxy_cidrs,
    paths={
        ("POST", "/chat"),
        ("POST", "/chat/stream"),
        ("POST", "/rag/query"),
        ("POST", "/agent/stream"),
    },
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


@app.exception_handler(InfrastructureError)
async def infrastructure_error_handler(
    _: Request,
    exc: InfrastructureError,
) -> JSONResponse:
    return SafeJSONResponse(
        status_code=503,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


@app.exception_handler(SafeInputError)
async def safe_input_error_handler(
    _: Request,
    exc: SafeInputError,
) -> JSONResponse:
    return SafeJSONResponse(
        status_code=exc.status_code,
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
app.include_router(rag.router)
app.include_router(documents.router)
app.include_router(admin_router)
app.include_router(chat_history_router)
app.include_router(agent.router)
