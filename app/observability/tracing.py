from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import structlog

logger = structlog.get_logger("tracing")
_tracer_provider: Any | None = None
_tracing_initialized = False


def setup_tracing(
    project_name: str = "diploma-fastapi",
    *,
    enabled: bool | None = None,
) -> Any | None:
    """Configure OpenAI and LlamaIndex on one Phoenix tracer provider."""
    global _tracer_provider, _tracing_initialized

    if enabled is None:
        enabled = os.environ.get("RAG_TRACING_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
        }
    if not enabled:
        return None
    if _tracing_initialized:
        return _tracer_provider

    try:
        from openinference.instrumentation.llama_index import (
            LlamaIndexInstrumentor,
        )
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register
    except ImportError:
        logger.warning("tracing.optional_dependency_missing")
        return None

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
    tracer_provider = register(
        project_name=project_name,
        endpoint=endpoint,
    )
    _tracer_provider = tracer_provider
    _tracing_initialized = True
    try:
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception:
        logger.exception("tracing.instrumentation_failed")
    return tracer_provider


@contextmanager
def trace_span(name: str) -> Iterator[Any | None]:
    if _tracer_provider is None:
        yield None
        return
    tracer = _tracer_provider.get_tracer("diploma.rag")
    with tracer.start_as_current_span(name) as span:
        yield span


def set_span_attributes(span: Any | None, **attributes: Any) -> None:
    if span is None:
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def force_flush_tracing(timeout_millis: int = 10_000) -> bool:
    if _tracer_provider is None:
        return False
    return bool(_tracer_provider.force_flush(timeout_millis=timeout_millis))


def reset_tracing_for_tests() -> None:
    global _tracer_provider, _tracing_initialized
    _tracer_provider = None
    _tracing_initialized = False
