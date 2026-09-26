from __future__ import annotations

import json
from typing import Any

import openai

from app.core.exceptions import (
    InfrastructureError,
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    PublicGenerationControlError,
)


def public_stream_error(exc: Exception) -> dict[str, str]:
    """Map internal failures to a stable public contract without exception text."""

    if isinstance(exc, PublicGenerationControlError):
        return {
            "type": "error",
            "code": exc.code,
            "message": exc.message,
        }
    if isinstance(exc, (LLMRateLimitError, openai.RateLimitError)):
        return {
            "type": "error",
            "code": "llm_rate_limit",
            "message": "LLM provider is temporarily rate limited",
        }
    if isinstance(exc, (LLMTimeoutError, openai.APITimeoutError)):
        return {
            "type": "error",
            "code": "llm_timeout",
            "message": "LLM provider timed out",
        }
    if isinstance(exc, (LLMAuthError, openai.AuthenticationError)):
        return {
            "type": "error",
            "code": "llm_unavailable",
            "message": "LLM provider is temporarily unavailable",
        }
    if isinstance(exc, (LLMError, openai.OpenAIError)):
        return {
            "type": "error",
            "code": "llm_unavailable",
            "message": "LLM provider is temporarily unavailable",
        }
    if isinstance(exc, InfrastructureError):
        return {
            "type": "error",
            "code": exc.code,
            "message": exc.message,
        }
    return {
        "type": "error",
        "code": "stream_error",
        "message": "Stream terminated unexpectedly",
    }


def sse_json_event(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def sse_error_events(exc: Exception) -> tuple[str, str]:
    return (
        sse_json_event("error", public_stream_error(exc)),
        sse_json_event("done", {"type": "done", "status": "error"}),
    )
