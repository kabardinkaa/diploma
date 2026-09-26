from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Safe public error message")
    details: list[dict[str, Any]] | None = Field(
        default=None,
        description="Validation details when available",
    )


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "validation_error",
                        "message": "Ошибка валидации запроса",
                    }
                }
            ]
        }
    )

    error: ErrorDetail


class HTTPDetailResponse(BaseModel):
    detail: str | dict[str, Any] = Field(
        description="FastAPI HTTP error detail",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"status": "ok"}})

    status: Literal["ok"]


class DependencyStatus(BaseModel):
    status: Literal["ok", "unavailable", "missing"]


class ReadinessDependencies(BaseModel):
    postgres: DependencyStatus
    qdrant: DependencyStatus
    corporate_rag: DependencyStatus


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ready",
                    "dependencies": {
                        "postgres": {"status": "ok"},
                        "qdrant": {"status": "ok"},
                        "corporate_rag": {"status": "ok"},
                    },
                },
                {
                    "status": "not_ready",
                    "dependencies": {
                        "postgres": {"status": "ok"},
                        "qdrant": {"status": "unavailable"},
                        "corporate_rag": {"status": "unavailable"},
                    },
                },
            ]
        }
    )

    status: Literal["ready", "not_ready"]
    dependencies: ReadinessDependencies


def error_response(
    description: str,
    *,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {
            "application/json": {
                "example": {"error": {"code": code, "message": message}}
            }
        },
    }


def http_error_response(
    description: str,
    *,
    detail: str,
) -> dict[str, Any]:
    return {
        "model": HTTPDetailResponse,
        "description": description,
        "content": {"application/json": {"example": {"detail": detail}}},
    }


def sse_response(description: str, example: str) -> dict[str, Any]:
    return {
        "description": description,
        "content": {
            "text/event-stream": {
                "schema": {"type": "string"},
                "examples": {
                    "success": {
                        "summary": "Successful stream",
                        "value": example,
                    },
                    "safeError": {
                        "summary": "Safe terminal error after streaming starts",
                        "value": (
                            "event: error\n"
                            "data: {\"type\":\"error\",\"code\":\"llm_unavailable\","
                            "\"message\":\"LLM provider temporarily unavailable\"}\n\n"
                            "event: done\n"
                            "data: {\"type\":\"done\",\"status\":\"error\"}\n\n"
                        ),
                    },
                },
            }
        },
    }
