from typing import Annotated, Any, Protocol

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.schemas.openapi import HealthResponse, ReadinessResponse

router = APIRouter(tags=["Health"])


class ReadinessProbe(Protocol):
    async def check(self) -> dict[str, Any]:
        ...


def get_readiness_probe(request: Request) -> ReadinessProbe:
    return request.app.state.readiness_probe


ReadinessDep = Annotated[ReadinessProbe, Depends(get_readiness_probe)]


def _live_response() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния API",
    description=(
        "Backward-compatible lightweight liveness endpoint. It confirms that "
        "the HTTP process responds and does not probe PostgreSQL, Qdrant, the "
        "LLM provider, or Phoenix."
    ),
    responses={
        200: {"description": "API работает"},
    },
)
async def health() -> dict[str, str]:
    """Backward-compatible alias for the lightweight liveness probe."""

    return _live_response()


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Проверка жизнеспособности процесса API",
    description=(
        "Fast liveness probe. It remains independent of PostgreSQL, Qdrant, "
        "the LLM provider, and Phoenix."
    ),
    responses={200: {"description": "Процесс API работает"}},
)
async def live() -> dict[str, str]:
    return _live_response()


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Проверка готовности API и обязательных зависимостей",
    description=(
        "Checks PostgreSQL, Qdrant, and the presence of the `corporate_rag` "
        "collection. It never performs a paid LLM request and never exposes "
        "connection strings or raw dependency errors."
    ),
    responses={
        200: {"description": "API готов обслуживать запросы"},
        503: {
            "model": ReadinessResponse,
            "description": "Обязательная зависимость недоступна",
        },
    },
)
async def ready(probe: ReadinessDep) -> JSONResponse:
    report = await probe.check()
    status_code = 200 if report["status"] == "ready" else 503
    return JSONResponse(status_code=status_code, content=report)
