from typing import Annotated, Any, Protocol

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


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
    summary="Проверка состояния API",
    responses={
        200: {"description": "API работает"},
    },
)
async def health() -> dict[str, str]:
    """Backward-compatible alias for the lightweight liveness probe."""

    return _live_response()


@router.get(
    "/health/live",
    summary="Проверка жизнеспособности процесса API",
    responses={200: {"description": "Процесс API работает"}},
)
async def live() -> dict[str, str]:
    return _live_response()


@router.get(
    "/health/ready",
    summary="Проверка готовности API и обязательных зависимостей",
    responses={
        200: {"description": "API готов обслуживать запросы"},
        503: {"description": "Обязательная зависимость недоступна"},
    },
)
async def ready(probe: ReadinessDep) -> JSONResponse:
    report = await probe.check()
    status_code = 200 if report["status"] == "ready" else 503
    return JSONResponse(status_code=status_code, content=report)
