import asyncio

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Liveness probe",
    responses={200: {"description": "Application process is alive"}},
)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness probe",
    responses={
        200: {"description": "Application is ready"},
        503: {"description": "Application is not ready"},
    },
)
async def ready(request: Request) -> JSONResponse:
    redis = getattr(request.app.state, "redis", None)

    if redis is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "redis": "down"},
        )

    try:
        await asyncio.wait_for(redis.ping(), timeout=2.0)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "redis": "down"},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "redis": "up"},
    )