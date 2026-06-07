from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Проверка состояния API",
    responses={
        200: {"description": "API работает"},
    },
)
async def health() -> dict[str, str]:
    return {"status": "ok"}