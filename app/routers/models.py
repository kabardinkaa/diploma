from fastapi import APIRouter

from app.schemas.models import AVAILABLE_MODELS, ModelInfo

router = APIRouter(tags=["models"])


@router.get(
    "/models",
    response_model=list[ModelInfo],
    summary="Список доступных моделей",
    responses={
        200: {"description": "Список моделей успешно получен"},
    },
)
async def get_models() -> list[ModelInfo]:
    return AVAILABLE_MODELS