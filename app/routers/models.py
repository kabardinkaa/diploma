from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.models import ModelInfo

router = APIRouter(tags=["Models"])


@router.get(
    "/models",
    response_model=list[ModelInfo],
    summary="Список доступных моделей",
    description=(
        "Returns the server-configured chat, RAG, and agent model identifiers. "
        "Clients cannot select an arbitrary production model through this endpoint."
    ),
    responses={
        200: {"description": "Список моделей успешно получен"},
    },
)
async def get_models(settings: Settings = Depends(get_settings)) -> list[ModelInfo]:
    base_url = settings.llm.base_url or ""
    provider = "openrouter" if "openrouter" in base_url.lower() else "openai"
    model_ids = dict.fromkeys(
        (
            settings.llm.default_model,
            settings.rag_generation_model,
            settings.agent_model,
        )
    )
    return [ModelInfo(id=model_id, provider=provider) for model_id in model_ids]
