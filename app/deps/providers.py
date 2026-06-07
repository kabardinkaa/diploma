from typing import Annotated

from fastapi import Depends, Request
from typing import Any

from app.core.config import Settings, get_settings
from app.services.llm import LLMService


def get_openai(request: Request):
    return request.app.state.openai


def get_cache(request: Request) -> Any:
    return request.app.state.cache


def get_llm_service(
    openai_client=Depends(get_openai),
    cache: Any = Depends(get_cache),
    settings: Settings = Depends(get_settings),
) -> LLMService:
    return LLMService(
        openai_client=openai_client,
        cache=cache,
        settings=settings,
    )


SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
CacheDep = Annotated[Any, Depends(get_cache)]