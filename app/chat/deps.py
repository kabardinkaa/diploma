from typing import Annotated

from fastapi import Depends, Request

from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repository import ChatRepository
from app.chat.service import ChatService
from app.core.config import Settings, get_settings
from app.deps.providers import LLMServiceDep
from app.moderation import ModerationService


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_repository(request: Request, settings: SettingsDep) -> ChatRepository:
    repository = getattr(request.app.state, "chat_repository", None)
    if repository is not None:
        return repository

    # This fallback keeps isolated route tests usable without running lifespan.
    if settings.chat_repository == "json":
        return JsonChatRepository(base_dir=settings.chat_storage_dir)
    raise RuntimeError("Chat repository is not initialized")


ChatRepositoryDep = Annotated[ChatRepository, Depends(get_repository)]


def get_moderation_service(settings: SettingsDep) -> ModerationService:
    return ModerationService(openai_enabled=settings.moderation_openai_enabled)


ModerationServiceDep = Annotated[ModerationService, Depends(get_moderation_service)]


def get_chat_service(
    repository: ChatRepositoryDep,
    llm_service: LLMServiceDep,
    moderation_service: ModerationServiceDep,
    settings: SettingsDep,
    request: Request,
) -> ChatService:
    return ChatService(
        repository=repository,
        llm_service=llm_service,
        moderation_service=moderation_service,
        context_window=settings.chat_context_window,
        rag_service=getattr(request.app.state, "rag_service", None),
        max_tokens=settings.chat_max_tokens,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
