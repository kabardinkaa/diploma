from typing import Annotated

from fastapi import Depends

from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repository import ChatRepository
from app.chat.service import ChatService
from app.core.config import Settings, get_settings
from app.deps.providers import LLMServiceDep


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_repository(settings: SettingsDep) -> ChatRepository:
    if settings.chat_repository == "json":
        return JsonChatRepository(base_dir=settings.chat_storage_dir)

    if settings.chat_repository == "postgres":
        raise NotImplementedError("Postgres repository will be added in the next step")

    raise ValueError(f"Unknown CHAT_REPOSITORY value: {settings.chat_repository}")


ChatRepositoryDep = Annotated[ChatRepository, Depends(get_repository)]


def get_chat_service(
    repository: ChatRepositoryDep,
    llm_service: LLMServiceDep,
    settings: SettingsDep,
) -> ChatService:
    return ChatService(
        repository=repository,
        llm_service=llm_service,
        context_window=settings.chat_context_window,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]