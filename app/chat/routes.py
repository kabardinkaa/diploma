from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage


router = APIRouter(prefix="/chats", tags=["chat-history"])


class CreateChatIn(BaseModel):
    owner_external_id: str = Field(..., min_length=1)
    interface: str = Field(..., min_length=1)
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    chat_id: UUID


class MessageIn(BaseModel):
    content: str = Field(..., min_length=1)


@router.post("", response_model=CreateChatOut)
async def create_chat(
    request: CreateChatIn,
    service: ChatServiceDep,
) -> CreateChatOut:
    chat = await service.create_chat(
        owner_external_id=request.owner_external_id,
        interface=request.interface,
        system_prompt=request.system_prompt,
    )

    return CreateChatOut(chat_id=chat.id)


@router.get("/{chat_id}", response_model=Chat)
async def get_chat(
    chat_id: UUID,
    service: ChatServiceDep,
) -> Chat:
    chat = await service.get_chat(chat_id)

    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return chat


@router.get("/{chat_id}/messages", response_model=list[ChatMessage])
async def list_messages(
    chat_id: UUID,
    service: ChatServiceDep,
    limit: int = 50,
) -> list[ChatMessage]:
    chat = await service.get_chat(chat_id)

    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return await service.list_messages(chat_id=chat_id, limit=limit)


@router.post("/{chat_id}/messages")
async def send_message(
    chat_id: UUID,
    request: MessageIn,
    service: ChatServiceDep,
) -> StreamingResponse:
    async def event_generator():
        try:
            async for chunk in service.send_message(
                chat_id=chat_id,
                user_content=request.content,
            ):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except ValueError as exc:
            yield f"data: ERROR: {exc}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/{chat_id}/messages")
async def clear_messages(
    chat_id: UUID,
    service: ChatServiceDep,
) -> dict[str, str]:
    chat = await service.get_chat(chat_id)

    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    await service.clear_history(chat_id)

    return {"status": "ok"}