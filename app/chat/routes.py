import json
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage
from app.chat.media import media_to_part
from app.chat.service import ModerationBlockedError
from app.services.rag import sanitize_sse_payload


router = APIRouter(prefix="/chats", tags=["chat-history"])


class CreateChatIn(BaseModel):
    owner_external_id: str = Field(..., min_length=1)
    interface: str = Field(..., min_length=1)
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    chat_id: UUID


class MessageIn(BaseModel):
    content: str = Field(..., min_length=1)


class FeedbackIn(BaseModel):
    value: str = Field(pattern="^(up|down)$")


class FeedbackOut(BaseModel):
    status: str
    duplicate: bool = False


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
    service: ChatServiceDep,
    content: str = Form(..., min_length=1),
    media: UploadFile | None = File(None),
) -> StreamingResponse:
    media_refs = None

    if media is not None:
        media_part = await media_to_part(
            media,
            llm_client=service.llm_service.openai,
        )

        media_refs = {
            "mime": media.content_type,
            "size": media.size,
            "filename": media.filename,
            "part": media_part,
        }

    try:
        await service.check_user_content(content)
    except ModerationBlockedError as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "moderation_blocked",
                "categories": exc.result.categories,
                "reasons": exc.result.reasons,
                "blocked_by": exc.result.blocked_by,
            },
        ) from exc

    async def event_generator():
        try:
            async for chunk in service.send_message(
                chat_id=chat_id,
                user_content=content,
                media_refs=media_refs,
            ):
                payload = {
                    "type": "token",
                    "delta": chunk,
                }

                yield (
                    "data: "
                    + sanitize_sse_payload(payload)
                    + "\n\n"
                )

            sources_payload = {
                "type": "sources",
                "sources": service.last_sources,
                **service.last_rag_meta,
            }
            yield (
                "event: sources\n"
                "data: "
                + sanitize_sse_payload(sources_payload)
                + "\n\n"
            )

            done_payload = {
                "type": "done",
                "message_id": (
                    str(service.last_assistant_message_id)
                    if service.last_assistant_message_id
                    else None
                ),
            }

            yield (
                "data: "
                + sanitize_sse_payload(done_payload)
                + "\n\n"
            )

        except ValueError as exc:
            payload = {
                "type": "error",
                "message": str(exc),
            }

            yield (
                "data: "
                + json.dumps(payload, ensure_ascii=False)
                + "\n\n"
            )
            yield 'data: {"type":"done"}\n\n'

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


@router.post("/{chat_id}/messages/{message_id}/feedback", response_model=FeedbackOut)
async def save_feedback(
    chat_id: UUID,
    message_id: UUID,
    request: FeedbackIn,
    service: ChatServiceDep,
) -> FeedbackOut:
    try:
        feedback = await service.save_feedback(
            chat_id=chat_id,
            message_id=message_id,
            value=request.value,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FeedbackOut(
        status="ok",
        duplicate=feedback is None,
    )


@router.post("/{chat_id}/handoff")
async def set_handoff(
    chat_id: UUID,
    service: ChatServiceDep,
) -> dict[str, str]:
    chat = await service.set_handoff_status(
        chat_id=chat_id,
        status="paused_for_human",
    )

    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {"status": chat.handoff_status}
