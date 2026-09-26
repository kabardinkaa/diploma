from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage
from app.chat.media import media_to_part
from app.chat.service import ModerationBlockedError
from app.admin.deps import require_internal_in_public
from app.core.sse import sse_error_events
from app.core.config import get_settings
from app.core.uploads import normalize_safe_filename
from app.security.public_budget import get_public_generation_budget
from app.services.rag import sanitize_sse_payload
from app.services.retention import protect_retained_state
from app.schemas.openapi import error_response, http_error_response, sse_response


router = APIRouter(
    prefix="/chats",
    tags=["Chat History"],
    dependencies=[Depends(require_internal_in_public)],
)


class CreateChatIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "owner_external_id": "telegram-user-123",
                "interface": "telegram",
            }
        }
    )

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


@router.post(
    "",
    response_model=CreateChatOut,
    summary="Создать persistent chat",
    description=(
        "Creates a server-side chat for a trusted client. In public deployments "
        "this service-to-service API requires `X-Internal-Token`."
    ),
    responses={
        403: http_error_response(
            "Missing or invalid internal token in public mode",
            detail="Internal token required",
        ),
        422: error_response(
            "Request validation failed",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
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


@router.get(
    "/{chat_id}",
    response_model=Chat,
    summary="Получить persistent chat",
    description="Returns trusted-client chat metadata by ID.",
    responses={
        403: http_error_response(
            "Missing or invalid internal token in public mode",
            detail="Internal token required",
        ),
        404: http_error_response("Chat not found", detail="Chat not found"),
        422: error_response(
            "Invalid chat UUID",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
async def get_chat(
    chat_id: UUID,
    service: ChatServiceDep,
) -> Chat:
    chat = await service.get_chat(chat_id)

    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return chat


@router.get(
    "/{chat_id}/messages",
    response_model=list[ChatMessage],
    summary="Получить историю сообщений",
    description="Returns a bounded persistent message history for a trusted client.",
    responses={
        403: http_error_response(
            "Missing or invalid internal token in public mode",
            detail="Internal token required",
        ),
        404: http_error_response("Chat not found", detail="Chat not found"),
        422: error_response(
            "Invalid chat UUID or limit",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
async def list_messages(
    chat_id: UUID,
    service: ChatServiceDep,
    limit: int = 50,
) -> list[ChatMessage]:
    chat = await service.get_chat(chat_id)

    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return await service.list_messages(chat_id=chat_id, limit=limit)


@router.post(
    "/{chat_id}/messages",
    response_class=StreamingResponse,
    summary="Отправить text/media message",
    description=(
        "Trusted-client multipart SSE endpoint used by the Telegram bot. Supports "
        "bounded text, image, PDF, DOCX, voice, and audio input. Events include "
        "token frames, a `sources` event, and terminal `done`; failures after stream "
        "start use safe `error` then `done` events."
    ),
    responses={
        200: sse_response(
            "SSE token stream followed by sources and done",
            "data: {\"type\":\"token\",\"delta\":\"VPN\"}\n\n"
            "event: sources\ndata: {\"type\":\"sources\",\"sources\":[]}\n\n"
            "data: {\"type\":\"done\",\"message_id\":\"...\"}\n\n",
        ),
        400: error_response(
            "Unsafe media filename",
            code="invalid_document",
            message="Недопустимое имя файла",
        ),
        403: http_error_response(
            "Internal authentication failed or moderation blocked the message",
            detail="Internal token required",
        ),
        413: error_response(
            "Media exceeds the configured size limit",
            code="payload_too_large",
            message="Размер файла превышает допустимый лимит",
        ),
        415: error_response(
            "Unsupported media type",
            code="unsupported_file_type",
            message="Тип файла не поддерживается",
        ),
        422: error_response(
            "Invalid multipart payload or media content",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
        429: error_response(
            "Public generation quota is exhausted before streaming starts",
            code="public_quota_exhausted",
            message="Public demo generation quota is exhausted",
        ),
        503: error_response(
            "Public generation is disabled before streaming starts",
            code="public_generation_disabled",
            message="Public demo generation is disabled",
        ),
    },
)
async def send_message(
    chat_id: UUID,
    request: Request,
    service: ChatServiceDep,
    content: str = Form(..., min_length=1),
    media: UploadFile | None = File(None),
) -> StreamingResponse:
    budget = get_public_generation_budget(request)
    media_refs = None

    if media is not None:
        safe_media_name = normalize_safe_filename(media.filename or "upload.bin")
        media_part = await media_to_part(
            media,
            llm_client=service.llm_service.openai,
            max_bytes=get_settings().chat_media_max_bytes,
        )

        media_refs = {
            "mime": media.content_type,
            "size": media.size,
            "filename": safe_media_name,
            "part": media_part,
        }

    try:
        await budget.consume()
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
            async with protect_retained_state(request, f"chat:{chat_id}"):
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

        except Exception as exc:
            for event in sse_error_events(exc):
                yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete(
    "/{chat_id}/messages",
    summary="Очистить историю chat",
    description="Soft-deletes message history for a trusted-client chat.",
    responses={
        403: http_error_response(
            "Missing or invalid internal token in public mode",
            detail="Internal token required",
        ),
        404: http_error_response("Chat not found", detail="Chat not found"),
        422: error_response(
            "Invalid chat UUID",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
async def clear_messages(
    chat_id: UUID,
    service: ChatServiceDep,
) -> dict[str, str]:
    chat = await service.get_chat(chat_id)

    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    await service.clear_history(chat_id)

    return {"status": "ok"}


@router.post(
    "/{chat_id}/messages/{message_id}/feedback",
    response_model=FeedbackOut,
    summary="Сохранить feedback",
    description="Stores idempotent up/down feedback for a trusted-client response.",
    responses={
        403: http_error_response(
            "Missing or invalid internal token in public mode",
            detail="Internal token required",
        ),
        404: http_error_response("Chat or message not found", detail="Chat not found"),
        422: error_response(
            "Invalid UUID or feedback value",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
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


@router.post(
    "/{chat_id}/handoff",
    summary="Передать chat оператору",
    description="Pauses automated processing for the trusted-client operator flow.",
    responses={
        403: http_error_response(
            "Missing or invalid internal token in public mode",
            detail="Internal token required",
        ),
        404: http_error_response("Chat not found", detail="Chat not found"),
        422: error_response(
            "Invalid chat UUID",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
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
