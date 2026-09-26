import json

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.admin.deps import AdminDep
from app.core.sse import sse_error_events
from app.deps.providers import SettingsDep
from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Message
from app.schemas.openapi import ErrorResponse, error_response, http_error_response, sse_response
from app.security.public_budget import (
    PublicGenerationBudget,
    get_public_generation_budget,
)
from typing import Annotated
from fastapi import Depends

router = APIRouter()
PublicBudgetDep = Annotated[
    PublicGenerationBudget,
    Depends(get_public_generation_budget),
]


class BatchChatRequest(BaseModel):
    requests: list[ChatRequest] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Список запросов для batch-обработки",
    )


class BatchChatResponse(BaseModel):
    results: list[ChatResponse | ErrorResponse] = Field(
        description=(
            "One result per request. Provider failures are represented as safe "
            "per-item error objects while the batch response remains HTTP 200."
        )
    )


def _apply_server_limits(request: ChatRequest, settings: SettingsDep) -> ChatRequest:
    return request.model_copy(
        update={
            "model": settings.llm.default_model,
            "max_tokens": settings.chat_max_tokens,
        }
    )


@router.post(
    "/chat",
    tags=["Chat"],
    response_model=ChatResponse,
    summary="Получить полный ответ LLM",
    description=(
        "Public bounded chat completion. The deployment selects the model and "
        "replaces the compatibility `max_tokens` value with `CHAT_MAX_TOKENS`. "
        "JSON is recommended; a plain-text body remains supported for backward "
        "compatibility. This operation consumes public generation budget."
    ),
    responses={
        200: {"description": "Ответ успешно получен"},
        422: error_response(
            "Request validation failed",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
        429: error_response(
            "Public quota, request-rate, or concurrency limit reached",
            code="public_quota_exhausted",
            message="Public demo generation quota is exhausted",
        ),
        502: error_response(
            "LLM provider unavailable or rejected authentication",
            code="llm_auth",
            message="Ошибка авторизации у LLM-провайдера",
        ),
        503: error_response(
            "Public generation is disabled",
            code="public_generation_disabled",
            message="Public demo generation is disabled",
        ),
        504: error_response(
            "LLM provider timeout",
            code="llm_timeout",
            message="LLM-провайдер не ответил за отведённое время",
        ),
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ChatRequest"}
                },
                "text/plain": {
                    "schema": {"type": "string", "maxLength": 4000},
                    "example": "Как подключиться к корпоративному VPN?",
                },
            },
        }
    },
)
async def chat(
    request: Request,
    service: LLMServiceDep,
    settings: SettingsDep,
    budget: PublicBudgetDep,
) -> ChatResponse:
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("text/plain"):
        prompt = (await request.body()).decode("utf-8", errors="replace").strip()
        chat_request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content=prompt,
                )
            ],
            temperature=0,
            max_tokens=300,
        )
    else:
        try:
            payload = await request.json()
            chat_request = ChatRequest.model_validate(payload)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc

    await budget.consume()
    return await service.complete(_apply_server_limits(chat_request, settings))


@router.post(
    "/chat/stream",
    tags=["Chat"],
    response_class=StreamingResponse,
    summary="Получить потоковый ответ LLM через SSE",
    description=(
        "Public `text/event-stream` chat. Data frames contain token text or usage "
        "JSON and the successful stream ends with `data: [DONE]`. If an error "
        "occurs after streaming starts, the endpoint emits safe `error` and `done` "
        "events without raw provider details. Server-selected model and token cap apply."
    ),
    responses={
        200: sse_response(
            "SSE token stream; provider errors after start are safe SSE events",
            "data: Для подключения\n\ndata: откройте VPN-клиент\n\ndata: [DONE]\n\n",
        ),
        422: error_response(
            "Request validation failed before streaming",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
        429: error_response(
            "Request-rate or concurrency limit reached before streaming",
            code="rate_limit_exceeded",
            message="Too many expensive requests. Try again later.",
        ),
    },
)
async def chat_stream(
    request: ChatRequest,
    service: LLMServiceDep,
    settings: SettingsDep,
    budget: PublicBudgetDep,
) -> StreamingResponse:
    request = _apply_server_limits(request, settings)

    async def event_generator():
        try:
            await budget.consume()
            async for delta in service.stream(request):
                if delta.content is not None:
                    yield f"data: {delta.content}\n\n"

                if delta.usage is not None:
                    yield f"data: {delta.model_dump_json(exclude_none=True)}\n\n"
        except Exception as exc:
            for event in sse_error_events(exc):
                yield event
            return

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/chat/batch",
    tags=["Admin"],
    response_model=BatchChatResponse,
    summary="Выполнить несколько LLM-запросов",
    description=(
        "Administrative batch generation protected by `X-Admin-Token`. Model and "
        "token limits remain server-controlled for every item. Individual provider "
        "failures are returned as safe error objects in the `results` array."
    ),
    responses={
        200: {"description": "Batch-запрос обработан"},
        403: http_error_response(
            "Missing or invalid admin token",
            detail="Admin token required",
        ),
        422: error_response(
            "Request validation failed",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
async def chat_batch(
    request: BatchChatRequest,
    service: LLMServiceDep,
    settings: SettingsDep,
    _: AdminDep,
):
    results = await service.batch(
        [_apply_server_limits(item, settings) for item in request.requests]
    )
    return {
        "results": [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in results
        ]
    }
