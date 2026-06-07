import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


class BatchChatRequest(BaseModel):
    requests: list[ChatRequest] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Список запросов для batch-обработки",
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Получить полный ответ LLM",
    responses={
        200: {"description": "Ответ успешно получен"},
        422: {"description": "Ошибка валидации запроса"},
        429: {"description": "Превышен лимит запросов к LLM"},
        502: {"description": "Ошибка LLM-провайдера"},
        504: {"description": "Таймаут LLM-провайдера"},
    },
)
async def chat(
    request: ChatRequest,
    service: LLMServiceDep,
) -> ChatResponse:
    return await service.complete(request)


@router.post(
    "/chat/stream",
    summary="Получить потоковый ответ LLM через SSE",
    responses={
        200: {"description": "Потоковый ответ успешно начат"},
        422: {"description": "Ошибка валидации запроса"},
        429: {"description": "Превышен лимит запросов к LLM"},
        502: {"description": "Ошибка LLM-провайдера"},
        504: {"description": "Таймаут LLM-провайдера"},
    },
)
async def chat_stream(
    request: ChatRequest,
    service: LLMServiceDep,
) -> StreamingResponse:
    async def event_generator():
        async for delta in service.stream(request):
            if delta.content is not None:
                yield f"data: {delta.content}\n\n"

            if delta.usage is not None:
                yield f"data: {delta.model_dump_json(exclude_none=True)}\n\n"

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
    summary="Выполнить несколько LLM-запросов",
    responses={
        200: {"description": "Batch-запрос обработан"},
        422: {"description": "Ошибка валидации запроса"},
        429: {"description": "Превышен лимит запросов к LLM"},
        502: {"description": "Ошибка LLM-провайдера"},
        504: {"description": "Таймаут LLM-провайдера"},
    },
)
async def chat_batch(
    request: BatchChatRequest,
    service: LLMServiceDep,
):
    results = await service.batch(request.requests)
    return {
        "results": [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in results
        ]
    }