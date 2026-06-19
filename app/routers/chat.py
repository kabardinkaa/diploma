import json

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Message

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
    request: Request,
    service: LLMServiceDep,
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

    return await service.complete(chat_request)


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