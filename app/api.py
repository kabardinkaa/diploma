import logging
from functools import lru_cache
from typing import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.llm.async_client import AsyncLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="Diploma AI Assistant API",
    description="API для ИИ-ассистента внутренней техподдержки",
    version="3.3.0",
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Сообщение пользователя")


class ChatResponse(BaseModel):
    answer: str


@lru_cache
def get_llm_client() -> AsyncLLMClient:
    """
    Создаём клиента лениво.

    Так приложение можно импортировать без немедленной проверки API-ключа,
    а сам клиент создаётся только при первом запросе.
    """
    return AsyncLLMClient(
    concurrency=5,
    sdk_timeout=60,
    logic_timeout=60,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Обычный endpoint: возвращает полный ответ после завершения генерации.
    """
    client = get_llm_client()
    answer = await client.complete(request.prompt)
    return ChatResponse(answer=answer)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    """
    SSE endpoint: отдаёт ответ по частям, по мере генерации токенов.
    """

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        client = get_llm_client()

        async for token in client.stream_chat(request.prompt):
            yield {
                "event": "token",
                "data": token,
            }

        yield {
            "event": "done",
            "data": "[DONE]",
        }

    return EventSourceResponse(event_generator())