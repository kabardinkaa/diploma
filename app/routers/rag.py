from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import InfrastructureError
from app.schemas.rag import RAGQueryRequest, RAGQueryResponse
from app.schemas.openapi import error_response
from app.security.public_budget import (
    PublicGenerationBudget,
    get_public_generation_budget,
)


class RAGAnswerService(Protocol):
    async def answer(self, question: str) -> dict:
        ...


router = APIRouter(prefix="/rag", tags=["RAG"])


def get_rag_service(request: Request) -> RAGAnswerService:
    return request.app.state.rag_service


RAGServiceDep = Annotated[RAGAnswerService, Depends(get_rag_service)]
PublicBudgetDep = Annotated[
    PublicGenerationBudget,
    Depends(get_public_generation_budget),
]


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Задать вопрос корпоративной базе знаний",
    description=(
        "Retrieves evidence from the production `corporate_rag` collection and "
        "generates a bounded answer with citations. Empty or low-confidence "
        "retrieval returns a normal refusal without provider generation. Qdrant "
        "outages are reported as a safe 503, not as an empty result."
    ),
    responses={
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
            "LLM provider unavailable after successful retrieval",
            code="llm_error",
            message="Ошибка LLM-сервиса",
        ),
        503: error_response(
            "RAG infrastructure unavailable or public generation disabled",
            code="rag_unavailable",
            message="Сервис поиска по базе знаний временно недоступен",
        ),
        504: error_response(
            "LLM provider timeout after successful retrieval",
            code="llm_timeout",
            message="LLM-провайдер не ответил за отведённое время",
        ),
    },
)
async def query_rag(
    payload: RAGQueryRequest,
    service: RAGServiceDep,
    budget: PublicBudgetDep,
) -> RAGQueryResponse | JSONResponse:
    await budget.consume()
    try:
        result = await service.answer(payload.question)
    except InfrastructureError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": exc.code, "message": exc.message}},
        )
    return RAGQueryResponse.model_validate(result)
