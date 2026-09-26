from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import InfrastructureError
from app.schemas.rag import RAGQueryRequest, RAGQueryResponse
from app.security.public_budget import (
    PublicGenerationBudget,
    get_public_generation_budget,
)


class RAGAnswerService(Protocol):
    async def answer(self, question: str) -> dict:
        ...


router = APIRouter(prefix="/rag", tags=["rag"])


def get_rag_service(request: Request) -> RAGAnswerService:
    return request.app.state.rag_service


RAGServiceDep = Annotated[RAGAnswerService, Depends(get_rag_service)]
PublicBudgetDep = Annotated[
    PublicGenerationBudget,
    Depends(get_public_generation_budget),
]


@router.post("/query", response_model=RAGQueryResponse)
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
