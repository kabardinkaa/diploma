from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, Request

from app.schemas.rag import RAGQueryRequest, RAGQueryResponse


class RAGAnswerService(Protocol):
    async def answer(self, question: str) -> dict:
        ...


router = APIRouter(prefix="/rag", tags=["rag"])


def get_rag_service(request: Request) -> RAGAnswerService:
    return request.app.state.rag_service


RAGServiceDep = Annotated[RAGAnswerService, Depends(get_rag_service)]


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(
    payload: RAGQueryRequest,
    service: RAGServiceDep,
) -> RAGQueryResponse:
    return RAGQueryResponse.model_validate(await service.answer(payload.question))
