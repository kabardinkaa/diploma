from __future__ import annotations

from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import discrete_metric
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.core.config import Settings


class CitationDecision(BaseModel):
    value: Literal["yes", "no"]
    reason: str


def build_eval_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.llm.api_key.get_secret_value(),
        base_url=settings.llm.base_url,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )


def build_judge(settings: Settings, client: AsyncOpenAI) -> Any:
    return llm_factory(
        settings.eval_judge_model,
        provider=settings.eval_judge_provider,
        client=client,
    )


def build_evaluator_embeddings(
    settings: Settings,
    client: AsyncOpenAI,
) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        client=client,
        model=settings.eval_embedding_model,
    )


def make_has_citation(llm: Any) -> Any:
    @discrete_metric(name="has_citation", allowed_values=["yes", "no"])
    async def has_citation(response: str) -> str:
        """Check whether an answer contains an explicit source citation."""
        prompt = (
            "Содержит ли ответ ссылку на источник: валидный маркер [1], [2] "
            "или [doc_id], имя файла либо явную формулировку «согласно ...»? "
            "Для этого RAG главным признаком считай numbered citation marker [N].\n\n"
            f"Ответ:\n{response}"
        )
        decision = await llm.agenerate(
            prompt,
            response_model=CitationDecision,
        )
        return decision.value

    return has_citation


def build_metrics(llm: Any, embeddings: Any) -> dict[str, Any]:
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "has_citation": make_has_citation(llm),
    }


async def eval_row(
    metrics: dict[str, Any],
    *,
    user_input: str,
    reference: str,
    response: str,
    retrieved_contexts: list[str],
) -> dict[str, float | None]:
    results = {
        "faithfulness": await metrics["faithfulness"].ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        ),
        "answer_relevancy": await metrics["answer_relevancy"].ascore(
            user_input=user_input,
            response=response,
        ),
        "context_precision": await metrics["context_precision"].ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        ),
        "context_recall": await metrics["context_recall"].ascore(
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        ),
        "has_citation": await metrics["has_citation"].ascore(response=response),
    }
    values: dict[str, float | None] = {}
    for name, result in results.items():
        if name == "has_citation":
            values[name] = (
                1.0 if result.value == "yes" else 0.0 if result.value == "no" else None
            )
        else:
            values[name] = (
                float(result.value) if result.value is not None else None
            )
    return values
