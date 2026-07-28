from __future__ import annotations

import asyncio
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics import discrete_metric
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.core.config import Settings
from app.services.embeddings import EmbeddingService


class CitationDecision(BaseModel):
    value: Literal["yes", "no"]
    reason: str


def build_eval_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.eval_judge_api_key.get_secret_value(),
        base_url=settings.eval_judge_base_url,
        timeout=settings.eval_request_timeout,
        max_retries=settings.llm.max_retries,
    )


def build_judge(settings: Settings, client: AsyncOpenAI) -> Any:
    if settings.eval_judge_provider == "local":
        import instructor
        from ragas.llms import InstructorLLM
        from ragas.llms.base import InstructorModelArgs

        patched_client = instructor.from_openai(
            client,
            mode=instructor.Mode.JSON_SCHEMA,
        )
        return InstructorLLM(
            client=patched_client,
            model=settings.eval_judge_model,
            provider="openai",
            model_args=InstructorModelArgs(
                temperature=0.01,
                top_p=0.1,
                max_tokens=settings.eval_judge_max_tokens,
            ),
        )
    return llm_factory(
        settings.eval_judge_model,
        provider="openai",
        client=client,
    )


class LocalE5Embeddings(BaseRagasEmbedding):
    """RAGAS adapter over the project's local cached E5 embedding service."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.model = settings.eval_embedding_model
        self._service = EmbeddingService(
            model_name=self.model,
            batch_size=settings.embedding_batch_size,
            cache_dir=settings.embedding_cache_dir,
        )

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self._service.embed_texts([text])[0]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)

    def embed_texts(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        return self._service.embed_texts(texts)

    async def aembed_texts(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_texts, texts)

    def close(self) -> None:
        self._service.close()


def build_evaluator_embeddings(
    settings: Settings,
    client: AsyncOpenAI | None = None,
) -> LocalE5Embeddings:
    return LocalE5Embeddings(settings)


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
